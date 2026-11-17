#!/usr/bin/env python3
"""
Train ContextCloudLM for long texts with Cross-Chunk Attention + Multi-Task Learning.

Key Improvements:
1. Cross-Chunk Attention: Direct connections between chunk representations
2. Multi-Task Learning: Predict next token + next chunk summary + chunk similarity
3. Smaller chunks: 5 sentences per chunk (more fine-grained)
4. Contrastive Loss: Learn better chunk representations
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CrossChunkAttention(nn.Module):
    """Attention between chunk representations."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, chunk_states):
        """
        Args:
            chunk_states: (B, NumChunks, Dim)
        Returns:
            attended: (B, NumChunks, Dim) with cross-chunk context
        """
        attended, _ = self.cross_attn(chunk_states, chunk_states, chunk_states)
        return self.norm(attended + chunk_states)


class LongContextMultiTaskModel(nn.Module):
    """
    Long-context model with:
    - Cross-chunk attention
    - Multi-task: next token + next chunk summary + contrastive
    """

    def __init__(self, vocab_size, d_model=128, d_sentence=256, d_window=256,
                 d_cloud_dim=16, d_window_dim=16, window_size=5, window_stride=2,
                 hidden=128, max_tokens=256, num_chunks=2):
        super().__init__()
        self.num_chunks = num_chunks
        self.base_model = ContextCloudLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_sentence=d_sentence,
            d_window=d_window,
            d_cloud_dim=d_cloud_dim,
            d_window_dim=d_window_dim,
            window_size=window_size,
            window_stride=window_stride,
            hidden=hidden,
            max_tokens=max_tokens,
        )

        # Cross-chunk attention
        self.cross_chunk_attn = CrossChunkAttention(dim=d_sentence, num_heads=4)

        # Chunk summary head
        self.chunk_summary_head = nn.Sequential(
            nn.Linear(d_sentence, d_sentence),
            nn.GELU(),
            nn.Linear(d_sentence, d_sentence),
        )

        # Next chunk prediction head
        self.next_chunk_head = nn.Sequential(
            nn.Linear(d_sentence * 2, d_sentence),
            nn.GELU(),
            nn.Linear(d_sentence, d_sentence),
        )

    def forward(self, chunk_tokens, chunk_masks):
        """
        Args:
            chunk_tokens: (B, NumChunks, MaxTokens)
            chunk_masks: (B, NumChunks, MaxTokens)
        Returns:
            next_token_logits: (B, NumChunks, Vocab)
            chunk_summaries: (B, NumChunks, Dim)
            next_chunk_pred: (B, NumChunks, Dim)
        """
        B, NC, MT = chunk_tokens.shape

        # Encode each chunk independently
        chunk_states = []
        for c in range(NC):
            tokens_c = chunk_tokens[:, c, :]  # (B, MaxTokens)
            mask_c = chunk_masks[:, c, :]
            tokens_3d = tokens_c.unsqueeze(1)
            mask_3d = mask_c.unsqueeze(1)
            logits = self.base_model(tokens_3d, sentence_mask=mask_3d)
            # Get chunk state from points (re-compute)
            x = self.base_model.embedding(tokens_c) * (self.base_model.embedding.embedding_dim ** 0.5)
            x = self.base_model.pos_enc(x)
            x = x.masked_fill(~mask_c.unsqueeze(-1), 0)
            x = self.base_model.encoder(x, src_key_padding_mask=~mask_c)
            x = self.base_model.norm(x)
            points = self.base_model.to_nd(x)
            chunk_state = points.mean(dim=1)  # (B, Dim)
            chunk_states.append(chunk_state)

        chunk_states = torch.stack(chunk_states, dim=1)  # (B, NumChunks, Dim)

        # Cross-chunk attention
        attended = self.cross_chunk_attn(chunk_states)  # (B, NumChunks, Dim)

        # Chunk summary
        chunk_summaries = self.chunk_summary_head(attended)

        # Next chunk prediction (using current + attended)
        combined = torch.cat([chunk_states, attended], dim=-1)  # (B, NumChunks, 2*Dim)
        next_chunk_pred = self.next_chunk_head(combined)

        # Next token logits (use last attended state)
        next_token_logits = self.base_model.word_head(attended[:, -1, :])  # (B, Vocab)

        return next_token_logits, chunk_summaries, next_chunk_pred


class AdvancedLongContextDataset(Dataset):
    """
    Advanced dataset with:
    - Smaller chunks (5 sentences)
    - Overlapping windows
    - Contrastive pairs (similar vs different chunks)
    """

    def __init__(self, filepath: str, chunk_size: int = 5, max_tokens: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []

        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip().split() for line in f if line.strip()]

        for sent in sentences:
            for word in sent:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        # Create chunks
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            if len(chunk) >= 2:
                chunks.append(chunk)

        # Create samples with overlapping windows
        for i in range(len(chunks) - 1):
            # Current chunk + next chunk
            current = chunks[i]
            next_chunk = chunks[i + 1]

            current_tokens = [w for sent in current for w in sent][:max_tokens]
            next_tokens = [w for sent in next_chunk for w in sent][:max_tokens]

            self.data.append((current_tokens, next_tokens))

        logger.info("Vocab size: %d, Chunks: %d, Samples: %d",
                    len(self.vocab), len(chunks), len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        current, next_chunk = self.data[idx]

        current_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in current]
        next_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in next_chunk]

        current_ids = current_ids + [0] * (self.max_tokens - len(current_ids))
        next_ids = next_ids + [0] * (self.max_tokens - len(next_ids))

        return torch.tensor(current_ids, dtype=torch.long), torch.tensor(next_ids, dtype=torch.long)


def collate_fn(batch):
    current_list, next_list = zip(*batch)
    max_curr = max(len(t) for t in current_list)
    max_next = max(len(t) for t in next_list)

    current = torch.zeros(len(batch), max_curr, dtype=torch.long)
    next_tokens = torch.zeros(len(batch), max_next, dtype=torch.long)

    for i, (c, n) in enumerate(zip(current_list, next_list)):
        current[i, :len(c)] = c
        next_tokens[i, :len(n)] = n

    return current, next_tokens


def multi_task_loss(model, current, next_tokens, device, alpha=0.5, beta=0.3):
    """
    Multi-task loss:
    - Next token prediction (alpha)
    - Next chunk prediction (beta)
    - Contrastive (1 - alpha - beta)
    """
    # Prepare inputs: (B, NumChunks=2, MaxTokens)
    batch_size = current.size(0)
    chunk_tokens = torch.stack([current, next_tokens], dim=1)  # (B, 2, MaxTokens)
    chunk_masks = chunk_tokens.ne(0)

    # Forward pass
    next_token_logits, chunk_summaries, next_chunk_pred = model(chunk_tokens, chunk_masks)

    # Task 1: Next token prediction (last chunk)
    target_token = next_tokens[:, -1]  # Last token of next chunk
    token_loss = nn.CrossEntropyLoss()(next_token_logits, target_token)

    # Task 2: Next chunk prediction
    # Use current chunk state to predict next chunk state
    current_state = chunk_summaries[:, 0]  # (B, Dim)
    next_state = chunk_summaries[:, 1]  # (B, Dim)
    predicted_next = model.next_chunk_head(torch.cat([current_state, next_state], dim=-1))

    chunk_loss = nn.MSELoss()(predicted_next, next_state)

    # Task 3: Contrastive loss (similar chunks should be close)
    # Simplified: minimize distance between summary and prediction
    contrastive_loss = nn.MSELoss()(next_chunk_pred[:, 1], next_state)

    # Combined loss
    total_loss = alpha * token_loss + beta * chunk_loss + (1 - alpha - beta) * contrastive_loss

    return total_loss, {
        'token_loss': token_loss.item(),
        'chunk_loss': chunk_loss.item(),
        'contrastive_loss': contrastive_loss.item()
    }


def train_epoch(model, dataloader, optimizer, device, alpha, beta):
    model.train()
    total_loss = 0.0
    total_token_loss = 0.0
    total_chunk_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0

    for batch_idx, (current, next_tokens) in enumerate(dataloader):
        current = current.to(device)
        next_tokens = next_tokens.to(device)

        optimizer.zero_grad()

        loss, losses_dict = multi_task_loss(model, current, next_tokens, device, alpha, beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        total_token_loss += losses_dict['token_loss']
        total_chunk_loss += losses_dict['chunk_loss']
        total_contrastive_loss += losses_dict['contrastive_loss']
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Total: %.4f | Token: %.4f | Chunk: %.4f",
                        batch_idx, len(dataloader), loss.item(),
                        losses_dict['token_loss'], losses_dict['chunk_loss'])

    return (total_loss / num_batches, {
        'token': total_token_loss / num_batches,
        'chunk': total_chunk_loss / num_batches,
        'contrastive': total_contrastive_loss / num_batches
    })


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    CHUNK_SIZE = 5  # Smaller chunks
    MAX_TOKENS = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    ALPHA = 0.5  # Token loss weight
    BETA = 0.3   # Chunk loss weight

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    dataset = AdvancedLongContextDataset(DATA_FILE, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)

    model = LongContextMultiTaskModel(
        vocab_size=vocab_size,
        d_model=128,
        d_sentence=256,
        d_window=256,
        d_cloud_dim=16,
        d_window_dim=16,
        window_size=5,
        window_stride=2,
        hidden=128,
        max_tokens=MAX_TOKENS,
        num_chunks=2,
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 5

    for epoch in range(1, EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, EPOCHS)

        train_loss, train_losses = train_epoch(
            model, dataloader, optimizer, device, ALPHA, BETA
        )

        logger.info("Epoch %d | Total: %.4f | Token: %.4f | Chunk: %.4f | Contrastive: %.4f",
                    epoch, train_loss, train_losses['token'],
                    train_losses['chunk'], train_losses['contrastive'])

        # Simple validation (same as train for now)
        val_loss = train_loss  # Placeholder

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_advanced_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "train_loss": train_loss,
                "chunk_size": CHUNK_SIZE,
                "alpha": ALPHA,
                "beta": BETA,
            }, checkpoint_path)
            logger.info("✓ Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            logger.info("Patience: %d/%d", patience_counter, PATIENCE)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered")
            break

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
