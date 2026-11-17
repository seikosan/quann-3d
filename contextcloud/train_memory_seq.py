#!/usr/bin/env python3
"""
Train ContextCloudLM with Sequential Chunk Processing.

Key Change:
- Process chunks SEQUENTIALLY with cross-attention between chunks
- Hidden state passes from chunk to chunk (like RNN)
- Memory bank updated sequentially with chunk states
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SequentialChunkEncoder(nn.Module):
    """
    Encode chunks sequentially with cross-attention.
    Each chunk attends to previous chunk's hidden state.
    """
    
    def __init__(self, vocab_size, d_model=128, d_cloud=256, max_tokens=256,
                 nhead=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        
        # Single transformer layer for chunk encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True
        )
        self.chunk_encoder = encoder_layer
        
        # Cross-attention: chunk attends to previous hidden state
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        
        self.cloud_proj = nn.Linear(d_model, d_cloud)
    
    def forward(self, tokens, mask=None, prev_hidden=None):
        """
        Args:
            tokens: (B, MaxTokens)
            mask: (B, MaxTokens)
            prev_hidden: (B, Dim) - previous chunk's hidden state
        
        Returns:
            chunk_state: (B, Dim)
            hidden: (B, Dim) - for next chunk
        """
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        
        # Apply cross-attention if we have previous hidden state
        if prev_hidden is not None:
            # Use chunk tokens as query, prev_hidden as key/value
            q = self.cross_attn_q(x)  # (B, MaxTokens, Dim)
            k = self.cross_attn_k(prev_hidden.unsqueeze(1))  # (B, 1, Dim)
            v = self.cross_attn_v(prev_hidden.unsqueeze(1))  # (B, 1, Dim)
            
            # Cross-attention
            scores = q @ k.transpose(1, 2) / (x.size(-1) ** 0.5)  # (B, MaxTokens, 1)
            attn = F.softmax(scores, dim=-1)
            cross_output = attn @ v  # (B, MaxTokens, Dim)
            
            # Combine with current input
            x = x + cross_output  # (B, MaxTokens, Dim)
        
        # Encode chunk
        x = self.chunk_encoder(x)
        
        # Get chunk state (mean pooling)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).float()
            chunk_state = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            chunk_state = x.mean(dim=1)
        
        return chunk_state, chunk_state


class MemoryBank(nn.Module):
    """Memory bank with sequential updates."""
    
    def __init__(self, num_memory=32, dim=128):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_memory, dim) * 0.02)
        self.num_memory = num_memory
        self.dim = dim
        
        # Read operation
        self.read_q = nn.Linear(dim, dim)
        self.read_k = nn.Linear(dim, dim)
        self.read_v = nn.Linear(dim, dim)
        
        # Write operation with gating
        self.write_gate = nn.Linear(dim, num_memory)
        self.write_update = nn.Linear(dim, dim)
    
    def read(self, query):
        """Read from memory using query vector."""
        q = self.read_q(query)  # (B, Dim)
        k = self.read_k(self.memory)  # (NumMemory, Dim)
        v = self.read_v(self.memory)  # (NumMemory, Dim)
        
        scores = q @ k.T  # (B, NumMemory)
        attn = torch.softmax(scores, dim=-1)
        
        memory_read = attn @ v  # (B, Dim)
        return memory_read
    
    def write(self, chunk_state, memory):
        """Write chunk state to memory (sequential update)."""
        gate = torch.sigmoid(self.write_gate(chunk_state))  # (B, NumMemory)
        new_values = self.write_update(chunk_state)  # (B, Dim)
        
        gate_exp = gate.unsqueeze(-1)  # (B, NumMemory, 1)
        new_values_exp = new_values.unsqueeze(1)  # (B, 1, Dim)
        
        # Sequential update: memory accumulates information
        updated_memory = (1 - gate_exp) * memory + gate_exp * new_values_exp
        
        return updated_memory


class SequentialMemoryModel(nn.Module):
    """
    Sequential chunk processing with memory network.
    Chunks are processed one by one, with cross-attention and memory.
    """
    
    def __init__(self, vocab_size, d_model=128, d_sentence=128, max_tokens=256,
                 num_chunks=4, num_memory=32, dropout=0.2):
        super().__init__()
        self.num_chunks = num_chunks
        
        # Sequential chunk encoder with cross-attention
        self.chunk_encoder = SequentialChunkEncoder(
            vocab_size, d_model=d_model, d_cloud=d_sentence,
            max_tokens=max_tokens, dropout=dropout
        )
        
        # Memory bank
        self.memory = MemoryBank(num_memory=num_memory, dim=d_sentence)
        
        # Final prediction head
        self.final_head = nn.Sequential(
            nn.Linear(d_sentence * 2, d_sentence),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_sentence, vocab_size)
        )
    
    def forward(self, chunk_tokens, chunk_masks):
        """
        Process chunks SEQUENTIALLY.
        
        Args:
            chunk_tokens: (B, NumChunks, MaxTokens)
            chunk_masks: (B, NumChunks, MaxTokens)
        
        Returns:
            logits: (B, Vocab)
        """
        B, NC, MT = chunk_tokens.shape
        
        # Initialize memory and hidden state
        memory = self.memory.memory.unsqueeze(0).expand(B, -1, -1).clone()
        prev_hidden = None
        
        chunk_states = []
        
        for c in range(NC):
            chunk = chunk_tokens[:, c, :]  # (B, MaxTokens)
            mask = chunk_masks[:, c, :]
            
            # Encode chunk with cross-attention to previous hidden
            chunk_state, prev_hidden = self.chunk_encoder(chunk, mask, prev_hidden)
            
            # Read from memory
            memory_read = self.memory.read(chunk_state)
            
            # Combine
            combined = torch.cat([chunk_state, memory_read], dim=-1)
            chunk_states.append(combined)
            
            # Write to memory (sequential update)
            memory = self.memory.write(chunk_state, memory)
        
        # Use last chunk's combined state
        final_state = chunk_states[-1]
        return self.final_head(final_state)


class SequentialChunkDataset(Dataset):
    """Dataset with sequential chunks."""
    
    def __init__(self, filepath, chunk_size=5, max_tokens=256, num_chunks=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.num_chunks = num_chunks
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []
        
        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip().split() for line in f if line.strip()]
        
        for word in set(word for sent in sentences for word in sent):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        for i in range(0, max(1, len(sentences) - chunk_size * (num_chunks - 1)), chunk_size):
            sample = []
            for c in range(num_chunks):
                start = i + c * chunk_size
                chunk = sentences[start:start + chunk_size]
                tokens = [w for sent in chunk for w in sent][:max_tokens]
                sample.append(tokens)
            if all(len(s) > 0 for s in sample):
                self.data.append(sample)
        
        logger.info("Vocab size: %d, Samples: %d", len(self.vocab), len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        token_ids = []
        for chunk in sample:
            ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in chunk]
            ids = ids + [0] * (self.max_tokens - len(ids))
            token_ids.append(ids)
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    max_tokens = max(len(chunk) for sample in batch for chunk in sample)
    batch_chunks = len(batch[0])
    chunks = torch.zeros(len(batch), batch_chunks, max_tokens, dtype=torch.long)
    for i, sample in enumerate(batch):
        for j, chunk in enumerate(sample):
            chunks[i, j, :len(chunk)] = chunk
    masks = chunks.ne(0)
    return chunks, masks


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (chunks, masks) in enumerate(dataloader):
        chunks = chunks.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(chunks, masks)
        
        # Target: last token of last chunk
        targets = []
        for i in range(chunks.size(0)):
            last_chunk = chunks[i, -1, :]
            mask = masks[i, -1, :]
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) > 0:
                targets.append(last_chunk[valid_indices[-1]].item())
            else:
                targets.append(0)
        target = torch.tensor(targets).to(device)
        
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Loss: %.4f", batch_idx, len(dataloader), loss.item())
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for chunks, masks in dataloader:
            chunks = chunks.to(device)
            masks = masks.to(device)
            
            logits = model(chunks, masks)
            
            targets = []
            for i in range(chunks.size(0)):
                last_chunk = chunks[i, -1, :]
                mask = masks[i, -1, :]
                valid_indices = torch.where(mask)[0]
                if len(valid_indices) > 0:
                    targets.append(last_chunk[valid_indices[-1]].item())
                else:
                    targets.append(0)
            target = torch.tensor(targets).to(device)
            
            loss = criterion(logits, target)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    CHUNK_SIZE = 5
    MAX_TOKENS = 256
    NUM_CHUNKS = 4
    NUM_MEMORY = 32
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4
    EPOCHS = 30
    PATIENCE = 5
    DROPOUT = 0.2
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("=== SEQUENTIAL CHUNK PROCESSING ===")
    logger.info("Cross-attention + Sequential Memory + Hidden State Passing")
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return
    
    dataset = SequentialChunkDataset(DATA_FILE, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)
    
    model = SequentialMemoryModel(
        vocab_size=vocab_size,
        d_model=128,
        d_sentence=128,
        max_tokens=MAX_TOKENS,
        num_chunks=NUM_CHUNKS,
        num_memory=NUM_MEMORY,
        dropout=DROPOUT
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, EPOCHS)
        
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, dataloader, criterion, device)
        
        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_memory_seq_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "val_loss": val_loss,
                "num_memory": NUM_MEMORY
            }, checkpoint_path)
            logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
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
