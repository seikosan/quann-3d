#!/usr/bin/env python3
"""Train ContextCloudLM with Global Context Cross-Attention."""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud.dataset import ChunkedVocabDataset, collate_chunks
from contextcloud.task_utils import shift_target_3d, split_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SimpleCloudLM(nn.Module):
    """Simple cloud LM for chunk encoding (standalone, no external deps)."""
    
    def __init__(self, vocab_size, d_model=128, d_cloud_dim=16, d_cloud=256,
                 max_tokens=256, nhead=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        
        # Cloud encoding (simplified)
        self.to_cloud = nn.Linear(d_model, d_cloud_dim)
        self.cloud_encoder = nn.Sequential(
            nn.Linear(d_cloud_dim, d_cloud),
            nn.GELU(),
            nn.Linear(d_cloud, d_cloud)
        )
        
        self.word_head = nn.Linear(d_cloud, vocab_size)
    
    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.encoder(x, src_key_padding_mask=~mask)
        points = self.to_cloud(x)
        cloud_state = self.cloud_encoder(points).mean(dim=1)
        return self.word_head(cloud_state), cloud_state


class CrossAttentionGlobal(nn.Module):
    """Cross-attention from local to global context."""
    
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.ones(1))
    
    def forward(self, local_states, global_context):
        global_expanded = global_context.unsqueeze(1).expand(-1, local_states.size(1), -1)
        attended, _ = self.cross_attn(local_states, global_expanded, global_expanded)
        output = local_states + self.gate * attended
        return self.norm(output)


class GlobalContextModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, d_sentence=256, max_tokens=256, num_chunks=4):
        super().__init__()
        self.num_chunks = num_chunks
        
        # Base encoders for each chunk
        self.chunk_encoders = nn.ModuleList([
            SimpleCloudLM(vocab_size, d_model=d_model, d_cloud=d_sentence, max_tokens=max_tokens)
            for _ in range(num_chunks)
        ])
        
        # Cross-attention to global context
        self.cross_attn_global = CrossAttentionGlobal(dim=d_sentence, num_heads=4)
        
        # Final prediction
        self.final_head = nn.Sequential(
            nn.Linear(d_sentence, d_sentence),
            nn.GELU(),
            nn.Linear(d_sentence, vocab_size)
        )
    
    def forward(self, chunk_tokens, chunk_masks):
        B, NC, MT = chunk_tokens.shape
        
        # Encode each chunk
        chunk_states = []
        for c in range(NC):
            logits_c, state_c = self.chunk_encoders[c](chunk_tokens[:, c, :], chunk_masks[:, c, :])
            chunk_states.append(state_c)
        
        chunk_states = torch.stack(chunk_states, dim=1)  # (B, NC, Dim)
        
        # Global context = mean of all chunk states
        global_context = chunk_states.mean(dim=1)  # (B, Dim)
        
        # Cross-attention
        attended = self.cross_attn_global(chunk_states, global_context)
        
        # Use last attended state
        final_state = attended[:, -1, :]
        return self.final_head(final_state)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (chunks, masks) in enumerate(dataloader):
        chunks = chunks.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        chunks, masks, target = shift_target_3d(chunks, masks)
        logits = model(chunks, masks)

        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
            
            chunks, masks, target = shift_target_3d(chunks, masks)
            logits = model(chunks, masks)

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
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    PATIENCE = 5
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return
    
    # Ayrık 80/10/10 bölme (AGENTS §4.1, seed=42); vocab yalnız train'den
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    train_lines, val_lines, test_lines = split_sentences(lines)
    logger.info("Split: train %d / val %d / test %d (seed=42)",
                len(train_lines), len(val_lines), len(test_lines))

    dataset = ChunkedVocabDataset(train_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS)
    val_dataset = ChunkedVocabDataset(val_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS, vocab=dataset.vocab)
    test_dataset = ChunkedVocabDataset(test_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS, vocab=dataset.vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_chunks)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_chunks)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_chunks)
    
    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)
    
    model = GlobalContextModel(vocab_size=vocab_size, d_model=128, d_sentence=256, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS)
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, EPOCHS)
        
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_global_best.pt")
            torch.save({"model_state_dict": model.state_dict(), "vocab_size": vocab_size, "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
            logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            logger.info("Patience: %d/%d", patience_counter, PATIENCE)
        
        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered")
            break
    
    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Best val loss: %.4f", best_val_loss)

    checkpoint_path = os.path.join(OUTPUT_DIR, "context_global_best.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info("TEST loss (best ckpt, ayrık holdout): %.4f", test_loss)


if __name__ == "__main__":
    main()
