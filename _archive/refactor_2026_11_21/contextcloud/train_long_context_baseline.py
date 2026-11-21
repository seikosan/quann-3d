#!/usr/bin/env python3
"""
Train CloudLM with Long Context (No Chunking).

Key Change:
- Process entire long text as ONE sequence (no chunking)
- Large context window (512+ tokens)
- No memory network, no cross-attention between chunks
- Just pure Transformer with long context
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LongContextModel(nn.Module):
    """
    Simple Transformer LM with long context window.
    No chunking, no memory - just long sequence.
    """
    
    def __init__(self, vocab_size, d_model=256, max_tokens=512, nhead=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        
        # Full transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        self.max_tokens = max_tokens
    
    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        
        x = self.transformer(x)
        return self.head(x)


class LongContextDataset(Dataset):
    """Dataset with long sequences (no chunking)."""
    
    def __init__(self, filepath, seq_length=512, stride=256):
        super().__init__()
        self.seq_length = seq_length
        self.stride = stride
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []
        
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        
        # Flatten all sentences
        all_tokens = [w for sent in lines for w in sent]
        
        # Build vocabulary
        for word in all_tokens:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create sliding window sequences
        for i in range(0, len(all_tokens) - seq_length, stride):
            sample = all_tokens[i:i + seq_length]
            self.data.append(sample)
        
        logger.info("Vocab size: %d, Sequences: %d (len=%d, stride=%d)", 
                   len(self.vocab), len(self.data), seq_length, stride)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in sample]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    # Pad to max length in batch
    max_len = max(len(seq) for seq in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, tokens in enumerate(dataloader):
        tokens = tokens.to(device)
        
        optimizer.zero_grad()
        
        # Shift for next-token prediction
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        # Mask for valid tokens (not padding)
        mask = targets.ne(0)
        
        logits = model(inputs, mask)
        
        # Only compute loss on valid positions
        valid_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss = (valid_loss * mask.reshape(-1)).sum() / mask.float().sum()
        
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
        for tokens in dataloader:
            tokens = tokens.to(device)
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            mask = targets.ne(0)
            
            logits = model(inputs, mask)
            valid_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = (valid_loss * mask.reshape(-1)).sum() / mask.float().sum()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    SEQ_LENGTH = 512  # Long context window
    STRIDE = 256
    D_MODEL = 256
    NHEAD = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    EPOCHS = 30
    PATIENCE = 5
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("=== LONG CONTEXT BASELINE (NO CHUNKING) ===")
    logger.info("Seq Length: %d, D Model: %d, Layers: %d", SEQ_LENGTH, D_MODEL, N_LAYERS)
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return
    
    dataset = LongContextDataset(DATA_FILE, seq_length=SEQ_LENGTH, stride=STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)
    
    model = LongContextModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        max_tokens=SEQ_LENGTH,
        nhead=NHEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
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
            checkpoint_path = os.path.join(OUTPUT_DIR, "long_context_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "val_loss": val_loss
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
