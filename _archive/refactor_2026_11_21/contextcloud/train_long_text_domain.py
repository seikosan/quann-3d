#!/usr/bin/env python3
"""
Train on Long Text Documents (500-800 words).

This addresses the domain gap issue:
- Training: 500-800 word documents
- Testing: 500-800 word documents
"""

import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LongTextModel(nn.Module):
    """Transformer LM optimized for long documents."""
    
    def __init__(self, vocab_size, d_model=512, max_tokens=1024, nhead=8, n_layers=8, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        self.max_tokens = max_tokens
        self.d_model = d_model
    
    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        
        x = self.transformer(x)
        return self.head(x)


class LongTextDataset(Dataset):
    """Dataset for long text documents."""
    
    def __init__(self, filepath, max_tokens=1024):
        super().__init__()
        self.max_tokens = max_tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                if len(words) > 10:  # Skip very short lines
                    self.data.append(words)
        
        # Build vocabulary
        for doc in self.data:
            for word in doc:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        
        logger.info("Vocab size: %d, Documents: %d", len(self.vocab), len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        words = self.data[idx]
        # Truncate or pad to max_tokens
        ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in words[:self.max_tokens]]
        # Pad to max_tokens
        ids = ids + [0] * (self.max_tokens - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    # Already padded in __getitem__
    return torch.stack(batch)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, tokens in enumerate(dataloader):
        tokens = tokens.to(device)
        
        optimizer.zero_grad()
        
        # Next-token prediction
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        mask = targets.ne(0)
        
        logits = model(inputs, mask)
        
        # Masked loss
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
    DATA_FILE = "data/turkish_long_texts.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    MAX_TOKENS = 1024
    D_MODEL = 512
    NHEAD = 8
    N_LAYERS = 8
    DROPOUT = 0.2
    BATCH_SIZE = 4  # Smaller batch for long sequences
    LEARNING_RATE = 3e-4
    EPOCHS = 30
    PATIENCE = 5
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("=== LONG TEXT TRAINING (500-800 WORDS) ===")
    logger.info("Max Tokens: %d, D Model: %d, Layers: %d", MAX_TOKENS, D_MODEL, N_LAYERS)
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return
    
    dataset = LongTextDataset(DATA_FILE, max_tokens=MAX_TOKENS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)
    
    model = LongTextModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        max_tokens=MAX_TOKENS,
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
            checkpoint_path = os.path.join(OUTPUT_DIR, "long_text_domain_best.pt")
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
