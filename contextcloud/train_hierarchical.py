#!/usr/bin/env python3
"""
Hierarchical Training: Short → Long Context Fine-tuning

Stage 1: Train on short sequences (already done, loss ~1.30)
Stage 2: Fine-tune on longer sequences with gradual context increase

Key Idea:
- Start with good short-context model
- Gradually increase context window
- Transfer learning from short to long
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


class HierarchicalLM(nn.Module):
    """
    Hierarchical Language Model.
    Uses sentence-level encoder + document-level aggregator.
    """
    
    def __init__(self, vocab_size, d_model=256, max_tokens=512, nhead=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        self.max_tokens = max_tokens
        
        # Document-level aggregation
        self.doc_proj = nn.Linear(d_model, d_model)
    
    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        
        x = self.transformer(x)
        return self.head(x)
    
    def load_from_short_context(self, checkpoint_path):
        """Load weights from short-context model."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Filter out head weights if vocab sizes differ
            model_dict = self.state_dict()
            for key in state_dict:
                if key in model_dict and state_dict[key].shape == model_dict[key].shape:
                    model_dict[key] = state_dict[key]
            
            self.load_state_dict(model_dict, strict=False)
            logger.info("Loaded weights from %s", checkpoint_path)
        else:
            logger.warning("Checkpoint not found: %s", checkpoint_path)


class HierarchicalDataset(Dataset):
    """Dataset with hierarchical sequences (short → long)."""
    
    def __init__(self, filepath, base_length=128, target_length=512, stride=64):
        super().__init__()
        self.base_length = base_length
        self.target_length = target_length
        self.stride = stride
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []
        
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        
        # Flatten all tokens
        all_tokens = [w for sent in lines for w in sent]
        
        # Build vocabulary
        for word in all_tokens:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create sequences with varying lengths (curriculum learning)
        for length in [base_length, base_length*2, base_length*3, target_length]:
            for i in range(0, len(all_tokens) - length, stride):
                sample = all_tokens[i:i + length]
                self.data.append(sample)
        
        logger.info("Vocab size: %d, Sequences: %d (curriculum: %d→%d)", 
                   len(self.vocab), len(self.data), base_length, target_length)
    
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


def train_epoch(model, dataloader, optimizer, criterion, device, warmup_steps=0, current_step=0):
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
        
        # Learning rate warmup
        if warmup_steps > 0 and current_step < warmup_steps:
            lr_scale = (current_step + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = model.base_lr * lr_scale
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        current_step += 1
        
        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Loss: %.4f | LR: %.6f", 
                       batch_idx, len(dataloader), loss.item(), optimizer.param_groups[0]['lr'])
    
    return total_loss / num_batches, current_step


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
    SHORT_CONTEXT_CHECKPOINT = "contextcloud/checkpoints/context_stable_best.pt"  # Single-sentence baseline (loss 1.30)
    
    BASE_LENGTH = 128
    TARGET_LENGTH = 512
    STRIDE = 64
    D_MODEL = 256
    NHEAD = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
    EPOCHS = 30
    PATIENCE = 5
    WARMUP_EPOCHS = 2
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("=== HIERARCHICAL TRAINING (SHORT → LONG) ===")
    logger.info("Base Length: %d, Target Length: %d, LR: %.5f", BASE_LENGTH, TARGET_LENGTH, LEARNING_RATE)
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return
    
    dataset = HierarchicalDataset(DATA_FILE, base_length=BASE_LENGTH, target_length=TARGET_LENGTH, stride=STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)
    
    model = HierarchicalLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        max_tokens=TARGET_LENGTH,
        nhead=NHEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    
    # Load from short-context model if available
    model.base_lr = LEARNING_RATE
    model.load_from_short_context(SHORT_CONTEXT_CHECKPOINT)
    
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    current_step = 0
    warmup_steps = WARMUP_EPOCHS * len(dataloader)
    
    for epoch in range(1, EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, EPOCHS)
        
        train_loss, current_step = train_epoch(
            model, dataloader, optimizer, criterion, device,
            warmup_steps=warmup_steps, current_step=current_step
        )
        val_loss = evaluate(model, dataloader, criterion, device)
        
        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "hierarchical_best.pt")
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
