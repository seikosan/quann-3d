#!/usr/bin/env python3
"""Train ContextCloudLM on long texts with proper window context."""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud.model import ContextCloudLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class LongTextDataset(Dataset):
    """Long texts with proper tokenization for ContextCloudLM."""

    def __init__(self, filepath: str, max_tokens: int = 256, min_sentences: int = 5):
        super().__init__()
        self.max_tokens = max_tokens
        self.min_sentences = min_sentences
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []

        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        # Group sentences into long texts (5+ sentences)
        current_batch = []
        current_length = 0

        for sent in sentences:
            words = sent.split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

            current_batch.append(sent)
            current_length += len(words)

            if len(current_batch) >= self.min_sentences and current_length >= 100:
                # Create long text sample
                text = " ".join(current_batch)
                self.data.append(text)
                current_batch = []
                current_length = 0

        logger.info("Created %d long text samples, vocab size: %d",
                    len(self.data), len(self.vocab))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        text = self.data[idx]
        words = text.split()[:self.max_tokens]
        token_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in words]
        # Pad to max_tokens
        token_ids = token_ids + [0] * (self.max_tokens - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    """Pad batch to same length."""
    tokens_list = batch
    max_len = max(len(t) for t in tokens_list)
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens[i, :len(t)] = t
    mask = tokens.ne(0)
    return tokens, mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (tokens, mask) in enumerate(dataloader):
        tokens = tokens.to(device)
        mask = mask.to(device)

        # (B, T) -> (B, S=1, T) for ContextCloudLM
        tokens_3d = tokens.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(tokens_3d, sentence_mask=mask_3d)

        # Target: last valid token
        targets = []
        for i in range(tokens.size(0)):
            valid_count = mask[i].sum().item()
            if valid_count > 0:
                targets.append(min(tokens[i, valid_count - 1].item(), model.vocab_size - 1))
            else:
                targets.append(0)
        target = torch.tensor(targets).to(device)

        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        # Gradient monitoring
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if total_norm > 50.0:
                logger.warning("Large gradient: %.2f", total_norm)
            elif total_norm < 1e-5:
                logger.warning("Vanishing gradient: %.6f", total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Loss: %.4f | Grad: %.2f",
                        batch_idx, len(dataloader), loss.item() * GRADIENT_ACCUMULATION_STEPS, total_norm)

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for tokens, mask in dataloader:
            tokens = tokens.to(device)
            mask = mask.to(device)

            tokens_3d = tokens.unsqueeze(1)
            mask_3d = mask.unsqueeze(1)
            logits = model(tokens_3d, sentence_mask=mask_3d)

            targets = []
            for i in range(tokens.size(0)):
                valid_count = mask[i].sum().item()
                if valid_count > 0:
                    targets.append(min(tokens[i, valid_count - 1].item(), model.vocab_size - 1))
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
    MAX_TOKENS = 256  # Longer sequences
    MIN_SENTENCES = 5  # At least 5 sentences per sample
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3  # Increased for better gradient flow
    EARLY_STOPPING_PATIENCE = 3
    MAX_EPOCHS = 50
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Dataset
    dataset = LongTextDataset(DATA_FILE, max_tokens=MAX_TOKENS, min_sentences=MIN_SENTENCES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)

    # Model for long texts
    model = ContextCloudLM(
        vocab_size=vocab_size,
        d_model=128,
        d_sentence=256,
        d_window=256,
        d_cloud_dim=16,
        d_window_dim=16,
        window_size=5,  # 5 sentences per window
        window_stride=2,
        hidden=128,
        max_tokens=MAX_TOKENS,
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize optimizer state
    optimizer.zero_grad()

    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, MAX_EPOCHS)

        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, dataloader, criterion, device)

        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_longtext_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, checkpoint_path)
            logger.info("✓ Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            logger.info("Patience: %d/%d", patience_counter, EARLY_STOPPING_PATIENCE)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered")
            break

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
