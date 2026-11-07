#!/usr/bin/env python3
"""ContextCloudLM full training with 80/10/10 split and early stopping."""

import logging
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class TurkishSentencesDataset(Dataset):
    """Load Turkish sentences with 80/10/10 split support."""

    def __init__(self, filepath: str, max_tokens: int = 128, split: str = "train"):
        """
        Args:
            filepath: Path to text file
            max_tokens: Maximum tokens per sentence
            split: "train", "val", or "test"
        """
        super().__init__()
        self.max_tokens = max_tokens
        self.split = split
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.sentences = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                if len(words) == 0:
                    continue
                self.sentences.append(words)
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)

        self.vocab_size = len(self.vocab)
        logger.info("Total sentences: %d, vocab size: %d", len(self.sentences), self.vocab_size)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        words = self.sentences[idx][:self.max_tokens]
        token_ids = [self.vocab.get(w, 1) for w in words]
        token_ids = token_ids + [0] * (self.max_tokens - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)


def create_splits(filepath: str, max_tokens: int = 128, seed: int = 42):
    """Create 80/10/10 train/val/test splits."""
    import random
    random.seed(seed)

    # Load all sentences first
    dataset = TurkishSentencesDataset(filepath, max_tokens)
    all_sentences = dataset.sentences
    vocab_size = dataset.vocab_size

    # Shuffle
    random.shuffle(all_sentences)

    n = len(all_sentences)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)

    train_data = all_sentences[:train_size]
    val_data = all_sentences[train_size:train_size + val_size]
    test_data = all_sentences[train_size + val_size:]

    logger.info("Split sizes: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
                len(train_data), 100*len(train_data)/n,
                len(val_data), 100*len(val_data)/n,
                len(test_data), 100*len(test_data)/n)

    return train_data, val_data, test_data, vocab_size


class SplitDataset(Dataset):
    """Dataset from pre-split sentences."""

    def __init__(self, sentences: list, max_tokens: int = 128, vocab: dict = None):
        super().__init__()
        self.max_tokens = max_tokens
        self.sentences = sentences
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        words = self.sentences[idx][:self.max_tokens]
        token_ids = [self.vocab.get(w, 1) for w in words]
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


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return average loss."""
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

            # Target: last valid token
            targets = []
            for i in range(tokens.size(0)):
                valid_count = mask[i].sum().item()
                if valid_count > 0:
                    targets.append(tokens[i, valid_count - 1])
                else:
                    targets.append(torch.tensor(0))
            target = torch.stack(targets).to(device)

            loss = criterion(logits, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (tokens, mask) in enumerate(dataloader):
        tokens = tokens.to(device)
        mask = mask.to(device)

        tokens_3d = tokens.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(tokens_3d, sentence_mask=mask_3d)

        targets = []
        for i in range(tokens.size(0)):
            valid_count = mask[i].sum().item()
            if valid_count > 0:
                targets.append(tokens[i, valid_count - 1])
            else:
                targets.append(torch.tensor(0))
        target = torch.stack(targets).to(device)

        loss = criterion(logits, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Loss: %.4f", batch_idx, len(dataloader), loss.item())

    return total_loss / num_batches


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    MAX_TOKENS = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 3  # Epochs without improvement
    MIN_LOSS_THRESHOLD = 4.0  # Stop if loss goes below this (good performance)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Create splits
    logger.info("Creating 80/10/10 splits...")
    train_data, val_data, test_data, vocab_size = create_splits(DATA_FILE, MAX_TOKENS)

    # Create datasets
    train_dataset = SplitDataset(train_data, MAX_TOKENS)
    val_dataset = SplitDataset(val_data, MAX_TOKENS)
    test_dataset = SplitDataset(test_data, MAX_TOKENS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model (larger for full training)
    model = ContextCloudLM(
        vocab_size=vocab_size,
        d_model=256,
        d_sentence=512,
        d_window=512,
        d_cloud_dim=16,
        d_window_dim=32,
        window_size=5,
        window_stride=2,
        hidden=256,
        max_tokens=MAX_TOKENS,
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Training with early stopping
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    epoch = 0

    while True:
        epoch += 1
        logger.info("\n=== Epoch %d ===", epoch)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_best.pt")
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

        # Test evaluation every epoch
        test_loss = evaluate(model, test_loader, criterion, device)
        logger.info("Test Loss: %.4f", test_loss)

        # Early stopping conditions
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered (no improvement for %d epochs)", EARLY_STOPPING_PATIENCE)
            break

        if best_val_loss < MIN_LOSS_THRESHOLD:
            logger.info("Loss threshold reached (%.4f < %.2f), training complete",
                        best_val_loss, MIN_LOSS_THRESHOLD)
            break

        # Safety: stop after 100 epochs
        if epoch >= 100:
            logger.info("Maximum epochs (100) reached")
            break

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Best validation loss: %.4f", best_val_loss)

    # Final test evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "context_best.pt"))["model_state_dict"])
    final_test_loss = evaluate(model, test_loader, criterion, device)
    logger.info("Final Test Loss: %.4f", final_test_loss)


if __name__ == "__main__":
    main()
