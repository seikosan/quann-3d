#!/usr/bin/env python3
"""ContextCloudLM training - simple sentence classification task."""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Load sentences, tokenize, and prepare for training."""

    def __init__(self, filepath: str, max_tokens: int = 64):
        super().__init__()
        self.max_tokens = max_tokens
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
        logger.info("Loaded %d sentences, vocab size: %d", len(self.sentences), self.vocab_size)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        words = self.sentences[idx][:self.max_tokens]
        token_ids = [self.vocab.get(w, 1) for w in words]
        # Pad to fixed length
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

        # Reshape to (B, 1, T) for single-sentence input
        tokens_3d = tokens.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(tokens_3d, sentence_mask=mask_3d)

        # For next-token prediction: use last valid token as target
        targets = []
        for i in range(tokens.size(0)):
            valid_count = mask[i].sum().item()
            if valid_count > 0:
                # Target is the last token in the sequence
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

        if batch_idx % 20 == 0:
            logger.info("Batch %d/%d | Loss: %.4f", batch_idx, len(dataloader), loss.item())

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

            # Same target extraction
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


def main():
    # Configuration
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    NUM_EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Dataset
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    dataset = SimpleDataset(DATA_FILE, max_tokens=32)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Model - smaller for quick test
    model = ContextCloudLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        d_sentence=256,
        d_window=256,
        d_cloud_dim=8,
        d_window_dim=16,
        window_size=3,
        window_stride=1,
        hidden=128,
        max_tokens=32,
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info("=== Epoch %d/%d ===", epoch, NUM_EPOCHS)
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, dataloader, criterion, device)

        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)

        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": dataset.vocab_size,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
            logger.info("Saved best checkpoint: %s (val_loss=%.4f)", checkpoint_path, val_loss)

    logger.info("Training complete! Best val loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
