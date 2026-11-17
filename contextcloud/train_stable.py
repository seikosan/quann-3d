#!/usr/bin/env python3
"""Train ContextCloudLM with gradient clipping and warmup for stability."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM
from contextcloud.task_utils import shift_target_2d, split_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple text dataset with controlled vocab."""

    def __init__(self, sentences, max_tokens: int = 128, vocab_size: int = 10000,
                 vocab: Optional[dict] = None):
        super().__init__()
        self.max_tokens = max_tokens

        if isinstance(sentences, str):
            with open(sentences, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        self.data = [line.split() for line in sentences if line.strip()]

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {"<pad>": 0, "<unk>": 1}
            for words in self.data:
                for word in words:
                    if len(self.vocab) < vocab_size and word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
            while len(self.vocab) < vocab_size:
                self.vocab[f"<extra_{len(self.vocab)}>"] = len(self.vocab)

        logger.info("Vocab size: %d, Samples: %d", len(self.vocab), len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        words = self.data[idx][:self.max_tokens]
        token_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in words]
        token_ids = token_ids + [0] * (self.max_tokens - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    tokens_list = batch
    max_len = max(len(t) for t in tokens_list)
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens[i, :len(t)] = t
    mask = tokens.ne(0)
    return tokens, mask


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, grad_clip, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (tokens, mask) in enumerate(dataloader):
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Shift: hedef = son geçerli token, girdiden çıkarılır (sızdırma yok)
        tokens, mask, target = shift_target_2d(tokens, mask, model.vocab_size)

        tokens_3d = tokens.unsqueeze(1)
        mask_3d = mask.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(tokens_3d, sentence_mask=mask_3d)

        loss = criterion(logits, target)

        # Gradient clipping
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()  # Learning rate warmup/scheduler

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Loss: %.4f | Grad Norm: %.2f | LR: %.6f",
                        batch_idx, len(dataloader), loss.item(), total_norm.item(),
                        optimizer.param_groups[0]['lr'])

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for tokens, mask in dataloader:
            tokens = tokens.to(device)
            mask = mask.to(device)

            tokens, mask, target = shift_target_2d(tokens, mask, model.vocab_size)

            tokens_3d = tokens.unsqueeze(1)
            mask_3d = mask.unsqueeze(1)
            logits = model(tokens_3d, sentence_mask=mask_3d)

            loss = criterion(logits, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    MAX_TOKENS = 128
    BATCH_SIZE = 16
    VOCAB_SIZE = 10000  # Controlled vocab size
    LEARNING_RATE = 1e-3
    WARMUP_EPOCHS = 3
    TOTAL_EPOCHS = 30
    GRAD_CLIP = 0.5  # Strong gradient clipping
    EARLY_STOPPING_PATIENCE = 5

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Dataset: ayrık 80/10/10 bölme (AGENTS §4.1, seed=42); vocab yalnız train'den
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    train_lines, val_lines, test_lines = split_sentences(lines)
    logger.info("Split: train %d / val %d / test %d (seed=42)",
                len(train_lines), len(val_lines), len(test_lines))

    dataset = SimpleDataset(train_lines, max_tokens=MAX_TOKENS, vocab_size=VOCAB_SIZE)
    val_dataset = SimpleDataset(val_lines, max_tokens=MAX_TOKENS, vocab=dataset.vocab)
    test_dataset = SimpleDataset(test_lines, max_tokens=MAX_TOKENS, vocab=dataset.vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(dataset.vocab)
    logger.info("Final vocab size: %d", vocab_size)

    # Model (medium size)
    model = ContextCloudLM(
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
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=len(dataloader),
        pct_start=WARMUP_EPOCHS / TOTAL_EPOCHS,
        anneal_strategy='cos'
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, TOTAL_EPOCHS)

        train_loss = train_epoch(model, dataloader, optimizer, scheduler, criterion, device, GRAD_CLIP, epoch)
        val_loss = evaluate(model, val_loader, criterion, device)

        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_stable_best.pt")
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

    checkpoint_path = os.path.join(OUTPUT_DIR, "context_stable_best.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info("TEST loss (best ckpt, ayrık holdout): %.4f", test_loss)


if __name__ == "__main__":
    main()
