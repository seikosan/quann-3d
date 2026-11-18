#!/usr/bin/env python3
"""Train ContextCloudLM with gradient clipping and warmup for stability."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM
from contextcloud.dataset import WordVocabDataset, collate_tokens
from contextcloud.task_utils import shift_target_2d, split_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description="Stable CC eğitimi (R5 akışı)")
    parser.add_argument("--seed", type=int, default=None,
                        help="torch RNG seedi (model init + shuffle); None = tohumlanmamış "
                             "(R5 orijinal davranışıyla birebir)")
    parser.add_argument("--ckpt-name", default="context_stable_best.pt",
                        help="best checkpoint adı; seed tekrarlarında ayrı ad orijinal kanıtı korur")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

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
    logger.info("Torch seed: %s | ckpt: %s | veri bölme seed=42 sabit (AGENTS §4.1)",
                args.seed, args.ckpt_name)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Dataset: ayrık 80/10/10 bölme (AGENTS §4.1, seed=42); vocab yalnız train'den
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    train_lines, val_lines, test_lines = split_sentences(lines)
    logger.info("Split: train %d / val %d / test %d (seed=42)",
                len(train_lines), len(val_lines), len(test_lines))

    dataset = WordVocabDataset(train_lines, max_tokens=MAX_TOKENS, vocab_size=VOCAB_SIZE)
    val_dataset = WordVocabDataset(val_lines, max_tokens=MAX_TOKENS, vocab=dataset.vocab)
    test_dataset = WordVocabDataset(test_lines, max_tokens=MAX_TOKENS, vocab=dataset.vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_tokens)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_tokens)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_tokens)

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
            checkpoint_path = os.path.join(OUTPUT_DIR, args.ckpt_name)
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

    checkpoint_path = os.path.join(OUTPUT_DIR, args.ckpt_name)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info("TEST loss (best ckpt, ayrık holdout): %.4f", test_loss)


if __name__ == "__main__":
    main()
