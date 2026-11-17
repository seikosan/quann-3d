#!/usr/bin/env python3
"""Görev-sabit mekanizma ablation'ı: memory GÖREVİNE stable pencere/bulut mekanizması.

Kalan ~0.4 loss farkının görev mi mekanizma mı olduğunu ayırır:
veri/bölme/vocab/görev memory koşusuyla birebir, eğitim reçetesi memory ile
birebir; tek fark mekanizma (ContextCloudLM pencere/bulut vs memory slotları).
"""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM
from contextcloud.dataset import ChunkedVocabDataset, collate_chunks
from contextcloud.task_utils import shift_target_3d, split_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_idx, (chunks, masks) in enumerate(dataloader):
        chunks = chunks.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        chunks, masks, target = shift_target_3d(chunks, masks, model.vocab_size)
        logits = model(chunks, sentence_mask=masks)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
            chunks, masks, target = shift_target_3d(chunks, masks, model.vocab_size)
            logits = model(chunks, sentence_mask=masks)
            loss = criterion(logits, target)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    CKPT_NAME = "ablate_stable_on_memorytask_best.pt"
    CHUNK_SIZE = 5
    MAX_TOKENS = 256
    NUM_CHUNKS = 4
    VOCAB_SIZE = 10000
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.02
    EPOCHS = 30
    PATIENCE = 5
    GRAD_CLIP = 1.0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Ablation: stable pencere/bulut mekanizması, memory görevi/reçetesi")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    train_lines, val_lines, test_lines = split_sentences(lines)
    logger.info("Split: train %d / val %d / test %d (seed=42)",
                len(train_lines), len(val_lines), len(test_lines))

    dataset = ChunkedVocabDataset(train_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS,
                                   num_chunks=NUM_CHUNKS, vocab_size=VOCAB_SIZE)
    val_dataset = ChunkedVocabDataset(val_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS,
                                       num_chunks=NUM_CHUNKS, vocab=dataset.vocab)
    test_dataset = ChunkedVocabDataset(test_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS,
                                        num_chunks=NUM_CHUNKS, vocab=dataset.vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_chunks)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_chunks)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_chunks)

    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)

    # stable'ın boyutları (train_stable.py ile birebir) — kapasite eşitlenmiş kıyas
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

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, EPOCHS)
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, GRAD_CLIP)
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "val_loss": val_loss,
            }, os.path.join(OUTPUT_DIR, CKPT_NAME))
            logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            logger.info("Patience: %d/%d", patience_counter, PATIENCE)

        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered")
            break

    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Best val loss: %.4f", best_val_loss)

    checkpoint_path = os.path.join(OUTPUT_DIR, CKPT_NAME)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info("TEST loss (best ckpt, ayrık holdout): %.4f", test_loss)


if __name__ == "__main__":
    main()
