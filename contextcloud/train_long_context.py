#!/usr/bin/env python3
"""Train ContextCloudLM for long texts using hierarchical approach."""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud import ContextCloudLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LongContextDataset(Dataset):
    """
    Long texts with hierarchical structure:
    - Group sentences into chunks (e.g., 10 sentences per chunk)
    - Create overlapping windows (stride=5)
    - Target: predict next token in context of previous chunks
    """

    def __init__(self, filepath: str, chunk_size: int = 10, max_tokens: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.data = []

        # Load all sentences
        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip().split() for line in f if line.strip()]

        # Build vocab
        for sent in sentences:
            for word in sent:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

        # Create chunks (each chunk = chunk_size sentences)
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            if len(chunk) >= 3:  # Minimum 3 sentences per chunk
                chunks.append(chunk)

        # Create training samples: each sample = 2 consecutive chunks
        for i in range(len(chunks) - 1):
            prev_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Flatten chunks into token sequences
            prev_tokens = [w for sent in prev_chunk for w in sent][:max_tokens]
            next_tokens = [w for sent in next_chunk for w in sent][:max_tokens]

            # Create sample: (prev_tokens, next_tokens)
            self.data.append((prev_tokens, next_tokens))

        logger.info("Vocab size: %d, Chunks: %d, Samples: %d",
                    len(self.vocab), len(chunks), len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        prev_tokens, next_tokens = self.data[idx]

        # Tokenize
        prev_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in prev_tokens]
        next_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in next_tokens]

        # Pad
        prev_ids = prev_ids + [0] * (self.max_tokens - len(prev_ids))
        next_ids = next_ids + [0] * (self.max_tokens - len(next_ids))

        return torch.tensor(prev_ids, dtype=torch.long), torch.tensor(next_ids, dtype=torch.long)


def collate_fn(batch):
    prev_list, next_list = zip(*batch)
    max_prev = max(len(t) for t in prev_list)
    max_next = max(len(t) for t in next_list)

    prev = torch.zeros(len(batch), max_prev, dtype=torch.long)
    next_tokens = torch.zeros(len(batch), max_next, dtype=torch.long)

    for i, (p, n) in enumerate(zip(prev_list, next_list)):
        prev[i, :len(p)] = p
        next_tokens[i, :len(n)] = n

    prev_mask = prev.ne(0)
    next_mask = next_tokens.ne(0)

    return prev, next_tokens, prev_mask, next_mask


def hierarchical_forward(model, prev_tokens, next_tokens, prev_mask, next_mask, device):
    """
    Hierarchical forward pass:
    1. Encode previous chunk (context)
    2. Encode next chunk (target)
    3. Use previous chunk state as additional context
    """
    # Encode previous chunk (context)
    prev_3d = prev_tokens.unsqueeze(1)
    prev_mask_3d = prev_mask.unsqueeze(1)
    prev_logits = model(prev_3d, sentence_mask=prev_mask_3d)

    # Encode next chunk (target) - this is what we predict
    next_3d = next_tokens.unsqueeze(1)
    next_mask_3d = next_mask.unsqueeze(1)
    next_logits = model(next_3d, sentence_mask=next_mask_3d)

    return prev_logits, next_logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (prev, next_tokens, prev_mask, next_mask) in enumerate(dataloader):
        prev = prev.to(device)
        next_tokens = next_tokens.to(device)
        prev_mask = prev_mask.to(device)
        next_mask = next_mask.to(device)

        optimizer.zero_grad()

        # Hierarchical forward
        prev_logits, next_logits = hierarchical_forward(
            model, prev, next_tokens, prev_mask, next_mask, device
        )

        # Loss: predict next chunk based on previous chunk
        # Target: last token of next chunk
        targets = []
        for i in range(next_tokens.size(0)):
            valid_count = next_mask[i].sum().item()
            if valid_count > 0:
                targets.append(min(next_tokens[i, valid_count - 1].item(), model.vocab_size - 1))
            else:
                targets.append(0)
        target = torch.tensor(targets).to(device)

        loss = criterion(next_logits, target)

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
        for prev, next_tokens, prev_mask, next_mask in dataloader:
            prev = prev.to(device)
            next_tokens = next_tokens.to(device)
            prev_mask = prev_mask.to(device)
            next_mask = next_mask.to(device)

            prev_logits, next_logits = hierarchical_forward(
                model, prev, next_tokens, prev_mask, next_mask, device
            )

            targets = []
            for i in range(next_tokens.size(0)):
                valid_count = next_mask[i].sum().item()
                if valid_count > 0:
                    targets.append(min(next_tokens[i, valid_count - 1].item(), model.vocab_size - 1))
                else:
                    targets.append(0)
            target = torch.tensor(targets).to(device)

            loss = criterion(next_logits, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    CHUNK_SIZE = 10  # 10 sentences per chunk
    MAX_TOKENS = 256  # Longer sequences
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 4

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Dataset
    dataset = LongContextDataset(DATA_FILE, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)

    # Model (same architecture, trained on chunk pairs)
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
    criterion = torch.nn.CrossEntropyLoss()

    # Training
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
            checkpoint_path = os.path.join(OUTPUT_DIR, "context_long_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "chunk_size": CHUNK_SIZE,
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
