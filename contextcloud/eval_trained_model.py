#!/usr/bin/env python3
"""Test trained ContextCloudLM on long texts (>=500 words)."""

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
    """Create long text samples by concatenating sentences."""

    def __init__(self, filepath: str, min_length: int = 500, max_length: int = 1000):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.long_texts = []

        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        current_text = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

            current_text.append(sentence)
            current_length += len(words)

            if current_length >= min_length:
                if current_length <= max_length:
                    self.long_texts.append(" ".join(current_text))
                else:
                    trimmed = []
                    trimmed_length = 0
                    for sent in current_text:
                        sent_words = sent.split()
                        if trimmed_length + len(sent_words) <= max_length:
                            trimmed.append(sent)
                            trimmed_length += len(sent_words)
                        else:
                            break
                    if trimmed:
                        self.long_texts.append(" ".join(trimmed))

                current_text = []
                current_length = 0

        self.vocab_size = len(self.vocab)
        logger.info("Created %d long texts (min=%d, max=%d words)",
                    len(self.long_texts), min_length, max_length)
        logger.info("Vocab size: %d", self.vocab_size)

    def __len__(self):
        return len(self.long_texts)

    def __getitem__(self, idx: int):
        text = self.long_texts[idx]
        words = text.split()
        token_ids = [min(self.vocab.get(w, 1), self.vocab_size - 1) for w in words]
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


def evaluate_long_text(model, tokens, mask, vocab_size):
    """Evaluate model on a single long text."""
    model.eval()
    tokens = tokens.unsqueeze(0)
    mask = mask.unsqueeze(0)

    with torch.no_grad():
        logits = model(tokens.unsqueeze(1), sentence_mask=mask.unsqueeze(1))

    valid_count = mask[0].sum().item()
    if valid_count > 0:
        target = min(tokens[0, valid_count - 1].item(), vocab_size - 1)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        loss = -log_probs[target]
        ppl = torch.exp(loss)
        return loss.item(), ppl.item()
    return float('nan'), float('nan')


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    CHECKPOINT_PATH = "contextcloud/checkpoints/context_best.pt"
    MIN_LENGTH = 500

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    if not os.path.exists(CHECKPOINT_PATH):
        logger.error("Checkpoint not found: %s", CHECKPOINT_PATH)
        return

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    vocab_size = checkpoint["vocab_size"]
    logger.info("Loaded checkpoint from epoch %d, val_loss=%.4f",
                checkpoint.get("epoch", "N/A"), checkpoint.get("val_loss", "N/A"))

    # Create dataset - limit to 128 tokens to match checkpoint config
    dataset = LongTextDataset(DATA_FILE, min_length=MIN_LENGTH, max_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Create model with checkpoint vocab and config (must match training)
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
        max_tokens=128,  # Must match checkpoint
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    logger.info("Model loaded with %d parameters", sum(p.numel() for p in model.parameters()))

    # Test
    logger.info("\n=== TRAINED CONTEXT CLOUDLM TEST (min=%d words) ===", MIN_LENGTH)

    total_loss = 0.0
    total_ppl = 0.0
    num_samples = 0

    for batch_idx, (tokens, mask) in enumerate(dataloader):
        tokens = tokens.to(device)
        mask = mask.to(device)

        for i in range(tokens.size(0)):
            single_mask = mask[i]
            loss, ppl = evaluate_long_text(model, tokens[i], single_mask, vocab_size)
            if not torch.isnan(torch.tensor(loss)):
                total_loss += loss
                total_ppl += ppl
                num_samples += 1

        if batch_idx % 100 == 0:
            logger.info("Batch %d/%d", batch_idx, len(dataloader))

    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_ppl = total_ppl / num_samples
        logger.info("\n=== RESULTS ===")
        logger.info("Trained ContextCloudLM (Epoch %d)", checkpoint.get("epoch", "N/A"))
        logger.info("Avg Loss: %.4f", avg_loss)
        logger.info("Avg PPL: %.2f", avg_ppl)
        logger.info("Samples: %d", num_samples)
    else:
        logger.warning("No valid samples evaluated")


if __name__ == "__main__":
    main()
