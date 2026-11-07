#!/usr/bin/env python3
"""Test models on long texts (>=500 words) to evaluate context handling."""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cloudlm.model import CloudLM
from purecloudlm.model import PureCloudLM
from contextcloud import ContextCloudLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class LongTextDataset(Dataset):
    """Create long text samples by concatenating sentences."""

    def __init__(self, filepath: str, min_length: int = 500, max_length: int = 1000):
        """
        Args:
            filepath: Path to text file with one sentence per line
            min_length: Minimum word count for long text
            max_length: Maximum word count for long text
        """
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.long_texts = []

        # Load all sentences
        with open(filepath, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        # Build vocab and create long texts
        current_text = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

            current_text.append(sentence)
            current_length += len(words)

            # When we have enough words, save as a long text
            if current_length >= min_length:
                if current_length <= max_length:
                    self.long_texts.append(" ".join(current_text))
                else:
                    # Trim to max_length
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

                # Reset for next long text
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
        token_ids = [self.vocab.get(w, 1) for w in words]
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


def evaluate_long_text(model, tokens, mask, model_name):
    """Evaluate model on a single long text."""
    model.eval()
    tokens = tokens.unsqueeze(0)  # Add batch dimension
    mask = mask.unsqueeze(0)

    with torch.no_grad():
        if model_name == "CloudLM":
            logits, _, _ = model(tokens, mask)
        elif model_name == "PureCloudLM":
            logits, _, _ = model(tokens, mask)
        elif model_name == "ContextCloudLM":
            # ContextCloudLM expects (B, S, T) - treat as single sentence
            logits = model(tokens.unsqueeze(1), sentence_mask=mask.unsqueeze(1))
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Calculate perplexity for last valid token
    valid_count = mask[0].sum().item()
    if valid_count > 0:
        target = tokens[0, valid_count - 1]
        log_probs = torch.log_softmax(logits[0], dim=-1)
        loss = -log_probs[target]
        ppl = torch.exp(loss)
        return loss.item(), ppl.item()
    return float('nan'), float('nan')


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    MIN_LENGTH = 500  # Minimum words
    MAX_LENGTH = 800  # Maximum words
    BATCH_SIZE = 4

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Dataset
    dataset = LongTextDataset(DATA_FILE, min_length=MIN_LENGTH, max_length=MAX_LENGTH)
    if len(dataset) == 0:
        logger.warning("No long texts found with min_length=%d", MIN_LENGTH)
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Models (small configs for quick test)
    models = {}

    # CloudLM
    models["CloudLM"] = CloudLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        d_cloud=256,
        d_cloud_dim=8,
        n_layers=2,
        nhead=4,
        max_pos_len=1024,
        hidden=128,
        proj_hidden=None,
        point_hidden=64,
        pool="max",
        cloud_attn=False,
        cloud_attn_heads=4,
        attn_scale_init=1.0,
    )

    # PureCloudLM
    models["PureCloudLM"] = PureCloudLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        d_cloud=256,
        d_cloud_dim=8,
        n_layers=1,
        pos_len=1024,
        hidden=128,
        proj_hidden=None,
        point_hidden=64,
        pool_mode="all",
        use_pair_approx=False,
        num_sampled_pairs=64,
    )

    # ContextCloudLM
    models["ContextCloudLM"] = ContextCloudLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        d_sentence=256,
        d_window=256,
        d_cloud_dim=8,
        d_window_dim=16,
        window_size=5,
        window_stride=2,
        hidden=128,
        max_tokens=1024,
    )

    # Test each model on long texts
    logger.info("\n=== LONG TEXT TEST (min=%d words) ===", MIN_LENGTH)
    logger.info("Model              | Avg Loss | Avg PPL  | Samples")
    logger.info("-------------------|----------|----------|--------")

    for name, model in models.items():
        model = model.to(device)
        logger.info("Evaluating %s on long texts...", name)

        total_loss = 0.0
        total_ppl = 0.0
        num_samples = 0

        for batch_idx, (tokens, mask) in enumerate(dataloader):
            tokens = tokens.to(device)
            mask = mask.to(device)

            for i in range(tokens.size(0)):
                single_mask = mask[i]
                loss, ppl = evaluate_long_text(model, tokens[i], single_mask, name)
                if not torch.isnan(torch.tensor(loss)):
                    total_loss += loss
                    total_ppl += ppl
                    num_samples += 1

        if num_samples > 0:
            avg_loss = total_loss / num_samples
            avg_ppl = total_ppl / num_samples
            logger.info("%-17s | %.4f    | %.2f     | %d",
                        name, avg_loss, avg_ppl, num_samples)
        else:
            logger.info("%-17s | N/A (no valid samples)", name)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    main()
