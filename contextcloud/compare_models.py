#!/usr/bin/env python3
"""Compare CloudLM vs PureCloudLM vs ContextCloudLM on same corpus."""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

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


def evaluate(model, dataloader, criterion, device, model_name):
    """Evaluate model and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for tokens, mask in dataloader:
            tokens = tokens.to(device)
            mask = mask.to(device)

            if model_name == "CloudLM":
                # CloudLM: (B, T) -> (logits, next_point, points)
                logits, _, _ = model(tokens, mask)
            elif model_name == "PureCloudLM":
                # PureCloudLM: (B, T) -> (logits, next_point, points)
                logits, _, _ = model(tokens, mask)
            elif model_name == "ContextCloudLM":
                # ContextCloudLM: (B, 1, T) -> logits
                logits = model(tokens.unsqueeze(1), sentence_mask=mask.unsqueeze(1))
            else:
                raise ValueError(f"Unknown model: {model_name}")

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


def main():
    DATA_FILE = "data/turkish_sentences.txt"
    BATCH_SIZE = 16

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return

    # Dataset
    dataset = SimpleDataset(DATA_FILE, max_tokens=32)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    criterion = torch.nn.CrossEntropyLoss()

    # Models (small configs for quick comparison)
    models = {}

    # CloudLM
    models["CloudLM"] = CloudLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        d_cloud=256,
        d_cloud_dim=8,
        n_layers=2,
        nhead=4,
        max_pos_len=32,
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
        pos_len=32,
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
        window_size=3,
        window_stride=1,
        hidden=128,
        max_tokens=32,
    )

    # Evaluate each model
    results = {}
    for name, model in models.items():
        model = model.to(device)
        logger.info("Evaluating %s...", name)
        logger.info("  Parameters: %d", sum(p.numel() for p in model.parameters()))

        loss = evaluate(model, dataloader, criterion, device, name)
        results[name] = loss
        logger.info("  %s Loss: %.4f", name, loss)

    # Summary
    logger.info("\n=== COMPARISON SUMMARY ===")
    logger.info("Model              | Loss     | Params")
    logger.info("-------------------|----------|--------")
    for name, loss in results.items():
        params = sum(p.numel() for p in models[name].parameters())
        logger.info("%-17s | %.4f   | %d", name, loss, params)

    # Best model
    best_model = min(results, key=results.get)
    logger.info("\nBest performer: %s (loss=%.4f)", best_model, results[best_model])


if __name__ == "__main__":
    main()
