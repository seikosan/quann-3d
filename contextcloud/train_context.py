"""Train ContextCloudLM with sentence-, window-, and context-level objectives."""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from contextcloud.model import ContextCloudLM


logger = logging.getLogger(__name__)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def choose_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU according to runtime availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_sentences(path: Path) -> list[str]:
    """Read non-empty UTF-8 sentence lines from a corpus file."""
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")
    with path.open("r", encoding="utf-8") as corpus_file:
        sentences = [line.strip() for line in corpus_file if line.strip()]
    if len(sentences) < 3:
        raise ValueError("The corpus must contain at least three non-empty sentences")
    return sentences


def split_sentences(
    sentences: Sequence[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle sentences deterministically and split them into train/validation/test."""
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    validation_end = train_end + int(len(shuffled) * 0.1)
    train_sentences = shuffled[:train_end]
    validation_sentences = shuffled[train_end:validation_end]
    test_sentences = shuffled[validation_end:]
    if min(len(train_sentences), len(validation_sentences), len(test_sentences)) < 2:
        raise ValueError("Each data split must contain at least two sentences")
    return train_sentences, validation_sentences, test_sentences


def build_vocabulary(
    sentences: Sequence[str], max_vocab_size: int
) -> dict[str, int]:
    """Build a frequency-ordered vocabulary with padding and unknown IDs reserved."""
    if max_vocab_size < 3:
        raise ValueError("max_vocab_size must be at least 3")
    counts: Counter[str] = Counter(
        token.lower() for sentence in sentences for token in sentence.split()
    )
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    available = max_vocab_size - len(vocabulary)
    ranked_tokens = sorted(counts, key=lambda token: (-counts[token], token))[:available]
    vocabulary.update({token: index + 2 for index, token in enumerate(ranked_tokens)})
    return vocabulary


def _window_starts(sequence_length: int, window_size: int, stride: int) -> list[int]:
    """Return starts matching ContextCloudLM's sliding-window boundary behavior."""
    if sequence_length <= window_size:
        return [0]
    last_start = sequence_length - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] + window_size < sequence_length:
        starts.append(last_start)
    return starts


class ContextDataset(Dataset[dict[str, torch.Tensor]]):
    """Create fixed-size preceding-sentence contexts and next-sentence targets."""

    def __init__(
        self,
        sentences: Sequence[str],
        vocabulary: Mapping[str, int],
        context_size: int = 4,
        max_tokens: int = 32,
        window_size: int = 2,
        window_stride: int = 1,
    ) -> None:
        """Encode text and prepare one prediction example per sentence after the first."""
        if context_size <= 0 or max_tokens <= 0:
            raise ValueError("context_size and max_tokens must be positive")
        if window_size <= 0 or window_stride <= 0:
            raise ValueError("window_size and window_stride must be positive")
        if len(sentences) < 2:
            raise ValueError("ContextDataset requires at least two sentences")
        self.pad_id = vocabulary[PAD_TOKEN]
        self.unk_id = vocabulary[UNK_TOKEN]
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.window_stride = window_stride
        self.encoded_sentences = [
            [vocabulary.get(token.lower(), self.unk_id) for token in sentence.split()][
                :max_tokens
            ]
            for sentence in sentences
        ]
        self.encoded_sentences = [tokens for tokens in self.encoded_sentences if tokens]
        if len(self.encoded_sentences) < 2:
            raise ValueError("At least two corpus sentences must contain tokens")
        self.window_starts = _window_starts(context_size, window_size, window_stride)

    def __len__(self) -> int:
        """Return the number of next-sentence prediction examples."""
        return len(self.encoded_sentences) - 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return padded context, latest sentence, context windows, and target token."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_sentence_index = index + 1
        context_sentences = self.encoded_sentences[
            max(0, target_sentence_index - self.context_size) : target_sentence_index
        ]
        token_rows = [
            torch.tensor(tokens, dtype=torch.long) for tokens in context_sentences
        ]
        token_rows = [
            F.pad(row, (0, self.max_tokens - row.numel()), value=self.pad_id)
            for row in token_rows
        ]
        padding_rows = self.context_size - len(token_rows)
        token_rows = [
            torch.full((self.max_tokens,), self.pad_id, dtype=torch.long)
            for _ in range(padding_rows)
        ] + token_rows
        context_tokens = torch.stack(token_rows)
        sentence_mask = context_tokens.ne(self.pad_id).any(dim=-1)
        window_tokens = torch.stack(
            [context_tokens[start : start + self.window_size] for start in self.window_starts]
        )
        window_sentence_mask = window_tokens.ne(self.pad_id).any(dim=-1)
        target = torch.tensor(self.encoded_sentences[target_sentence_index][0], dtype=torch.long)
        return {
            "context_tokens": context_tokens,
            "sentence_mask": sentence_mask,
            "sentence_tokens": context_tokens[-1],
            "window_tokens": window_tokens,
            "window_sentence_mask": window_sentence_mask,
            "target": target,
        }


def multi_level_loss(
    sentence_logits: torch.Tensor,
    window_logits: torch.Tensor,
    context_logits: torch.Tensor,
    window_mask: torch.Tensor,
    targets: torch.Tensor,
    sentence_weight: float = 0.25,
    window_weight: float = 0.25,
    context_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine next-token cross entropy from sentence, window, and full-context levels."""
    sentence_loss = F.cross_entropy(sentence_logits, targets)
    batch_size, num_windows, vocab_size = window_logits.shape
    per_window_loss = F.cross_entropy(
        window_logits.reshape(batch_size * num_windows, vocab_size),
        targets.unsqueeze(1).expand(batch_size, num_windows).reshape(-1),
        reduction="none",
    ).reshape(batch_size, num_windows)
    valid_weights = window_mask.to(dtype=per_window_loss.dtype)
    window_loss = (per_window_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1)
    context_loss = F.cross_entropy(context_logits, targets)
    total_loss = (
        sentence_weight * sentence_loss
        + window_weight * window_loss
        + context_weight * context_loss
    )
    return total_loss, {
        "sentence": sentence_loss.detach(),
        "window": window_loss.detach(),
        "context": context_loss.detach(),
    }


def _level_logits(
    model: ContextCloudLM, batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model at sentence, window, and complete-context granularity."""
    context_tokens = batch["context_tokens"].to(device)
    sentence_mask = batch["sentence_mask"].to(device)
    sentence_tokens = batch["sentence_tokens"].to(device)
    window_tokens = batch["window_tokens"].to(device)
    window_sentence_mask = batch["window_sentence_mask"].to(device)
    batch_size, num_windows, window_size, token_count = window_tokens.shape
    sentence_logits = model(sentence_tokens)
    context_logits = model(context_tokens, sentence_mask=sentence_mask)
    window_logits = model(
        window_tokens.reshape(batch_size * num_windows, window_size, token_count),
        sentence_mask=window_sentence_mask.reshape(batch_size * num_windows, window_size),
    ).reshape(batch_size, num_windows, -1)
    window_mask = window_sentence_mask.any(dim=-1)
    return sentence_logits, window_logits, context_logits, window_mask


def run_epoch(
    model: ContextCloudLM,
    batches: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch and return mean multi-level loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    example_count = 0
    for batch in batches:
        targets = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            sentence_logits, window_logits, context_logits, window_mask = _level_logits(
                model, batch, device
            )
            loss, _ = multi_level_loss(
                sentence_logits,
                window_logits,
                context_logits,
                window_mask,
                targets,
            )
            if training:
                loss.backward()
                optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.detach().item() * batch_size
        example_count += batch_size
    if example_count == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    return total_loss / example_count


def parse_args() -> argparse.Namespace:
    """Parse training options from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/turkish_sentences.txt"))
    parser.add_argument(
        "--output", type=Path, default=Path("contextcloud/checkpoints/contextcloud_best.pt")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Configure logging, prepare data, train the model, and save the best checkpoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sentences = read_sentences(args.data)
    train_sentences, validation_sentences, test_sentences = split_sentences(
        sentences, args.seed
    )
    vocabulary = build_vocabulary(train_sentences, args.max_vocab_size)
    dataset_options = {
        "vocabulary": vocabulary,
        "context_size": args.context_size,
        "max_tokens": args.max_tokens,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
    }
    train_dataset = ContextDataset(train_sentences, **dataset_options)
    validation_dataset = ContextDataset(validation_sentences, **dataset_options)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = choose_device()
    model = ContextCloudLM(
        vocab_size=len(vocabulary),
        window_size=args.window_size,
        window_stride=args.window_stride,
        max_tokens=args.max_tokens,
        padding_idx=vocabulary[PAD_TOKEN],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    logger.info(
        "device=%s train_examples=%d validation_examples=%d held_out_test_sentences=%d vocab=%d",
        device,
        len(train_dataset),
        len(validation_dataset),
        len(test_sentences),
        len(vocabulary),
    )

    best_validation_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, device)
        logger.info(
            "epoch=%d/%d train_loss=%.4f validation_loss=%.4f",
            epoch,
            args.epochs,
            train_loss,
            validation_loss,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint: dict[str, Any] = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocabulary": vocabulary,
                "validation_loss": validation_loss,
                "config": vars(args),
            }
            torch.save(checkpoint, args.output)
            logger.info("Saved best checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
