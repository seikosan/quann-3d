#!/usr/bin/env python3
"""Matched QUANN control / pair-channel-freeze experiment for R8.

The QUANN engine and QCloudTrainer remain unchanged.  The pair variant routes
QUANN selections into ``cloud_encoder.pair_mlp`` and then fine-tunes only the
word-classification head after search.  The held-out test split is evaluated
once, after all training is complete.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import random
import sys
from bisect import bisect_right
from pathlib import Path
from types import MethodType
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloudlm import BPETokenizer  # noqa: E402
from contextcloud.task_utils import split_sentences  # noqa: E402
from quann_cloud.src.qcloud_network import QCloudNetwork  # noqa: E402
from quann_cloud.src.qcloud_quann import QUANN, QUANNConfig  # noqa: E402
from quann_cloud.src.qcloud_trainer import QCloudTrainer  # noqa: E402

LOGGER = logging.getLogger("quann_pair_experiment")
DEFAULT_TOKENIZER = REPO_ROOT / "checkpoints/wiki_checkpoints_200k/wiki_bpe.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the R8 queue interface and optional CPU-smoke overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("control", "pair-freeze"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, required=True,
                        help="Maximum number of non-empty source lines to load")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="80/10/10")
    parser.add_argument("--search-channel", choices=("pair",))
    parser.add_argument("--freeze-head-during-search", action="store_true")
    parser.add_argument("--finetune-head-after-search", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto",
                        help="auto uses MPS when available, otherwise CPU")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-cloud", type=int, default=64)
    parser.add_argument("--d-cloud-dim", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--k-weights", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=8)
    parser.add_argument("--head-finetune-epochs", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if args.train_limit < 1 or args.epochs < 1:
        parser.error("--train-limit and --epochs must be positive")
    if args.seed != 42:
        parser.error("R8 data protocol fixes --seed at 42")
    if args.split != "80/10/10":
        parser.error("R8 data protocol fixes --split at 80/10/10")
    if args.d_model < 16 or args.d_model % 16:
        parser.error("--d-model must be a positive multiple of 16 (minimum 16)")
    if min(args.d_cloud, args.d_cloud_dim, args.n_layers, args.seq_len,
           args.k_weights, args.minibatch) < 1:
        parser.error("model dimensions, sequence length and search sizes must be positive")
    if args.head_finetune_epochs < 1 or args.head_lr <= 0:
        parser.error("head fine-tuning requires positive epochs and learning rate")
    pair_flags = (args.search_channel == "pair",
                  args.freeze_head_during_search,
                  args.finetune_head_after_search)
    if args.condition == "pair-freeze" and not all(pair_flags):
        parser.error("pair-freeze requires --search-channel pair, --freeze-head-during-search, "
                     "and --finetune-head-after-search")
    if args.condition == "control" and any(pair_flags):
        parser.error("pair-specific flags are only valid for pair-freeze")
    return args


def load_corpus(path: Path, limit: int) -> list[str]:
    """Read up to ``limit`` non-empty lines, preserving source order."""
    if not path.is_file():
        raise FileNotFoundError(f"corpus file not found: {path}")
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            text = line.strip()
            if text:
                lines.append(text)
                if len(lines) == limit:
                    break
    if len(lines) < 3:
        raise ValueError("corpus must contain at least three non-empty lines")
    return lines


def choose_device(requested: str) -> torch.device:
    """Resolve the execution device without silently falling back from CPU."""
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loss_fn(device: torch.device) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], float]:
    """Create the QUANN cross-entropy loss for QCloudNetwork batches."""
    def loss_fn(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = (inputs != 0) & (inputs != 1)
        with torch.no_grad():
            logits, _, _ = model(inputs, mask)
            return float(F.cross_entropy(logits, targets).item())
    return loss_fn


def route_pair_channel(engine: QUANN) -> int:
    """Restrict ``_select`` to coordinates belonging to the pair MLP.

    Returns the number of eligible pair-channel parameters.  Selection still
    uses the engine's seeded RNG, without replacement, and its normal index map.
    """
    parameter_index = {id(parameter): index
                       for index, parameter in enumerate(engine._params)}
    pair_parameters = [
        (name, parameter)
        for name, parameter in engine.model.named_parameters()
        if "cloud_encoder.pair_mlp." in name and id(parameter) in parameter_index
    ]
    ranges: list[tuple[int, int, int]] = []
    pair_starts: list[int] = []
    pair_size = 0
    for _, parameter in pair_parameters:
        index = parameter_index[id(parameter)]
        start_global = engine._cum[index]
        size = parameter.numel()
        pair_starts.append(pair_size)
        ranges.append((pair_size, pair_size + size, start_global))
        pair_size += size
    if pair_size == 0:
        raise RuntimeError("pair-freeze has no searchable cloud_encoder.pair_mlp parameters")

    def pair_select(this: QUANN, k: int) -> list[tuple[int, int]]:
        chosen = this.rng.sample(range(pair_size), min(max(k, 0), pair_size))
        selected: list[tuple[int, int]] = []
        for local_index in chosen:
            range_index = bisect_right(pair_starts, local_index) - 1
            local_start, _, global_start = ranges[range_index]
            global_index = global_start + local_index - local_start
            param_index = this._bisect_param(global_index)
            selected.append((param_index, global_index - this._cum[param_index]))
        return selected

    engine._select = MethodType(pair_select, engine)
    return pair_size


def measure_candidate_deltas(engine: QUANN, inputs: torch.Tensor,
                             targets: torch.Tensor,
                             step_fn: Callable[[torch.Tensor, torch.Tensor], dict] | None = None
                             ) -> tuple[dict, list[float]]:
    """Run one engine step and measure each actual candidate loss delta.

    QUANN.step exposes counts and final ``base_err`` but no candidate-delta
    field. Its candidate evaluations call ``_loss`` in sequence; this wrapper
    records those values and mirrors the engine's reward rule to keep the
    comparison reference aligned when a candidate is accepted.
    """
    original_loss = engine._loss
    observed: list[float] = []

    def observe_loss(step_inputs: torch.Tensor, step_targets: torch.Tensor) -> float:
        value = original_loss(step_inputs, step_targets)
        observed.append(value)
        return value

    engine._loss = observe_loss  # type: ignore[method-assign]
    try:
        stats = (step_fn or engine.step)(inputs, targets)
    finally:
        engine._loss = original_loss  # type: ignore[method-assign]

    candidate_deltas: list[float] = []
    if observed:
        reference = observed[0]
        for candidate_loss in observed[1:]:
            candidate_deltas.append(reference - candidate_loss)
            if candidate_loss < reference - engine.cfg.eps:
                reference = candidate_loss
    return stats, candidate_deltas


def instrument_step_deltas(engine: QUANN) -> list[float]:
    """Wrap the engine step to retain actual candidate gain/loss samples."""
    deltas: list[float] = []
    original_step = engine.step

    def measured_step(this: QUANN, inputs: torch.Tensor,
                      targets: torch.Tensor) -> dict:
        stats, step_deltas = measure_candidate_deltas(
            this, inputs, targets, step_fn=original_step
        )
        deltas.extend(step_deltas)
        return stats

    engine.step = MethodType(measured_step, engine)
    return deltas


def finetune_word_head(model: QCloudNetwork, examples: list[tuple[torch.Tensor, torch.Tensor]],
                       device: torch.device, seed: int, epochs: int,
                       learning_rate: float, batch_size: int) -> None:
    """Fine-tune only the classifier head after pair-directed search."""
    if epochs <= 0:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head_parameters = list(model.word_head.parameters())
    for parameter in head_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(head_parameters, lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed + 1)
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        model.train()
        for offset in range(0, len(order), batch_size):
            batch_indices = order[offset:offset + batch_size]
            inputs = torch.stack([examples[index][0] for index in batch_indices]).to(device)
            targets = torch.stack([examples[index][1] for index in batch_indices]).to(device)
            mask = (inputs != 0) & (inputs != 1)
            logits, _, _ = model(inputs, mask)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()


def use_contextual_points_for_eval(model: QCloudNetwork) -> None:
    """Adapt QCloudTrainer evaluation to QCloudNetwork's normal forward path.

    QCloudTrainer.evaluate_ppl calls ``points_for`` then supplies the returned
    points to ``forward``. Its default points are token-weight state, whereas
    QUANN trains QCloudNetwork's nn.Module parameters. This instance adapter
    makes its existing evaluation path use the model's contextual ``to_nd``
    points, exactly as ``QCloudNetwork.forward`` does without an override.
    """
    def contextual_points(this: QCloudNetwork, tokens: torch.Tensor,
                          mask: torch.Tensor | None = None,
                          eval_mode: bool = False) -> torch.Tensor:
        del eval_mode
        seq_len = tokens.size(1)
        hidden = this.embedding(tokens) * math.sqrt(this.embedding.embedding_dim)
        hidden = hidden + this.pos_enc[:, :seq_len, :]
        if mask is not None:
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0)
        hidden = this.context_encoder(
            hidden, src_key_padding_mask=~mask if mask is not None else None
        )
        hidden = this.norm(hidden)
        return this.to_nd(hidden)

    model.points_for = MethodType(contextual_points, model)  # type: ignore[method-assign]


def run(args: argparse.Namespace) -> tuple[int, float, float]:
    """Run one condition and return (train sentence count, delta, held-out PPL)."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lines = load_corpus(args.corpus, args.train_limit)
    train_texts, val_texts, test_texts = split_sentences(lines, seed=args.seed)
    if not train_texts or not val_texts or not test_texts:
        raise ValueError("80/10/10 split produced an empty partition")
    LOGGER.info("sentence split: train=%d validation=%d test=%d",
                len(train_texts), len(val_texts), len(test_texts))
    # Validation is intentionally held aside; no tuning is performed in R8.
    del val_texts

    if not args.tokenizer.is_file():
        raise FileNotFoundError(f"shared BPE tokenizer not found: {args.tokenizer}")
    tokenizer = BPETokenizer.load(str(args.tokenizer))
    device = choose_device(args.device)
    LOGGER.info("device=%s vocab=%d", device, tokenizer.num_tokens)

    model = QCloudNetwork(
        vocab_size=tokenizer.num_tokens,
        d_model=args.d_model,
        d_cloud=args.d_cloud,
        d_cloud_dim=args.d_cloud_dim,
        n_layers=args.n_layers,
    ).to(device)
    if args.condition == "pair-freeze":
        for parameter in model.word_head.parameters():
            parameter.requires_grad_(False)

    trainer = QCloudTrainer(
        model=model,
        corpus=train_texts,
        eval_corpus=test_texts,
        device=device,
        tokenizer=tokenizer,
        max_epochs=args.epochs,
        eval_interval=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.minibatch,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        raw_examples = trainer._get_examples()
        test_examples = trainer._get_eval_examples()
    examples = [
        (torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long))
        for input_ids, target_id in raw_examples
    ]
    if not examples:
        raise ValueError("training split yielded no examples at the selected sequence length")
    if not test_examples:
        raise ValueError("held-out test split yielded no examples at the selected sequence length")

    config = QUANNConfig(
        seed=args.seed,
        k_weights=args.k_weights,
        minibatch=args.minibatch,
    )
    engine = QUANN(model, device, cfg=config, loss_fn=make_loss_fn(device))
    if args.condition == "pair-freeze":
        pair_parameter_count = route_pair_channel(engine)
        LOGGER.info("pair-channel selection enabled for %d parameters", pair_parameter_count)
    candidate_deltas = instrument_step_deltas(engine)
    engine.fit(examples, epochs=args.epochs, verbose=False)

    if args.condition == "pair-freeze":
        finetune_word_head(
            model, examples, device, args.seed, args.head_finetune_epochs,
            args.head_lr, args.minibatch,
        )

    if not candidate_deltas:
        raise RuntimeError("QUANN produced no candidate loss evaluations; delta_cand is undefined")
    delta_cand = sum(candidate_deltas) / len(candidate_deltas)

    # QCloudTrainer.evaluate_ppl is the sole final test evaluation. Adapt its
    # point source so it measures the parameters actually trained by QUANN.
    use_contextual_points_for_eval(model)
    heldout_ppl = trainer.evaluate_ppl(batch=args.minibatch)
    if not math.isfinite(heldout_ppl):
        raise RuntimeError("held-out PPL is not finite")
    return len(train_texts), delta_cand, heldout_ppl


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; emit one machine-readable result line on success."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                        stream=sys.stderr)
    args = parse_args(argv)
    try:
        n_train, delta_cand, heldout_ppl = run(args)
    except Exception:
        LOGGER.exception("R8 condition failed")
        return 1
    print(
        f"R8_RESULT | condition={args.condition} | epochs={args.epochs} "
        f"| delta_cand={delta_cand:.8g} | heldout_PPL={heldout_ppl:.8g} "
        f"| split={args.split} | seed={args.seed} | n_train={n_train}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
