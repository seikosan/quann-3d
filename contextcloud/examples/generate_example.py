"""Run a short ContextCloudLM autoregressive-generation example."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextcloud.model import ContextCloudLM


def choose_device() -> torch.device:
    """Select an available accelerator, falling back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Generate token IDs from a prompt using a small randomly initialized model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = choose_device()
    model = ContextCloudLM(
        vocab_size=32,
        d_model=16,
        d_sentence=16,
        d_window=16,
        d_cloud_dim=4,
        d_window_dim=4,
        proj_hidden=16,
        window_size=2,
        window_stride=1,
        hidden=16,
        max_tokens=16,
    ).to(device)
    prompt = torch.tensor([[3, 7, 11]], dtype=torch.long, device=device)
    generated = model.generate(
        prompt,
        max_new=4,
        temperature=0.8,
        eos_id=1,
        top_k=10,
    )
    logging.info("device=%s prompt=%s generated=%s", device, prompt.tolist(), generated.tolist())
    logging.info("This demo model has random weights; load a trained checkpoint for useful text.")


if __name__ == "__main__":
    main()
