"""Fast CPU/GPU smoke test for model construction, inference, and generation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextcloud.model import ContextCloudLM


logger = logging.getLogger(__name__)


def choose_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU according to runtime availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_tokens(
    model: ContextCloudLM,
    prompt: torch.Tensor,
    max_new_tokens: int = 4,
    eos_id: int = 2,
    padding_idx: int = 0,
) -> torch.Tensor:
    """Greedily append tokens by repeatedly evaluating the model's next-token head."""
    if prompt.ndim != 2 or prompt.size(0) != 1:
        raise ValueError("prompt must have shape (1, tokens)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if padding_idx < 0 or padding_idx >= model.vocab_size:
        raise ValueError("padding_idx must be a valid vocabulary index")
    generated = prompt.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            logits[:, padding_idx] = float("-inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if int(next_token.item()) == eos_id:
                break
    return generated


def main() -> None:
    """Run a tiny forward pass and short greedy generation smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
        padding_idx=0,
        max_tokens=16,
    ).to(device)
    prompt = torch.tensor([[3, 7, 11]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(prompt)
    generated = generate_tokens(model, prompt, max_new_tokens=4)
    if logits.shape != (1, 32) or not torch.isfinite(logits).all():
        raise RuntimeError("ContextCloudLM smoke-test forward pass failed")
    logger.info("device=%s logits_shape=%s generated=%s", device, tuple(logits.shape), generated.tolist())


if __name__ == "__main__":
    main()
