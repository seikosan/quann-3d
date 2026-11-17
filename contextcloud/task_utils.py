"""Sızıntısız tek-hedef görev yardımcıları.

Eski görev tanımı hedef token'ı model girdisinin İÇİNDE bırakıyordu
(kopya yolu → anlamsız loss). Bu modül hedefi girdiden çıkarır:
girdi = hedef hariç geçerli tokenlar, hedef = son geçerli token.
"""

from __future__ import annotations

import random
from typing import Optional

import torch


def split_sentences(
    lines: list[str], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """AGENTS §4.1: 80/10/10 train/val/test, seed=42; satır düzeyinde ayrık bölme."""
    idx = list(range(len(lines)))
    random.Random(seed).shuffle(idx)
    n = len(lines)
    tr, va = int(n * train_ratio), int(n * val_ratio)
    return (
        [lines[i] for i in idx[:tr]],
        [lines[i] for i in idx[tr:tr + va]],
        [lines[i] for i in idx[tr + va:]],
    )


def shift_target_2d(
    tokens: torch.Tensor, mask: torch.Tensor, vocab_size: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, T) girdide son geçerli token'ı hedefe çevirir ve girdiden siler.

    Returns:
        (clean_tokens, clean_mask, targets) — girdi tensörleri değiştirilmez.
    """
    clean_tokens = tokens.clone()
    clean_mask = mask.clone()
    targets = []
    for i in range(tokens.size(0)):
        valid = torch.where(clean_mask[i])[0]
        if valid.numel() > 0:
            pos = int(valid[-1].item())
            target_id = int(tokens[i, pos].item())
            targets.append(min(target_id, vocab_size - 1) if vocab_size else target_id)
            clean_tokens[i, pos] = 0
            clean_mask[i, pos] = False
        else:
            targets.append(0)
    return clean_tokens, clean_mask, torch.tensor(targets, device=tokens.device)


def shift_target_3d(
    chunks: torch.Tensor, masks: torch.Tensor, vocab_size: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, NC, MT) chunk girdisinde SON chunk'ın son geçerli token'ını hedef yapar.

    Returns:
        (clean_chunks, clean_masks, targets) — girdi tensörleri değiştirilmez.
    """
    clean_chunks = chunks.clone()
    clean_masks = masks.clone()
    targets = []
    for i in range(chunks.size(0)):
        valid = torch.where(clean_masks[i, -1, :])[0]
        if valid.numel() > 0:
            pos = int(valid[-1].item())
            target_id = int(chunks[i, -1, pos].item())
            targets.append(min(target_id, vocab_size - 1) if vocab_size else target_id)
            clean_chunks[i, -1, pos] = 0
            clean_masks[i, -1, pos] = False
        else:
            targets.append(0)
    return clean_chunks, clean_masks, torch.tensor(targets, device=chunks.device)
