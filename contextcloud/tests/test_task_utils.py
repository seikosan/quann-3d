"""task_utils regresyon testleri (hedef sızıntısı yok + 80/10/10 bölme)."""

import torch

from contextcloud.task_utils import shift_target_2d, shift_target_3d, split_sentences


def test_shift_target_2d_removes_target_from_input() -> None:
    tokens = torch.tensor([[5, 7, 9, 0, 0], [3, 4, 0, 0, 0]])
    mask = tokens.ne(0)
    clean_tokens, clean_mask, target = shift_target_2d(tokens, mask, vocab_size=10)

    assert target.tolist() == [9, 4]
    assert clean_tokens[0, 2].item() == 0 and not clean_mask[0, 2].item()
    assert clean_tokens[1, 1].item() == 0 and not clean_mask[1, 1].item()
    assert tokens[0, 2].item() == 9, "girdi tensörü değiştirilmemeli"


def test_shift_target_2d_empty_row_target_zero() -> None:
    tokens = torch.zeros(1, 4, dtype=torch.long)
    mask = tokens.ne(0)
    _, _, target = shift_target_2d(tokens, mask)
    assert target.tolist() == [0]


def test_shift_target_3d_removes_target_from_last_chunk() -> None:
    chunks = torch.tensor([[[1, 2, 0], [3, 4, 5]]])
    masks = chunks.ne(0)
    clean_chunks, clean_masks, target = shift_target_3d(chunks, masks)

    assert target.tolist() == [5]
    assert clean_chunks[0, -1, 2].item() == 0 and not clean_masks[0, -1, 2].item()
    assert chunks[0, -1, 2].item() == 5, "girdi tensörü değiştirilmemeli"


def test_shift_target_3d_clamps_to_vocab() -> None:
    chunks = torch.tensor([[[1, 2, 0], [3, 4, 99]]])
    masks = chunks.ne(0)
    _, _, target = shift_target_3d(chunks, masks, vocab_size=50)
    assert target.tolist() == [49]


def test_split_sentences_ratios_and_disjoint() -> None:
    lines = [f"s{i}" for i in range(100)]
    tr, va, te = split_sentences(lines)
    assert len(tr) == 80 and len(va) == 10 and len(te) == 10
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert sorted(tr + va + te) == sorted(lines)


def test_split_sentences_seed42_deterministic() -> None:
    lines = [f"s{i}" for i in range(50)]
    assert split_sentences(lines) == split_sentences(lines)
    assert split_sentences(lines, seed=42) != split_sentences(lines, seed=7)
