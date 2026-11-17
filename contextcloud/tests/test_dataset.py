"""dataset.py regresyon testleri — R5 davranış kilidi.

Kilitlenen semantikler (bkz. modül docstring):
- WordVocabDataset: vocab tam vocab_size'a <extra_N> ile doldurulur (train_stable).
- ChunkedVocabDataset: vocab_size sınırı memory, None sınırsız vocab global
  davranışıdır; kayan blok adımlaması chunk_size başına.
"""

import pytest
import torch

from contextcloud.dataset import (
    ChunkedVocabDataset,
    WordVocabDataset,
    collate_chunks,
    collate_tokens,
)


# --- WordVocabDataset (stable görevi) ---

def test_word_vocab_pads_to_exact_vocab_size() -> None:
    ds = WordVocabDataset(["a b c", "b c d"], max_tokens=4, vocab_size=8)
    assert len(ds.vocab) == 8, "vocab <extra_N> ile tam vocab_size'a dolmalı"
    assert ds.vocab["<pad>"] == 0 and ds.vocab["<unk>"] == 1


def test_word_vocab_val_reuses_train_vocab_without_growth() -> None:
    train = WordVocabDataset(["elma armut"], max_tokens=4, vocab_size=6)
    before = dict(train.vocab)
    val = WordVocabDataset(["kirkeeez yeni"], max_tokens=4, vocab=train.vocab)
    assert val.vocab is train.vocab
    assert len(val.vocab) == len(before), "val vocab'ü büyütmemeli (train-only)"
    assert val[0][0].item() == 1, "OOV kelime <unk>=1 olmalı"


def test_word_vocab_getitem_fixed_length() -> None:
    ds = WordVocabDataset(["tek", "bir iki uc dort bes"], max_tokens=4, vocab_size=8)
    assert len(ds[0]) == 4 and len(ds[1]) == 4
    assert ds[0][1:].eq(0).all(), "kısa örnek sonundan pad'lenmeli"
    assert ds[1].ne(0).all(), "tam uzun örnek pad içermemeli"


# --- ChunkedVocabDataset (memory/global görevi) ---

def _sentences(n: int) -> list[str]:
    return [f"w{i}a w{i}b" for i in range(n)]


def test_chunked_sliding_block_count() -> None:
    # 12 cümle, chunk_size=2, num_chunks=3 → range(0, 12-2*2, 2) → 4 blok
    ds = ChunkedVocabDataset(_sentences(12), chunk_size=2, max_tokens=8, num_chunks=3)
    assert len(ds) == 4


def test_chunked_vocab_size_cap_stops_growth() -> None:
    ds = ChunkedVocabDataset(_sentences(12), chunk_size=2, max_tokens=8,
                             num_chunks=3, vocab_size=4)
    assert len(ds.vocab) == 4, "memory davranışı: vocab_cap'te durmalı"


def test_chunked_vocab_unlimited_when_none() -> None:
    ds = ChunkedVocabDataset(_sentences(12), chunk_size=2, max_tokens=8, num_chunks=3)
    assert len(ds.vocab) == 2 + 24, "global davranışı: sınır yok, tüm kelimeler"


def test_chunked_val_shares_vocab() -> None:
    train = ChunkedVocabDataset(_sentences(6), chunk_size=2, max_tokens=8, num_chunks=2)
    val = ChunkedVocabDataset(["baska sozcukler"], chunk_size=1, max_tokens=8,
                              num_chunks=2, vocab=train.vocab)
    assert len(val.vocab) == len(train.vocab), "val vocab'ü büyütmemeli"


# --- collate fonksiyonları ---

def test_collate_tokens_mask_matches_padding() -> None:
    batch = [torch.tensor([3, 5, 0, 0]), torch.tensor([7, 8, 9, 0])]
    tokens, mask = collate_tokens(batch)
    assert tokens.shape == (2, 4)
    assert torch.equal(mask, tokens.ne(0))
    assert tokens[:, 3].eq(0).all()


def test_collate_chunks_shape_and_mask() -> None:
    b = torch.arange(1, 13).reshape(3, 4).tolist()
    batch = [torch.tensor(b), torch.tensor(b)]
    chunks, masks = collate_chunks(batch)
    assert chunks.shape == (2, 3, 4)
    assert torch.equal(masks, chunks.ne(0))
    assert chunks[0, 0, 0].item() == 1


@pytest.mark.parametrize("batch_size", [1, 3])
def test_chunked_dataloader_roundtrip(batch_size: int) -> None:
    from torch.utils.data import DataLoader

    ds = ChunkedVocabDataset(_sentences(12), chunk_size=2, max_tokens=6, num_chunks=2)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_chunks)
    chunks, masks = next(iter(loader))
    assert chunks.shape[0] == batch_size
    assert chunks.shape[1] == 2
    assert masks.shape == chunks.shape
