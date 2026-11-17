"""contextcloud aktif eğitim script'leri için ortak veri setleri.

train_stable (2D tek-cümle görevi) ile train_memory_context / train_global_context /
ablate (3D chunk görevi) arasındaki kopya iskeleti burada birleştirildi.
Hedef sızıntısı bu modülün işi DEĞİL: girdiden hedef çıkarma task_utils
(shift_target_2d/3d) sorumluluğundadır; burada yalnız girdi/padding üretilir.

Semantik notlar (R5 loglarıyla birebir davranış kilidi):
- WordVocabDataset: vocab tam `vocab_size`'a `<extra_N>` placeholder'larıyla
  doldurulur (train_stable geleneği).
- ChunkedVocabDataset: vocab `vocab_size` verildiğinde o sınırla durur,
  None ise sınırsızdır (train_global_context geleneği).
Vocab her zaman YALNIZ train bölmesinden kurulur; val/test aynı dict'i
`vocab=` ile alır (AGENTS §4.1).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "WordVocabDataset",
    "ChunkedVocabDataset",
    "collate_tokens",
    "collate_chunks",
]


def _read_word_lists(sentences) -> list[list[str]]:
    """List[str] veya dosya yolundan cümle-başına-kelime listesi üretir."""
    if isinstance(sentences, str):
        with open(sentences, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    return [line.split() for line in sentences if line.strip()]


class WordVocabDataset(Dataset):
    """2D tek-cümle görevi: kontrollü vocab + sabit uzunlukta padding.

    Eski train_stable.SimpleDataset ile birebir aynı semantik: vocab
    `vocab_size`'a `<extra_N>` ile tamamlanır, örnekler `max_tokens`'a pad'lenir.
    """

    def __init__(
        self,
        sentences,
        max_tokens: int = 128,
        vocab_size: int = 10000,
        vocab: Optional[dict] = None,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.data = _read_word_lists(sentences)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {"<pad>": 0, "<unk>": 1}
            for words in self.data:
                for word in words:
                    if len(self.vocab) < vocab_size and word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
            while len(self.vocab) < vocab_size:
                self.vocab[f"<extra_{len(self.vocab)}>"] = len(self.vocab)

        logger.info("Vocab size: %d, Samples: %d", len(self.vocab), len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        words = self.data[idx][: self.max_tokens]
        token_ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in words]
        token_ids = token_ids + [0] * (self.max_tokens - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)


class ChunkedVocabDataset(Dataset):
    """3D chunk görevi: ardışık `chunk_size` cümle × `num_chunks` blok.

    Eski MemoryContextDataset/GlobalContextDataset birleşik hâli: `vocab_size`
    sınırı memory davranışıdır, None ise global davranışıdır (sınırsız vocab).
    Kayan pencere adımlaması chunk_size başınadır ve son tam blok dahil edilir.
    """

    def __init__(
        self,
        sentences,
        chunk_size: int = 5,
        max_tokens: int = 256,
        num_chunks: int = 4,
        vocab: Optional[dict] = None,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        self.num_chunks = num_chunks
        self.vocab = vocab if vocab is not None else {"<pad>": 0, "<unk>": 1}
        self.data: list[list[list[str]]] = []

        sentences_words = _read_word_lists(sentences)

        if vocab is None:
            for sent in sentences_words:
                for word in sent:
                    if vocab_size is not None and len(self.vocab) >= vocab_size:
                        break
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)

        for i in range(0, len(sentences_words) - chunk_size * (num_chunks - 1), chunk_size):
            sample = []
            for c in range(num_chunks):
                start = i + c * chunk_size
                chunk = sentences_words[start : start + chunk_size]
                tokens = [w for sent in chunk for w in sent][:max_tokens]
                sample.append(tokens)
            if all(len(s) > 0 for s in sample):
                self.data.append(sample)

        logger.info("Vocab size: %d, Samples: %d", len(self.vocab), len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.data[idx]
        token_ids = []
        for chunk in sample:
            ids = [min(self.vocab.get(w, 1), len(self.vocab) - 1) for w in chunk]
            ids = ids + [0] * (self.max_tokens - len(ids))
            token_ids.append(ids)
        return torch.tensor(token_ids, dtype=torch.long)


def collate_tokens(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L) girdi + padding mask; boş olmayan max uzunluğa pad'ler."""
    max_len = max(len(t) for t in batch)
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(batch):
        tokens[i, : len(t)] = t
    mask = tokens.ne(0)
    return tokens, mask


def collate_chunks(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, NC, MT) chunk girdisi + padding mask."""
    max_tokens = max(len(chunk) for sample in batch for chunk in sample)
    batch_chunks = len(batch[0])
    chunks = torch.zeros(len(batch), batch_chunks, max_tokens, dtype=torch.long)
    for i, sample in enumerate(batch):
        for j, chunk in enumerate(sample):
            chunks[i, j, : len(chunk)] = chunk
    masks = chunks.ne(0)
    return chunks, masks
