"""
Tokenizer'lar
=============
- Tok:          kelime tabanlı (özel token'lar: <pad>=0, <eos>=1, <unk>=2)
- BPETokenizer: BPE (tokenizers kütüphanesi; <pad>/<eos>/<unk> special)
"""

import json
import os
from collections import Counter
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class _TokenMap(dict):
    """Dictionary lookup that also supports the historical ``stoi(word)`` API."""

    def __call__(self, word: str) -> int:
        return self.get(word, 2)


class Tok:
    """Kelime tabanlı tokenizer.

    itos: id → kelime (int → str) · stoi: kelime → id (str → int)

    punct:  noktalama (".", ",", "!", "?") → boşluk/çıkar (sentetik corpus demoları)
    fold_i: "ı" → "i" (Türkçe normalizasyon)
    """

    PUNCT = (".", ",", "!", "?")

    def __init__(self, punct: bool = False, fold_i: bool = False):
        self.itos = {0: "<pad>", 1: "<eos>", 2: "<unk>"}
        self.stoi = _TokenMap({"<pad>": 0, "<eos>": 1, "<unk>": 2})
        self.punct = punct
        self.fold_i = fold_i

    def _norm(self, text: str, sep: str) -> str:
        t = text.lower()
        if self.fold_i:
            t = t.replace("ı", "i")
        if self.punct:
            for p in self.PUNCT:
                t = t.replace(p, sep)
        return t

    def build(self, texts: List[str], max_vocab: int = 8000):
        counts = Counter()
        for t in texts:
            for w in self._norm(t, "").split():
                if len(w) > 1:
                    counts[w] += 1
        for word, _ in counts.most_common(max_vocab):
            if word not in self.stoi:
                self.itos[len(self.itos)] = word
                self.stoi[word] = len(self.itos) - 1
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, 2) for w in self._norm(text, " ").split() if w] + [1]

    def decode(self, tokens) -> str:
        return " ".join(self.itos.get(t, "?") for t in tokens if t not in (0, 1))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def itos_get(self, i: int) -> str:
        return self.itos.get(i, "?")

    def save(self, path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "itos": self.itos,
                "stoi": self.stoi,
                "normalization": {"punct": self.punct, "fold_i": self.fold_i},
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Tok":
        """Eski + yeni format uyumlu yükleme.

        JSON int anahtarları string'e çevirdiği için (ve eski dosyalarda
        itos/stoi ters yönde kaydedilmiş olabileceği için) iki kayıttan da
        (id, kelime) çiftlerini toplayıp dict'leri yeniden kurar.
        """
        with open(path, "r") as f:
            data = json.load(f)
        normalization = data.get("normalization", {})
        tok = cls(
            punct=normalization.get("punct", False),
            fold_i=normalization.get("fold_i", False),
        )
        pairs = {}
        for k, v in data.get("itos", {}).items():
            if isinstance(v, int) and not str(k).isdigit():
                pairs[v] = k              # "<pad>" → 0
            elif isinstance(v, str) and str(k).isdigit():
                pairs[int(k)] = v         # "3" → "kedi"
        for k, v in data.get("stoi", {}).items():
            if isinstance(v, int) and not str(k).startswith("<"):
                pairs[v] = k              # "kedi" → 3
            elif isinstance(v, str) and str(k).isdigit():
                pairs[int(k)] = v         # "0" → "<pad>"
        pairs.setdefault(0, "<pad>")
        pairs.setdefault(1, "<eos>")
        pairs.setdefault(2, "<unk>")
        tok.itos = pairs
        tok.stoi = _TokenMap({w: i for i, w in pairs.items()})
        return tok


class BPETokenizer:
    """BPE tokenizer sarmalayıcısı (train_wiki pipeline'ınınki)."""

    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self._tokenizer = None

    def train(self, texts: List[str], vocab_size: int = None):
        if vocab_size:
            self.vocab_size = vocab_size
        tokenizer = Tokenizer(models.BPE(vocab={}, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<eos>", "<unk>"],
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer
        return self

    def encode(self, text: str) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer train edilmemiş!")
        ids = self._tokenizer.encode(text).ids
        eos_id = self._tokenizer.token_to_id("<eos>")
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(eos_id)
        return ids

    def decode(self, ids) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def __call__(self, text: str):
        return self._tokenizer.encode(text)

    @property
    def vocab(self) -> dict:
        return self._tokenizer.get_vocab()

    @property
    def num_tokens(self) -> int:
        return len(self._tokenizer.get_vocab())

    def save(self, path: str) -> None:
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        tok._tokenizer = Tokenizer.from_file(path)
        tok.vocab_size = len(tok._tokenizer.get_vocab())
        return tok

    def sample_tokens(self, text: str, n: int = 15) -> str:
        tokens = self._tokenizer.encode(text).tokens
        return " ".join(tokens[:n]) + ("..." if len(tokens) > n else "")
