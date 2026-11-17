"""Canonical PureCloudLM eğitmeni — 21 tek-atışlık script'in birleşik hâli.

Arşivlenen scriptlerin (bkz. `_archive/refactor_2026_11/purecloudlm/`) ortak
iskeleti: corpus yükleme (dosya başına cap, toplam cap) → BPE encode + uzunluk
filtresi → shuffle(seed=42) → batch döngüsü (NaN bekçisi, gradyan kırpma,
cosine plan, nokta-uzayı yardımcı kaybı) → en iyi PPL checkpoint'i + erken
durdurma. Hyperparameter farkları `TrainConfig` üzerinden CLI ile verilir.

Checkpoint şeması arşiv scriptleriyle birebirdir (`model_state`, `config` …);
`eval_test.py` bunu `checkpoint["model_state"]` olarak okur.

Örnek (stable reçetesi):
    python3 -u purecloudlm/train.py --d-model 128 --d-cloud 256 --lr 1e-4 --epochs 15
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from cloudlm import BPETokenizer
from purecloudlm import PureCloudLM

logger = logging.getLogger(__name__)

DEFAULT_CORPUS: list[tuple[str, int]] = [
    ("data/turkish_sentences.txt", 200_000),
    ("data/turkish_wiki_1500k.txt", 800_000),
]


@dataclass
class TrainConfig:
    """Eğitim reçetesi; alanlar arşiv script'lerin birleşik seçenekleridir."""

    bpe_path: str = "checkpoints/wiki_checkpoints_200k/wiki_bpe.json"
    corpus: list[tuple[str, int]] = field(
        default_factory=lambda: list(DEFAULT_CORPUS))
    total_cap: int = 1_000_000
    min_len: int = 11
    max_len: int = 256
    seq_len: int = 20
    batch_size: int = 512
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    point_weight: float = 0.1
    d_model: int = 128
    d_cloud: int = 256
    d_cloud_dim: int = 8
    n_layers: int = 1
    hidden: int = 128
    proj_hidden: int = 128
    point_hidden: int = 64
    seed: int = 42
    out_dir: str = "purecloudlm/checkpoints"
    ckpt_name: str = "pure_cloud_best.pt"

    def model_cfg(self, vocab_size: int) -> dict:
        """Checkpoint'e yazılan ve PureCloudLM ctor'una giren model config."""
        return {
            "vocab_size": vocab_size,
            "d_model": self.d_model,
            "d_cloud": self.d_cloud,
            "d_cloud_dim": self.d_cloud_dim,
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "proj_hidden": self.proj_hidden,
            "point_hidden": self.point_hidden,
            "use_pair_approx": False,
            "pool_mode": "all",
        }


def load_corpus(cfg: TrainConfig) -> list[str]:
    """Dosya başına cap + toplam cap ile corpus toplar; yok dosya atlanır."""
    texts: list[str] = []
    for filepath, limit in cfg.corpus:
        if not Path(filepath).exists():
            logger.warning("Corpus dosyası yok, atlanıyor: %s", filepath)
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        take = min(len(lines), max(limit - len(texts), 0))
        texts.extend(lines[:take])
        logger.info("  %s: %d cümle", Path(filepath).name, take)
        if len(texts) >= cfg.total_cap:
            break
    texts = texts[: cfg.total_cap]
    logger.info("Toplam: %d cümle", len(texts))
    return texts


def encode_filter(tok: BPETokenizer, texts: Iterable[str],
                  cfg: TrainConfig) -> list[list[int]]:
    """Encode eder ve `min_len <= len <= max_len` filtreler (tüm script'lerin kuralı)."""
    out: list[list[int]] = []
    total = 0
    for text in texts:
        encoded = tok.encode(text)
        total += 1
        if cfg.min_len <= len(encoded) <= cfg.max_len:
            out.append(encoded)
        if total % 200_000 == 0:
            logger.info("  %d / %d encode edildi (%d geçerli)",
                        total, len(texts) if hasattr(texts, "__len__") else total, len(out))
    logger.info("Geçerli cümle: %d", len(out))
    return out


def make_batch(batch_tokens: list[list[int]], seq_len: int, eos_id: int,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(inputs, targets, mask): ilk seq_len token girdi, sonraki token hedef."""
    batch_inputs: list[list[int]] = []
    batch_targets: list[int] = []
    for t in batch_tokens:
        batch_inputs.append(t[:seq_len] + [0] * max(0, seq_len - len(t)))
        batch_targets.append(t[seq_len] if len(t) > seq_len else 0)
    inputs = torch.tensor(batch_inputs, device=device, dtype=torch.long)
    targets = torch.tensor(batch_targets, device=device, dtype=torch.long)
    mask = (inputs != 0) & (inputs != eos_id)
    return inputs, targets, mask


def train_epoch(model: PureCloudLM, all_tokens: list[list[int]],
                optimizer: torch.optim.Optimizer, device: torch.device,
                cfg: TrainConfig, eos_id: int) -> tuple[float, int, int]:
    """Tek epoch; (ortalama word loss, NaN atlanan batch, ölçülen batch) döner.

    Adımlama: shuffle YOKTUR (epoch başlangıcında bir kez çağırılır); NaN loss
    backward YAPILMADAN atlanır (arşiv "stable" reçetelerinin davranışı).
    """
    model.train()
    epoch_loss, n_batches, nan_count = 0.0, 0, 0
    for i in range(0, len(all_tokens), cfg.batch_size):
        inputs, targets, mask = make_batch(
            all_tokens[i : i + cfg.batch_size], cfg.seq_len, eos_id, device)

        optimizer.zero_grad()
        logits, next_point, _ = model(inputs, mask)
        word_loss = F.cross_entropy(logits, targets)

        target_embedding = model.embedding(targets) * math.sqrt(
            model.embedding.embedding_dim)
        target_points = model.to_nd(target_embedding.unsqueeze(1))[:, 0]
        point_loss = F.mse_loss(next_point, target_points)
        loss = word_loss + cfg.point_weight * point_loss

        if not torch.isfinite(loss):
            nan_count += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if torch.isfinite(word_loss):
            epoch_loss += word_loss.item()
            n_batches += 1
    avg = epoch_loss / n_batches if n_batches else float("inf")
    return avg, nan_count, n_batches


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PureCloudLM eğitim reçetesi")
    p.add_argument("--bpe", default=None, help="BPE json yolu (vars: checkpoints/wiki_checkpoints_200k/wiki_bpe.json)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-cloud", type=int, default=256)
    p.add_argument("--d-cloud-dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--point-weight", type=float, default=0.1)
    p.add_argument("--total-cap", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="purecloudlm/checkpoints")
    p.add_argument("--ckpt-name", default="pure_cloud_best.pt")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    cfg = TrainConfig(
        bpe_path=args.bpe or TrainConfig().bpe_path,
        d_model=args.d_model, d_cloud=args.d_cloud, d_cloud_dim=args.d_cloud_dim,
        hidden=args.hidden, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs, seq_len=args.seq_len,
        grad_clip=args.grad_clip, point_weight=args.point_weight,
        total_cap=args.total_cap, seed=args.seed,
        out_dir=args.out_dir, ckpt_name=args.ckpt_name,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Cihaz: %s", device)

    texts = load_corpus(cfg)
    tok = BPETokenizer.load(cfg.bpe_path)
    logger.info("BPE: %d token", tok.num_tokens)

    all_tokens = encode_filter(tok, texts, cfg)
    rng = random.Random(cfg.seed)
    rng.shuffle(all_tokens)
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    model_cfg = cfg.model_cfg(tok.num_tokens)
    model = PureCloudLM(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parametre (%.1fM)", n_params, n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ppl = float("inf")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        avg, nan_count, n_batches = train_epoch(
            model, all_tokens, optimizer, device, cfg, eos_id)
        scheduler.step()
        if n_batches == 0:
            logger.warning("Epoch %d tamamen NaN — eğitim durduruluyor", epoch)
            break
        ppl = math.exp(min(avg, 20))
        logger.info("Epoch %d/%d | loss=%.4f | ppl=%.2f | NaN=%d | %.1f dk",
                    epoch, cfg.epochs, avg, ppl, nan_count, (time.time() - t0) / 60)

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_ppl": best_ppl,
                "config": model_cfg,
                "corpus_size": len(all_tokens),
                "total_time": time.time() - t0,
            }, out_dir / cfg.ckpt_name)
            logger.info("✓ En iyi checkpoint (ppl=%.2f)", best_ppl)

    logger.info("BİTTİ — en iyi ppl %.2f, %.1f dk, %d cümle",
                best_ppl, (time.time() - t0) / 60, len(all_tokens))


if __name__ == "__main__":
    main()
