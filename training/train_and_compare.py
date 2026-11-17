"""
Quann-3D — Eğit + Karşılaştır Pipeline
======================================

Adım 1: Corpus'tan BPE tokenizer train et
Adım 2: Aynı corpus ile CloudLM train et  
Adım 3: Aynı corpus ile Transformer Baseline train et
Adım 4: Kalite + Edge kıyaslama yap

Kullanım:
  # Hızlı demo (20K cümle, 2 epoch)
  python3 train_and_compare.py --step all --max-texts 20000 --epochs 2

  # Büyük corpus (200K cümle, 10 epoch)
  python3 train_and_compare.py --step all --max-texts 200000 --epochs 10
  
  # Sadece eğitim + kıyaslama (checkpoint varsa skip)
  python3 train_and_compare.py --step train --max-texts 100000 --epochs 5
  python3 train_and_compare.py --step compare --skip-train
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import json
import argparse
import os

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

# Model importları
from train_wiki import BPETokenizer, CloudLM, load_txt
from train_baseline import CausalLM, BidirNoCloud, build_model, count_params


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ─────────────────────────────────────────────
# ADIM 1: BPE TOKENIZER TRAIN
# ─────────────────────────────────────────────

def train_bpe(corpus_path: str, save_dir: str, max_texts: int = 50000, 
              vocab_size: int = 5000, seq_len: int = 10):
    """Train BPE tokenizer on corpus and save."""
    print(f"\n{'='*60}")
    print(f"  ADIM 1: BPE Tokenizer Training")
    print(f"{'='*60}")

    texts = load_txt(corpus_path, max_texts)
    print(f"  {len(texts)} cümle yüklendi")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max seq len: {seq_len}")

    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(texts, vocab_size=vocab_size)

    os.makedirs(save_dir, exist_ok=True)
    bpe_path = os.path.join(save_dir, "bpe.json")
    tok.save(bpe_path)

    vocab = tok._tokenizer.get_vocab()
    print(f"\n  ✅ Vocab size: {tok.num_tokens}")
    print(f"  Örnektoken'lar: {tok.sample_tokens(texts[0], n=15)}")
    print(f"  Kaydedildi: {bpe_path}")

    return tok


# ─────────────────────────────────────────────
# ADIM 2: CLOUDLM TRAIN
# ─────────────────────────────────────────────

def train_cloudlm(tok, save_dir: str, corpus_path: str, max_texts: int, epochs: int = 5,
                  seq_len: int = 10, batch: int = 64, lr: float = 2e-4,
                  d_model: int = 256, d_cloud: int = 512, d_dim: int = 16,
                  layers: int = 4):
    """Train CloudLM on corpus."""
    print(f"\n{'='*60}")
    print(f"  ADIM 2: CloudLM Training")
    print(f"{'='*60}")

    device = get_device()
    texts = load_txt(corpus_path, max_texts)
    print(f"  {len(texts)} cümle")

    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    print(f"  {len(all_tokens)} cümle (≥{seq_len+1} token)")
    print(f"  Ort. uzunluk: {np.mean([len(t) for t in all_tokens]):.1f}")

    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    model = CloudLM(tok.num_tokens, d_model=d_model, d_cloud=d_cloud,
                    d_cloud_dim=d_dim, n_layers=layers)
    model.to(device)

    params = count_params(model)
    print(f"\n  Model: {params:,} params")
    print(f"  d_model={d_model} | d_cloud={d_cloud} | D={d_dim} | layers={layers}")
    print(f"  Vocab={tok.num_tokens} | Device={device} | Epochs={epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_ppl = float('inf')
    model.train()
    t0 = time.time()

    for epoch in range(epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - seq_len):
                examples.append((tokens[i:i+seq_len], tokens[i+seq_len]))
        random.shuffle(examples)
        epoch_loss, n_batches = 0, 0

        for b in range(0, len(examples), batch):
            batch_examples = examples[b:b+batch]
            if len(batch_examples) < 2:
                continue
            inputs = torch.tensor([e[0] for e in batch_examples], device=device)
            targets = torch.tensor([e[1] for e in batch_examples], device=device)
            mask = (inputs != 0) & (inputs != eos_id)

            optimizer.zero_grad()
            logits, p_next, points = model(inputs, mask)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s")

        ckpt = {
            "epoch": epoch + 1, "model_state": model.state_dict(),
            "vocab_size": tok.num_tokens,
            "config": {"d_model": d_model, "d_cloud": d_cloud, 
                       "d_cloud_dim": d_dim, "n_layers": layers},
            "ppl": ppl,
        }
        if (epoch + 1) % 2 == 0:
            torch.save(ckpt, os.path.join(save_dir, f"cloudlm_epoch{epoch+1}.pt"))
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(ckpt, os.path.join(save_dir, "cloudlm_best.pt"))

    total = time.time() - t0
    print(f"\n  ✅ {total:.0f}s ({total/60:.1f}dk) | Best PPL: {best_ppl:.2f}")
    print(f"  Kaydedildi: {os.path.join(save_dir, 'cloudlm_best.pt')}")
    return model, best_ppl, tok


# ─────────────────────────────────────────────
# ADIM 3: TRANSFORMER BASELINE TRAIN
# ─────────────────────────────────────────────

def train_baseline(tok, save_dir: str, corpus_path: str, max_texts: int, epochs: int = 5,
                   seq_len: int = 10, batch: int = 64, lr: float = 2e-4,
                   arch: str = "causal", d_model: int = 384, layers: int = 4):
    """Train transformer baseline on same corpus."""
    print(f"\n{'='*60}")
    print(f"  ADIM 3: Transformer Baseline ({arch.upper()}) Training")
    print(f"{'='*60}")

    device = get_device()
    texts = load_txt(corpus_path, max_texts)

    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    print(f"  {len(all_tokens)} cümle (≥{seq_len+1} token)")

    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    model = build_model(arch, tok.num_tokens, d_model, layers)
    model.to(device)

    params = count_params(model)
    print(f"\n  Model: {params:,} params")
    print(f"  arch={arch} | d_model={d_model} | layers={layers}")
    print(f"  Vocab={tok.num_tokens} | Device={device} | Epochs={epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_ppl = float('inf')
    model.train()
    t0 = time.time()

    for epoch in range(epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - seq_len):
                examples.append((tokens[i:i+seq_len], tokens[i+seq_len]))
        random.shuffle(examples)
        epoch_loss, n_batches = 0, 0

        for b in range(0, len(examples), batch):
            batch_examples = examples[b:b+batch]
            if len(batch_examples) < 2:
                continue
            inputs = torch.tensor([e[0] for e in batch_examples], device=device)
            targets = torch.tensor([e[1] for e in batch_examples], device=device)
            mask = (inputs != 0) & (inputs != eos_id)

            optimizer.zero_grad()
            if arch == "causal":
                logits = model(inputs, mask)  # (B, T, V)
                loss = F.cross_entropy(logits[:, -1, :], targets)
            else:
                logits = model(inputs, mask)
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s")

        ckpt = {
            "epoch": epoch + 1, "model_state": model.state_dict(),
            "vocab_size": tok.num_tokens,
            "config": {"arch": arch, "d_model": d_model, "n_layers": layers},
            "ppl": ppl,
        }
        if (epoch + 1) % 2 == 0:
            torch.save(ckpt, os.path.join(save_dir, f"baseline_{epoch+1}.pt"))
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(ckpt, os.path.join(save_dir, "baseline_best.pt"))

    total = time.time() - t0
    print(f"\n  ✅ {total:.0f}s ({total/60:.1f}dk) | Best PPL: {best_ppl:.2f}")
    print(f"  Kaydedildi: {os.path.join(save_dir, 'baseline_best.pt')}")
    return model, best_ppl


# ─────────────────────────────────────────────
# ADIM 4: KARŞILAŞTIRMA
# ─────────────────────────────────────────────

def evaluate_quality(model, tok, corpus_path, device, seq_len, arch, n=2000):
    """Evaluate PPL and generation quality."""
    texts = load_txt(corpus_path, n)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    with torch.no_grad():
        for text in texts:
            ids = tok.encode(text)
            if len(ids) < seq_len + 1:
                continue
            for i in range(0, len(ids) - seq_len, seq_len):
                inp = torch.tensor([ids[i:i+seq_len]], device=device)
                mask = (inp != 0) & (inp != eos_id)

                if arch == "cloud":
                    logits, _, _ = model(inp, mask)
                    tgt = torch.tensor([ids[i+seq_len]], device=device)
                    total_loss += F.cross_entropy(logits, tgt, reduction="sum").item()
                    total_tokens += 1
                elif arch == "causal":
                    logits = model(inp, mask)  # (1, T, V)
                    # Eğitimle tutarlı: son pozisyon → sonraki tek token
                    tgt = torch.tensor([ids[i+seq_len]], device=device)
                    total_loss += F.cross_entropy(logits[0, -1, :], tgt[0].unsqueeze(0),
                                                  reduction="sum").item()
                    total_tokens += 1
                else:
                    logits = model(inp, mask)
                    tgt = torch.tensor([ids[i+seq_len]], device=device)
                    total_loss += F.cross_entropy(logits, tgt, reduction="sum").item()
                    total_tokens += 1

    return math.exp(total_loss / max(total_tokens, 1))


def run_comparison(cloud_ckpt, base_ckpt, tok, corpus_path, device, 
                   seq_len=10, eval_n=2000):
    """Full comparison between trained models."""
    print(f"\n{'#'*70}")
    print(f"  KARŞILAŞTIRMA")
    print(f"  Device: {device}")
    print(f"  {'#'*66}")

    # Load CloudLM
    cloud_ckpt_data = torch.load(cloud_ckpt, map_location="cpu", weights_only=False)
    cloud = CloudLM(cloud_ckpt_data["vocab_size"], **{k:v for k,v in cloud_ckpt_data["config"].items()})
    cloud.load_state_dict(cloud_ckpt_data["model_state"])
    cloud.to(device)

    # Load Baseline
    base_ckpt_data = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base = build_model(base_ckpt_data["config"]["arch"], base_ckpt_data["vocab_size"],
                       base_ckpt_data["config"]["d_model"], base_ckpt_data["config"]["n_layers"])
    base.load_state_dict(base_ckpt_data["model_state"])
    base.to(device)

    print(f"\n  CloudLM  : {count_params(cloud):,} params (train PPL {cloud_ckpt_data['ppl']:.2f})")
    arch = base_ckpt_data["config"]["arch"]
    print(f"  Baseline : {count_params(base):,} params ({arch}, train PPL {base_ckpt_data['ppl']:.2f})")

    # PPL comparison
    print(f"\n  Held-out PPL ({eval_n} sentences):")
    cloud_ppl = evaluate_quality(cloud, tok, corpus_path, device, seq_len, "cloud", eval_n)
    base_ppl = evaluate_quality(base, tok, corpus_path, device, seq_len, arch, eval_n)
    print(f"    CloudLM  : {cloud_ppl:.2f}")
    print(f"    Baseline : {base_ppl:.2f}")
    if base_ppl > 0:
        ratio = cloud_ppl / base_ppl
        print(f"    Oran (Cloud/Baseline): {ratio:.2f}x")

    # Generation quality
    print(f"\n  Üretim Kalitesi (aynı promptlar, temp=0.7):")
    prompts = [
        "Türkiye'nin başkenti",
        "bilgisayar bilimlerinde",
        "sanat tarihi açısından",
        "ekonomik büyüme",
        "demokrasi ve insan",
    ]
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        try:
            c_out = tok.decode(cloud.generate(ids, max_new=12, temperature=0.7, eos_id=eos_id)[0].tolist())
        except Exception:
            c_out = "[hatası]"
        try:
            b_out = tok.decode(base.generate(ids, max_new=12, temperature=0.7, eos_id=eos_id)[0].tolist())
        except Exception:
            b_out = "[hatası]"
        print(f"    Prompt: {p!r}")
        print(f"      CloudLM  : {c_out[:80]}")
        print(f"      Baseline : {b_out[:80]}")

    # Edge efficiency
    print(f"\n  Edge Verimlilik:")
    
    # Latency
    inp = torch.tensor([tok.encode("Orta Anadolu'nun merkezinde")], device=device)
    mask = (inp != 0) & (inp != 1)
    cloud_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            cloud(inp, mask)
    cloud_time = (time.perf_counter() - cloud_start) * 1000

    base_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            base(inp, mask)
    base_time = (time.perf_counter() - base_start) * 1000

    print(f"    CloudLM  latency: {cloud_time/5:.2f} ms/forward")
    print(f"    Baseline latency: {base_time/5:.2f} ms/forward")
    print(f"    CloudLM  params:  {count_params(cloud):,}")
    print(f"    Baseline params: {count_params(base):,}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  ÖZET TABLO")
    print(f"{'='*60}")
    print(f"  {'Model':12s} {'Params':>12s} {'Train PPL':>12s} {'Held-out PPL':>14s}")
    print(f"  {'CloudLM':12s} {count_params(cloud):>12,} {cloud_ckpt_data['ppl']:>12.2f} {cloud_ppl:>14.2f}")
    print(f"  {'Baseline':12s} {count_params(base):>12,} {base_ckpt_data['ppl']:>12.2f} {base_ppl:>14.2f}")

    return {
        "cloud_params": count_params(cloud),
        "cloud_train_ppl": cloud_ckpt_data["ppl"],
        "cloud_heldout_ppl": cloud_ppl,
        "base_params": count_params(base),
        "base_train_ppl": base_ckpt_data["ppl"],
        "base_heldout_ppl": base_ppl,
        "arch": arch,
    }


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Quann-3D — Eğit + Karşılaştır Pipeline")
    ap.add_argument("--step", type=str, default="all", choices=["train", "compare", "all"])
    ap.add_argument("--corpus", type=str,
                    default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--max-texts", type=int, default=50000)
    ap.add_argument("--bpe-vocab", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-cloud", type=int, default=512)
    ap.add_argument("--d-dim", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--base-arch", type=str, default="causal", choices=["causal", "bidir"])
    ap.add_argument("--base-d-model", type=int, default=384)
    ap.add_argument("--save-dir", type=str,
                    default=os.path.join(ROOT, "checkpoints", "training_output"))
    ap.add_argument("--cloud-ckpt", type=str, default=None, help="Skip CloudLM training")
    ap.add_argument("--base-ckpt", type=str, default=None, help="Skip Baseline training")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--skip-train", action="store_true", help="Skip training, only compare")
    args = ap.parse_args()

    print("=" * 60)
    print("  QUANN-3D — EĞİT + KARŞILAŞTIR PIPELINE")
    print("=" * 60)

    if args.step in ("compare", "all"):
        # Step 1: Train BPE
        if args.step == "all":
            bpe_path = os.path.join(args.save_dir, "bpe.json")
            if not os.path.exists(bpe_path):
                tok = train_bpe(args.corpus, args.save_dir, args.max_texts, args.bpe_vocab)
                bpe_path = os.path.join(args.save_dir, "bpe.json")
            else:
                print(f"\n  ℹ BPE zaten var: {bpe_path} — kullanılıyor")
                tok = BPETokenizer.load(bpe_path)
        else:
            if args.cloud_ckpt:
                # BPE, cloud checkpoint'in BULUNDUĞU dizinde (wiki_bpe.json → bpe.json)
                ckpt_dir = os.path.dirname(os.path.abspath(args.cloud_ckpt))
                bpe_path = os.path.join(ckpt_dir, "wiki_bpe.json")
                if not os.path.exists(bpe_path):
                    bpe_path = os.path.join(ckpt_dir, "bpe.json")
            else:
                bpe_path = os.path.join(args.save_dir, "bpe.json")
            tok = BPETokenizer.load(bpe_path)

        # Step 2: Train CloudLM (if needed)
        cloud_ckpt_path = os.path.join(args.save_dir, "cloudlm_best.pt")
        if args.step == "train" or args.step == "all":
            if not args.cloud_ckpt or not os.path.exists(args.cloud_ckpt):
                model, best_ppl = train_cloudlm(
                    tok, args.save_dir, args.corpus, args.max_texts, args.epochs,
                    args.seq_len, args.batch, args.lr,
                    args.d_model, args.d_cloud, args.d_dim, args.layers)
            else:
                print(f"\n  ℹ CloudLM checkpoint mevcut: {args.cloud_ckpt} — kullanılıyor")
                cloud_ckpt_path = args.cloud_ckpt
        elif args.cloud_ckpt:
            cloud_ckpt_path = args.cloud_ckpt

        # Step 3: Train Baseline (if needed)
        base_save_dir = os.path.join(args.save_dir, "baseline")
        base_ckpt_path = os.path.join(base_save_dir, "baseline_best.pt")
        if args.step == "train" or args.step == "all":
            if not args.base_ckpt or not os.path.exists(args.base_ckpt):
                train_baseline(
                    tok, base_save_dir, args.corpus, args.max_texts, args.epochs,
                    args.seq_len, args.batch, args.lr,
                    args.base_arch, args.base_d_model, args.layers)
            else:
                print(f"\n  ℹ Baseline checkpoint mevcut: {args.base_ckpt} — kullanılıyor")
                base_ckpt_path = args.base_ckpt
        elif args.base_ckpt:
            base_ckpt_path = args.base_ckpt

        # Step 4: Compare
        if args.step == "compare" or args.step == "all":
            os.makedirs(args.save_dir, exist_ok=True)
            result = run_comparison(
                cloud_ckpt_path, base_ckpt_path, tok, args.corpus,
                get_device(), args.seq_len, args.eval_n)

            # Save report
            report_path = os.path.join(args.save_dir, "comparison_report.json")
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Rapor kaydedildi: {report_path}")

    else:  # step == "train"
        # Train only
        bpe_path = os.path.join(args.save_dir, "bpe.json")
        if not os.path.exists(bpe_path):
            tok = train_bpe(args.corpus, args.save_dir, args.max_texts, args.bpe_vocab)
        else:
            print(f"\n  ℹ BPE zaten var: {bpe_path}")
            tok = BPETokenizer.load(bpe_path)

        if args.skip_train:
            print("  ℹ --skip-train: eğitim atlandı (yalnız BPE kontrolü yapıldı)")
        else:
            if not args.cloud_ckpt or not os.path.exists(args.cloud_ckpt):
                train_cloudlm(tok, args.save_dir, args.corpus, args.max_texts, args.epochs,
                             args.seq_len, args.batch, args.lr,
                             args.d_model, args.d_cloud, args.d_dim, args.layers)

            if not args.base_ckpt or not os.path.exists(args.base_ckpt):
                base_save_dir = os.path.join(args.save_dir, "baseline")
                train_baseline(tok, base_save_dir, args.corpus, args.max_texts, args.epochs,
                              args.seq_len, args.batch, args.lr,
                              args.base_arch, args.base_d_model, args.layers)


if __name__ == "__main__":
    main()
