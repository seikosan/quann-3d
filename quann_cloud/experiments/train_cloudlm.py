"""
CloudLM — Tam Ölçekli Eğitim (temiz BPE + gerçek Türkçe korpus)
================================================================

CloudLM nokta-bulut LM'yi backprop ile eğitir, gerçek PPL + üretim kalitesi
ölçer, model + BPE kaydeder. Edge paketleme (INT8/ONNX) ayrı scriptte
(`package_cloudlm.py`).

Çalıştır:
    PYTORCH_MPS_ACTIVE_MEMORY_CAPACITY=60 python3 quann_cloud/experiments/train_cloudlm.py \
        --corpus turkish_sentences.txt --max-texts 60000 --bpe-vocab 20000 \
        --epochs 4 --d-model 256 --d-cloud 512 --d-dim 32 --layers 4 --batch 64
"""

import os
import sys
import math
import time
import random
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
from train_wiki import CloudLM, BPETokenizer, load_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--max-texts", type=int, default=60000)
    ap.add_argument("--bpe-vocab", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-cloud", type=int, default=512)
    ap.add_argument("--d-dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--eval-sent", type=int, default=400)
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "quann_cloud", "checkpoints"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device or
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 66)
    print("  CloudLM — Tam Ölçekli Eğitim")
    print("=" * 66)
    print(f"  corpus={os.path.basename(args.corpus)} max_texts={args.max_texts} "
          f"epochs={args.epochs} device={device}")
    print(f"  model: d_model={args.d_model} d_cloud={args.d_cloud} D={args.d_dim} "
          f"layers={args.layers} | vocab={args.bpe_vocab}")

    # ---- Veri ----
    all_texts = load_txt(args.corpus, max_lines=args.max_texts + args.eval_sent + 2000,
                         min_len=15, max_len=300)
    random.shuffle(all_texts)
    train_texts = all_texts[:args.max_texts]
    eval_texts = all_texts[args.max_texts:args.max_texts + args.eval_sent]
    print(f"  Veri: {len(train_texts)} train + {len(eval_texts)} eval")

    # ---- BPE ----
    bpe_path = os.path.join(args.save_dir, "cloudlm_bpe.json")
    if os.path.exists(bpe_path):
        tok = BPETokenizer.load(bpe_path)
        print(f"  BPE yüklendi: {tok.num_tokens} token")
    else:
        t0 = time.time()
        tok = BPETokenizer(args.bpe_vocab)
        tok.train(all_texts)
        tok.save(bpe_path)
        print(f"  BPE eğitildi: {tok.num_tokens} token ({time.time()-t0:.0f}s)")
    eos = tok._tokenizer.token_to_id("<eos>") or 1

    # ---- Örnekler ----
    train_ex = []
    for t in train_texts:
        ids = tok.encode(t)
        for i in range(len(ids) - args.seq_len):
            train_ex.append((ids[i:i + args.seq_len], ids[i + args.seq_len]))
    print(f"  Eğitim örneği: {len(train_ex)}")

    # ---- Model ----
    model = CloudLM(tok.num_tokens, args.d_model, args.d_cloud, args.d_dim,
                    args.layers).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n:,} param")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def eval_ppl(max_n=400):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for t in eval_texts[:max_n]:
                ids = tok.encode(t)
                for i in range(len(ids) - args.seq_len):
                    inp = torch.tensor([ids[i:i + args.seq_len]], device=device)
                    tgt = torch.tensor([ids[i + args.seq_len]], device=device)
                    tot += F.cross_entropy(model(inp, inp != 0)[0], tgt).item()
                    cnt += 1
        return math.exp(min(tot / max(cnt, 1), 20))

    best_ppl = float("inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_ex)
        loss_sum, nb, t0 = 0.0, 0, time.time()
        for b in range(0, len(train_ex), args.batch):
            be = train_ex[b:b + args.batch]
            if len(be) < 2:
                continue
            inp = torch.tensor([e[0] + [0] * (args.seq_len - len(e[0])) for e in be], device=device)
            tgt = torch.tensor([e[1] for e in be], device=device)
            optimizer.zero_grad()
            logits, _, _ = model(inp, inp != 0)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(be); nb += len(be)
        scheduler.step()
        avg = loss_sum / max(nb, 1)
        tr_ppl = math.exp(min(avg, 20))
        vppl = eval_ppl()
        elapsed = time.time() - t_start
        print(f"  E{epoch}/{args.epochs} | train_ppl={tr_ppl:.2f} eval_ppl={vppl:.2f} "
              f"| {elapsed:.0f}s", flush=True)

        if vppl < best_ppl:
            best_ppl = vppl
            torch.save({"model_state": model.state_dict(), "vocab_size": tok.num_tokens,
                        "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                                   "d_cloud_dim": args.d_dim, "n_layers": args.layers},
                        "epoch": epoch, "ppl": vppl},
                       os.path.join(args.save_dir, "cloudlm_best.pt"))
            print(f"    → best kaydedildi (PPL {vppl:.2f})")

    # ---- Son üretim + kaydet ----
    model.eval()
    print(f"\n  🧪 ÜRETİM (best PPL {best_ppl:.2f}):")
    prompts = ["Türkiye'nin başkenti", "Bilim ve teknoloji", "Deniz kenarında", "Eğitim sistemi"]
    gen = {}
    for p in prompts:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = tok.decode(model.generate(ids, max_new=15, temperature=0.8, eos_id=eos)[0].tolist())
        gen[p] = out
        print(f"    {p:22s} → {out}")

    out = {"device": str(device), "corpus": os.path.basename(args.corpus),
           "max_texts": args.max_texts, "epochs": args.epochs, "vocab": tok.num_tokens,
           "params": n, "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                                   "d_dim": args.d_dim, "n_layers": args.layers},
           "best_ppl": best_ppl, "final_eval_ppl": eval_ppl(),
           "generation": gen, "total_time_s": time.time() - t_start}
    with open(os.path.join(args.save_dir, "train_result.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Tamam. Kaydedildi: {args.save_dir}")


if __name__ == "__main__":
    main()
