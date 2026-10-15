"""
ADİL Control: CloudLM (backprop) — hibrit QUANN ile AYNI güncelleme sayısı
===========================================================================

Hibrit QUANN her cümleden sadece İLK pencereyi kullanır (20K örnek/epoch).
Önceki control tüm pencereleri kullanıyordu (~13× fazla güncelleme) — adil değil.

Bu script: aynı CloudLM mimarisi (21M, D=16), aynı corpus (20K),
aynı protokol (ilk pencere → next token), 10 epoch.
"""

import sys
import os
import math
import time
import random
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import argparse
from train_wiki import BPETokenizer, CloudLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-texts", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--d-dim", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-cloud", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--save-dir", type=str,
                    default=os.path.join(ROOT, "quann_cloud", "experiments", "control_20k_single"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json"))
    print(f"Vocab: {tok.num_tokens}")

    with open(os.path.join(ROOT, "data", "turkish_sentences.txt"), 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    texts = [t for t in lines if 20 <= len(t) <= 300][:args.max_texts]
    print(f"Corpus: {len(texts)} cümle")

    # Hibritle aynı protokol: ilk pencere → next token
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    examples = []
    for t in texts:
        ids = tok.encode(t)
        if len(ids) < args.seq_len + 1:
            continue
        examples.append((ids[:args.seq_len], ids[args.seq_len]))
    print(f"Örnek: {len(examples)} (ilk pencere)")

    model = CloudLM(tok.num_tokens, d_model=args.d_model, d_cloud=args.d_cloud,
                    d_cloud_dim=args.d_dim, n_layers=args.layers)
    model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_ppl = float('inf')
    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        random.shuffle(examples)
        epoch_loss, n_batches = 0, 0
        for b in range(0, len(examples), args.batch):
            batch = examples[b:b + args.batch]
            if len(batch) < 2:
                continue
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            mask = (inputs != 0) & (inputs != eos_id)
            optimizer.zero_grad()
            logits, _, _ = model(inputs, mask)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | loss={avg:.4f} | ppl={ppl:.2f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({
                "epoch": epoch + 1, "model_state": model.state_dict(),
                "vocab_size": tok.num_tokens,
                "config": {"d_model": args.d_model, "d_cloud": args.d_cloud,
                           "d_cloud_dim": args.d_dim, "n_layers": args.layers},
                "ppl": ppl,
            }, os.path.join(args.save_dir, "wiki_cloud_best.pt"))

    print(f"\n  ✅ {time.time()-t0:.0f}s | Best PPL: {best_ppl:.2f}")


if __name__ == "__main__":
    main()
