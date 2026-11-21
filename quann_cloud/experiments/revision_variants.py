"""
CloudLM Mimari Revizyon Varyantları — 20K ön-test
===================================================

Varyantlar (REVISION.md):
  r1_64   : D=64  (kapasite)
  r1_128  : D=128 (kapasite)
  r3      : attention-weighted pooling (max-pool kırıklığı)
  r2      : cloud attention (ilişkisel sözlük genişletme) + D=32

Hepsi: 20K cümle, 10 epoch, batch 64, aynı held-out (5.5M wiki, 2K).
Kontroller: D=32 standart (mevcut) + D=16.
"""

import sys
import os
import math
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from train_wiki import BPETokenizer, CloudLM


# ─────────────────────────────────────────────
# VARYANT ENKODERLER
# ─────────────────────────────────────────────

class AttnPoolCloud(nn.Module):
    """R3: max-pool yerine öğrenilmiş attention-pool (nokta branch)."""

    def __init__(self, d_cloud_dim: int, d_cloud: int = 512):
        super().__init__()
        self.d = d_cloud_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(d_cloud_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, d_cloud),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_cloud_dim + 4, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, d_cloud),
        )
        self.pool_attn = nn.Linear(d_cloud, 1)  # attention skorları
        self.fusion = nn.Sequential(
            nn.Linear(d_cloud * 2, d_cloud * 2), nn.GELU(),
            nn.Linear(d_cloud * 2, d_cloud),
            nn.LayerNorm(d_cloud),
        )

    def forward(self, points, mask=None):
        B, N, D = points.shape
        pf = self.point_mlp(points)
        if mask is not None:
            pf = pf.masked_fill(~mask.unsqueeze(-1), 0)
        # Attention-pool: skorlar, maske dışı -inf
        scores = self.pool_attn(pf).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1)  # (B, N)
        point_pooled = (pf * weights.unsqueeze(-1)).sum(dim=1)

        diff = points.unsqueeze(2) - points.unsqueeze(1)
        dist = diff.norm(dim=-1)
        norm_i = points.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_ang = (points.unsqueeze(2) * points.unsqueeze(1)).sum(-1) / \
                  (norm_i * norm_i.transpose(1, 2)).clamp(min=1e-8)
        ni = norm_i.unsqueeze(2).expand(-1, -1, N, -1)
        nj = norm_i.unsqueeze(1).expand(-1, N, -1, -1)
        pin = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1), ni, nj], -1)
        pair_feats = self.pair_mlp(pin)
        if mask is not None:
            pm = mask.unsqueeze(2) & mask.unsqueeze(1)
            pair_feats = pair_feats.masked_fill(~pm.unsqueeze(-1), 0)
            cnt = pm.float().sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
        else:
            cnt = torch.ones(B, 1, 1, device=points.device) * N * N
        pair_pooled = pair_feats.sum(dim=(1, 2)) / cnt

        return self.fusion(torch.cat([point_pooled, pair_pooled], dim=-1))


class CloudAttnEncoder(nn.Module):
    """R2: Noktalar önce self-attention'dan geçer (256D projeksiyon),
    sonra standart point/pair pipeline. D boyutunda fark yok — attention
    ilişkisel sözlüğü zenginleştirir."""

    def __init__(self, d_cloud_dim: int, d_cloud: int = 512):
        super().__init__()
        self.d = d_cloud_dim
        self.to256 = nn.Linear(d_cloud_dim, 256)
        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=2)
        self.point_mlp = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, d_cloud),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_cloud_dim + 4, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, d_cloud),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_cloud * 2, d_cloud * 2), nn.GELU(),
            nn.Linear(d_cloud * 2, d_cloud),
            nn.LayerNorm(d_cloud),
        )

    def forward(self, points, mask=None):
        B, N, D = points.shape
        h = self.to256(points)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)
        h = self.attn(h, src_key_padding_mask=~mask if mask is not None else None)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0)
        pf = self.point_mlp(h)
        point_pooled = pf.max(dim=1).values

        diff = points.unsqueeze(2) - points.unsqueeze(1)
        dist = diff.norm(dim=-1)
        norm_i = points.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_ang = (points.unsqueeze(2) * points.unsqueeze(1)).sum(-1) / \
                  (norm_i * norm_i.transpose(1, 2)).clamp(min=1e-8)
        ni = norm_i.unsqueeze(2).expand(-1, -1, N, -1)
        nj = norm_i.unsqueeze(1).expand(-1, N, -1, -1)
        pin = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1), ni, nj], -1)
        pair_feats = self.pair_mlp(pin)
        if mask is not None:
            pm = mask.unsqueeze(2) & mask.unsqueeze(1)
            pair_feats = pair_feats.masked_fill(~pm.unsqueeze(-1), 0)
            cnt = pm.float().sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
        else:
            cnt = torch.ones(B, 1, 1, device=points.device) * N * N
        pair_pooled = pair_feats.sum(dim=(1, 2)) / cnt

        return self.fusion(torch.cat([point_pooled, pair_pooled], dim=-1))


class VariantCloudLM(CloudLM):
    """CloudLM + değişir cloud encoder."""

    def __init__(self, vocab_size, d_model=256, d_cloud=512, d_cloud_dim=16,
                 n_layers=4, encoder_cls=None):
        super().__init__(vocab_size, d_model=d_model, d_cloud=d_cloud,
                         d_cloud_dim=d_cloud_dim, n_layers=n_layers)
        if encoder_cls is not None:
            self.cloud = encoder_cls(d_cloud_dim, d_cloud)


# ─────────────────────────────────────────────
# EĞİTİM + EVAL
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True,
                    choices=["d16", "d32", "r1_64", "r1_128", "r3", "r2"])
    ap.add_argument("--max-texts", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    d_dim = {"d16": 16, "d32": 32, "r1_64": 64, "r1_128": 128, "r3": 32, "r2": 32}[args.variant]
    enc_cls = {"r3": AttnPoolCloud, "r2": CloudAttnEncoder}.get(args.variant)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json"))

    with open(os.path.join(ROOT, "data", "turkish_sentences.txt"), 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    texts = [t for t in lines if 20 <= len(t) <= 300][:args.max_texts]

    # Ortak held-out: 5.5M wiki dosyası (eğitimle kesişimsiz — doğrulandı)
    with open(os.path.join(ROOT, "data", "turkish_wiki_sentences.txt"), 'r', encoding='utf-8') as f:
        wlines = [l.strip() for l in f]
    eval_texts = [t for t in wlines if 20 <= len(t) <= 300][:2000]

    eos_id = 1
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= args.seq_len + 1]

    model = VariantCloudLM(tok.num_tokens, d_model=256, d_cloud=512,
                           d_cloud_dim=d_dim, n_layers=4, encoder_cls=enc_cls)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== VARYANT: {args.variant} | D={d_dim} | {n_params:,} params ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def eval_ppl():
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for t in eval_texts:
                ids = tok.encode(t)
                if len(ids) < args.seq_len + 1:
                    continue
                for i in range(0, len(ids) - args.seq_len, args.seq_len):
                    if i + args.seq_len >= len(ids):
                        continue
                    inp = torch.tensor([ids[i:i + args.seq_len]], device=device)
                    tgt = torch.tensor([ids[i + args.seq_len]], device=device)
                    mask = (inp != 0) & (inp != eos_id)
                    logits, _, _ = model(inp, mask)
                    total_loss += F.cross_entropy(logits, tgt, reduction='sum').item()
                    total_tokens += 1
        return math.exp(total_loss / max(total_tokens, 1))

    os.makedirs(os.path.join(ROOT, "quann_cloud", "experiments", "revision"), exist_ok=True)
    best_ppl = float('inf')
    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - args.seq_len):
                examples.append((tokens[i:i + args.seq_len], tokens[i + args.seq_len]))
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
        held = eval_ppl()
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | train ppl={math.exp(min(avg,20)):.2f} | "
              f"held-out={held:.2f} | {time.time()-t0:.0f}s", flush=True)
        if held < best_ppl:
            best_ppl = held
            torch.save({
                "epoch": epoch + 1, "model_state": model.state_dict(),
                "vocab_size": tok.num_tokens,
                "config": {"d_model": 256, "d_cloud": 512, "d_cloud_dim": d_dim,
                           "n_layers": 4, "variant": args.variant},
                "ppl": held,
            }, os.path.join(ROOT, "quann_cloud", "experiments", "revision",
                            f"{args.variant}_best.pt"))

    print(f"\n  ✅ {args.variant}: best held-out PPL = {best_ppl:.2f} "
          f"({time.time()-t0:.0f}s)")

    # Üretim
    model.eval()
    print("  ÜRETİM:")
    for p in ["Türkiye'nin başkenti", "ekonomik büyüme"]:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = model.generate(ids, max_new=12, temperature=0.8, eos_id=eos_id)
        print(f"    {p[:26]:28s} → {tok.decode(out[0].tolist())[:70]}")


if __name__ == "__main__":
    main()
