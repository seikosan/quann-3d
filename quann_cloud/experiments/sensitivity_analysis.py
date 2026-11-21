"""
Bulut Hassasiyet Analizi — "tek nokta perturbasyonu ne kadar etki ediyor?"
===========================================================================

CloudLM 200K modeli üzerinde:
1. Her pozisyonun noktası rastgele yönle ε ölçeğinde kaydırılıyor
2. Ölçülenler: Δcloud_state (mutlak), ΔCE (hata değişimi), top-1 değişimi
3. Branch ablatasyonu: max-pool vs pair-pool katkısı

Amaç: Düz hata manzarasının (Δ≈0) kaynağını lokalleştirmek —
mimari revizyonun hedefi buradan çıkacak.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from train_wiki import BPETokenizer, CloudLM


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    tok = BPETokenizer.load(os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json"))
    ck = torch.load(os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_cloud_best.pt"),
                    map_location="cpu", weights_only=False)
    model = CloudLM(ck["vocab_size"], **ck["config"])
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()
    D = model.d_cloud_dim
    print(f"Model: D={D}, d_cloud={ck['config']['d_cloud']} | device={device}")

    with open(os.path.join(ROOT, "data", "turkish_sentences.txt"), 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    texts = [t for t in lines if 20 <= len(t) <= 300][:500]

    eos_id = 1
    SEQ = 10

    # ── 1) Tek nokta perturbasyonu → ΔCE, Δcloud_state ──
    d_ce, d_cs, top1_flip = [], [], 0
    n_tested = 0
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t)
            if len(ids) < SEQ + 1:
                continue
            tokens = torch.tensor([ids[:SEQ]], device=device)
            mask = (tokens != 0) & (tokens != eos_id)
            target = ids[SEQ]

            logits0, _, points0 = model(tokens, mask)
            ce0 = F.cross_entropy(logits0, torch.tensor([target], device=device)).item()
            top0 = logits0.argmax(1).item()

            # cloud_state'ı elle hesapla (points0 ile)
            cs0 = model.cloud(points0, mask)

            for i in range(SEQ):
                if not mask[0, i].item():
                    continue
                p = points0[0, i].clone()
                if p.norm() < 1e-6:
                    continue
                # Birim büyütme + rastgele yön (ε=0.5 — belirgin ama makul)
                direction = torch.randn_like(p)
                direction /= direction.norm()
                perturbed = points0.clone()
                perturbed[0, i] = p + 0.5 * direction

                new_cs = model.cloud(perturbed, mask)
                d_cs.append((new_cs - cs0).norm().item() / cs0.norm().item())

                logits1 = model.word_head(new_cs)
                ce1 = F.cross_entropy(logits1, torch.tensor([target], device=device)).item()
                d_ce.append(ce1 - ce0)
                if logits1.argmax(1).item() != top0:
                    top1_flip += 1
                n_tested += 1

    d_ce = np.array(d_ce)
    d_cs = np.array(d_cs)
    print(f"\n=== TEK NOKTA PERTURBASYONU (ε=0.5, n={n_tested}) ===")
    print(f"  ΔCE:  ortalama {d_ce.mean():+.5f} | |ΔCE| ort {np.abs(d_ce).mean():.5f} | "
          f"%95 {np.percentile(np.abs(d_ce), 95):.5f}")
    print(f"  ΔCS:  ortalama {d_cs.mean():.5f} (cloud_state normuna göre)")
    print(f"  Top-1 flip oranı: {top1_flip/n_tested:.1%}")

    # ── 2) Branch ablatasyonu: max-pool vs pair katkısı ──
    # CloudEncoderND: fusion([point_pooled, pair_pooled])
    # Tek nokta perturbasyonunun her branch'e etkisini ayrı ölç
    d_point_branch, d_pair_branch = [], []
    with torch.no_grad():
        for t in texts[:200]:
            ids = tok.encode(t)
            if len(ids) < SEQ + 1:
                continue
            tokens = torch.tensor([ids[:SEQ]], device=device)
            mask = (tokens != 0) & (tokens != eos_id)
            _, _, points0 = model(tokens, mask)
            if not mask[0].any():
                continue

            pf0 = model.cloud.point_mlp(points0)
            pf0 = pf0.masked_fill(~mask.unsqueeze(-1), torch.finfo(pf0.dtype).min)
            pp0 = pf0.max(dim=1).values
            pp0 = pp0.masked_fill(~mask.any(dim=1, keepdim=True), 0)

            def pair_pool(pts):
                B, N, Dd = pts.shape
                diff = pts.unsqueeze(2) - pts.unsqueeze(1)
                dist = diff.norm(dim=-1)
                norm_i = pts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos_ang = (pts.unsqueeze(2) * pts.unsqueeze(1)).sum(-1) / \
                          (norm_i * norm_i.transpose(1, 2)).clamp(min=1e-8)
                ni = norm_i.unsqueeze(2).expand(-1, -1, N, -1)
                nj = norm_i.unsqueeze(1).expand(-1, N, -1, -1)
                pin = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1), ni, nj], -1)
                pf = model.cloud.pair_mlp(pin)
                pm = mask.unsqueeze(2) & mask.unsqueeze(1)
                pf = pf.masked_fill(~pm.unsqueeze(-1), 0)
                cnt = pm.float().sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
                return pf.sum(dim=(1, 2)) / cnt

            rp0 = pair_pool(points0)

            i = int(mask[0].nonzero()[len(mask[0]) // 2])
            p = points0[0, i].clone()
            if p.norm() < 1e-6:
                continue
            direction = torch.randn_like(p)
            direction /= direction.norm()
            perturbed = points0.clone()
            perturbed[0, i] = p + 0.5 * direction

            pf1 = model.cloud.point_mlp(perturbed).masked_fill(~mask.unsqueeze(-1), 0)
            pp1 = pf1.max(dim=1).values
            rp1 = pair_pool(perturbed)

            d_point_branch.append((pp1 - pp0).norm().item() / pp0.norm().item())
            d_pair_branch.append((rp1 - rp0).norm().item() / rp0.norm().item())

    print(f"\n=== BRANCH ABLATASYONU (tek nokta, ε=0.5, n={len(d_point_branch)}) ===")
    print(f"  max-pool branch  Δ: {np.mean(d_point_branch):.5f}  "
          f"(→ {'BASKIN' if np.mean(d_point_branch) > np.mean(d_pair_branch) else 'sessiz'})")
    print(f"  pair-pool branch Δ: {np.mean(d_pair_branch):.5f}  "
          f"(→ {'BASKIN' if np.mean(d_pair_branch) > np.mean(d_point_branch) else 'sessiz'})")
    print(f"  Oran (pair/max): {np.mean(d_pair_branch) / max(np.mean(d_point_branch), 1e-9):.2f}×")

    # ── 3) D boyutunun kapasite sınırı ──
    # Noktaların tutabileceği bilgi: 10 nokta × D boyut × tanh[-1,1]
    print("\n=== KAPASİTE NOTU ===")
    print(f"  Bulut: 10 nokta × {D}D = {10 * D} sayı (tanh sınırlı)")
    print("  Transformer: 10 × 256D = 2560 sayı (sınırsız ölçek)")
    print(f"  Bilgi oranı: bulut/transformer ≈ {10 * D / 2560:.1%}")


if __name__ == "__main__":
    main()
