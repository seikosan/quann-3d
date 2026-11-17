"""
Cümle Bulutu Dil Modeli — temel demo (3D bulut, küçük corpus)
==============================================================
Transformer'ın "sıralı dikkat + tek vektör" yaklaşımı yerine:
  - Her kelime → 3D uzayda bir nokta
  - Cümle → nokta bulutu
  - Anlam → bulut geometrisi (mesafe, açı, küme formu)
  - Tahmin → bulutun mevcut konfigürasyonundan bir sonraki kelime

Mimari:
  Token → Context Encoder → 3D Projection → PointNet Cloud Encoder → Next Word

Bu script bir DEMO'dur — model mimarisi ve tokenizer tek kaynaktan gelir:
cloudlm paketi (kanonik CloudLM, d_cloud_dim=3 → 3D bulut).

Kullanım:
  python3 src/cloud_lm.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from cloudlm import CloudLM, Tok


# ─────────────────────────────────────────────
# EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────

def train_cloud_lm(
    model: CloudLM,
    train_texts: List[str],
    tokenizer,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    seq_len: int = 32,
    weight_decay: float = 0.01,
    device: str = "auto",
    verbose: bool = True,
):
    """
    Basit eğitim döngüsü.

    Loss = CE(word) + λ * MSE(point)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    all_tokens = [tokenizer.encode(t) for t in train_texts]
    vocab_size = model.word_head[-1].out_features

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_tokens = sum(len(t) for t in all_tokens)
    if verbose:
        print(f"Corpus: {len(train_texts)} cümle, {total_tokens} token")
        print(f"Vocab: {vocab_size}, Device: {device}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params:,}")
        print()

    model.train()
    for epoch in range(epochs):
        examples = []
        for tokens in all_tokens:
            for i in range(len(tokens) - seq_len):
                inp = tokens[i:i+seq_len]
                tgt = tokens[i+seq_len]
                examples.append((inp, tgt))

        np.random.shuffle(examples)

        epoch_loss = 0
        n_batches = 0
        for b in range(0, len(examples), batch_size):
            batch_ex = examples[b:b+batch_size]
            inputs = torch.tensor([e[0] for e in batch_ex], device=device)
            targets = torch.tensor([e[1] for e in batch_ex], device=device)
            mask = inputs != 0  # 0 = pad

            optimizer.zero_grad()
            word_logits, next_point, _ = model(inputs, mask)

            loss_ce = F.cross_entropy(word_logits, targets)

            # MSE loss — 3D nokta (auxiliary: hedef token'ın kendi projeksiyonu)
            with torch.no_grad():
                tgt_mask = torch.ones(targets.size(0), 1, dtype=torch.bool, device=device)
                _, _, tgt_points = model(targets.unsqueeze(1), tgt_mask)
                tgt_point = tgt_points.squeeze(1)

            loss_point = F.mse_loss(next_point, tgt_point)

            loss = loss_ce + 0.1 * loss_point
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            ppl = math.exp(min(avg_loss, 20))
            print(f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | ppl={ppl:.1f}")

        scheduler.step()

    model.eval()
    return model


# ─────────────────────────────────────────────
# DEMO: Küçük bir corpus üzerinde
# ─────────────────────────────────────────────

def demo():
    """Küçük bir Türkçe corpus üzerinde demo."""

    texts = [
        "kedi pencereden dışarı baktı",
        "kedi masanın üstünde uyudu",
        "köpek bahçede top kovaladı",
        "köpek bahçede koştu",
        "çocuk parkta bisiklet sürdü",
        "çocuk parkta kaydı",
        "kadın pazara domates aldı",
        "kadın pazara ekmek aldı",
        "adam otobüse bindi",
        "adam otobüsten indi",
        "kedi fareyi kovaladı",
        "kedi süt içti",
        "köpek kemiğini gömdü",
        "köpek topu ısırdı",
        "çocuk annesine sarıldı",
        "çocuk babasıyla oynadı",
        "kadın çay demledi",
        "kadın kitap okudu",
        "adam gazeteyi okudu",
        "adam kahve içti",
    ] * 10  # tekrarla

    tok = Tok()
    vocab_size = tok.build(texts)
    print(f"Vocabulary: {vocab_size} tokens")

    # 3D bulut: d_cloud_dim=3
    model = CloudLM(vocab_size=vocab_size, d_model=64, d_cloud=128,
                    d_cloud_dim=3, n_layers=2)

    model = train_cloud_lm(
        model, texts, tok,
        epochs=50, batch_size=16, lr=1e-3, seq_len=3,
        verbose=True
    )

    # Üretim testi
    print("\n" + "="*50)
    print("🧪 ÜRETİM TESTİ")
    print("="*50)

    prompts = [
        "kedi pencereden",
        "köpek bahçede",
        "çocuk parkta",
        "kadın pazara",
    ]

    device = next(model.parameters()).device
    for prompt in prompts:
        tokens = torch.tensor([tok.encode(prompt)], device=device)
        result = model.generate(tokens, max_new=8, temperature=0.8)
        print(f"  {prompt!r:30s} → {tok.decode(result[0].tolist())}")

    # Bulut görselleştirme
    print("\n" + "="*50)
    print("🌩️  BULUT GEOMETRİSİ (3D)")
    print("="*50)

    test_sentence = "kedi pencereden dışarı baktı"
    tokens = torch.tensor([tok.encode(test_sentence)], device=device)
    mask = tokens != 0

    with torch.no_grad():
        _, _, points = model(tokens, mask)

    print(f"\n  Cümle: {test_sentence!r}")
    print(f"  Nokta bulutu (3D):")
    words = test_sentence.split()
    for i, (w, p) in enumerate(zip(words, points[0].tolist())):
        print(f"    {w:15s} → ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    # Mesafe matrisi
    pts = points[0].cpu().numpy()
    n = len(words)
    print(f"\n  Mesafe matrisi:")
    header = "            " + "".join(f"{w[:8]:>10s}" for w in words)
    print(f"  {header}")
    for i in range(n):
        row = f"  {words[i][:10]:12s}"
        for j in range(n):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            row += f"{d:10.3f}"
        print(row)

    print(f"\n  Yorum: Yakın noktalar = güçlü semantik bağ")
    print(f"  (kedi-baktı yakın, pencereden uzak → özne-yüklem ilişkisi)")


if __name__ == "__main__":
    demo()
