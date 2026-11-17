"""
QUANN-Cloud Faz 1: XOR Gate Doğrulması
========================================

Amaç: QUANN'ın backprop'suz aralık-daraltma mekanizmasının, TEMİZ ve
düşük-boyutlu bir problemde (XOR) gerçekten çalıştığını kanıtlamak.

Bu, gerçek LM deneyindeki (FINDINGS.md) "0th-order ölçeklenmiyor" bulgusuyla
çelişmez: XOR düşük-boyutlu (~33 ağırlık) ve QUANN'ın unimodal varsayımının
çalışabildiği bir rejimdir. LM ise 111K+ ağırlık, çok modallı.

Üç model, aynı XORNet (2 → 8 → 1, tanh):
  - Random  : eğitilmemiş
  - Backprop: AdamW (1st-order)  — referans
  - QUANN   : aralık-daraltma (0th-order, backprop'suz)

Metrik: 4 XOR örneğinde doğruluk (%) + MSE.

Çalıştır:
    python3 quann_cloud/experiments/xor_gate/xor.py
"""

import os
import sys
import math
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "quann_cloud", "src"))
from qcloud_quann import QUANN, QUANNConfig


class XORNet(nn.Module):
    """2 giriş → 8 gizli (tanh) → 1 çıkış. ~33 ağırlık."""
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x))))


def xor_data():
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, Y


def xor_loss(model, inputs, targets):
    """MSE (grad yok)."""
    with torch.no_grad():
        pred = model(inputs)
        return F.mse_loss(pred, targets).item()


def accuracy(model, X, Y, thr=0.5):
    pred = (model(X) > thr).float()
    return (pred == Y).float().mean().item() * 100


def train_backprop(model, X, Y, epochs=300, lr=0.5, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
    return model


def main():
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = torch.device("cpu")
    X, Y = xor_data()

    print("=" * 64)
    print("  QUANN-Cloud Faz 1: XOR Gate (backprop'suz mekanizma doğrulaması)")
    print("=" * 64)

    def make():
        return XORNet(8).to(device)

    # ---- Random ----
    m0 = make()
    acc0 = accuracy(m0, X, Y)
    print(f"\n  [Random]    doğruluk = {acc0:.0f}%  (4 örnek: {['00','01','10','11']})")
    del m0

    # ---- Backprop ----
    m_bp = make()
    t0 = time.time()
    train_backprop(m_bp, X, Y, epochs=300)
    acc_bp = accuracy(m_bp, X, Y)
    mse_bp = xor_loss(m_bp, X, Y)
    print(f"  [Backprop]  doğruluk = {acc_bp:.0f}%  MSE={mse_bp:.4f}  ({time.time()-t0:.1f}s)")

    # ---- QUANN (backprop'suz) ----
    m_q = make()
    n_params = sum(p.numel() for p in m_q.parameters())
    cfg = QUANNConfig(
        k_weights=n_params,   # tüm ağırlıkları her adımda daralt (küçük ağ)
        minibatch=4,          # 4 XOR örneği = gürültüsüz sinyal
        sigma0=0.3,
        eps=1e-5,
        min_width=1e-3,
        seed=0,
    )
    quann = QUANN(m_q, device, cfg, loss_fn=xor_loss)
    qinfo = quann.info()
    print(f"  [QUANN]     {qinfo['total_params']} ağırlık, {qinfo['n_param_tensors']} tensor, σ0={cfg.sigma0}")

    examples = [(X[i], Y[i]) for i in range(4)]
    t0 = time.time()
    quann.fit(examples, epochs=400, eval_fn=None, verbose=False)
    acc_q = accuracy(m_q, X, Y)
    mse_q = xor_loss(m_q, X, Y)
    qinfo2 = quann.info()
    print(f"  [QUANN]     doğruluk = {acc_q:.0f}%  MSE={mse_q:.4f}  ({time.time()-t0:.1f}s)")
    print(f"              aralık: mean_width={qinfo2['mean_width']:.4f} "
          f"pct_converged={qinfo2['pct_converged']*100:.0f}%")

    # ---- Karşılaştırma tablosu ----
    print(f"\n{'='*64}\n  SONUÇ (XOR, 4 örnek)\n{'='*64}")
    print(f"  {'Model':12s} {'Doğruluk':>10s} {'MSE':>10s}")
    for name, acc, mse in [("Random", acc0, float('nan')),
                           ("Backprop", acc_bp, mse_bp),
                           ("QUANN", acc_q, mse_q)]:
        mse_s = f"{mse:.4f}" if mse == mse else "—"
        print(f"  {name:12s} {acc:9.0f}% {mse_s:>10s}")

    ok = acc_q >= 99.9
    verdict = ("✅ QUANN XOR'u backprop'suz çözdü" if ok
               else "⚠️ QUANN XOR'u tam çözemedi")
    print(f"\n  {verdict}")

    # Çıktı
    out = {
        "task": "xor", "hidden": 8, "params": n_params,
        "random_acc": acc0, "backprop_acc": acc_bp, "backprop_mse": mse_bp,
        "quann_acc": acc_q, "quann_mse": mse_q,
        "quann_mean_width": qinfo2["mean_width"],
        "quann_pct_converged": qinfo2["pct_converged"],
        "quann_solved": bool(ok),
    }
    out_path = os.path.join(ROOT, "quann_cloud", "logs", "xor_gate.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
