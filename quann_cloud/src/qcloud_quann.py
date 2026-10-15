"""
QUANN Engine — Backprop'suz, aralık-daraltmalı ağırlık öğrenmesi
================================================================

CloudLM (veya herhangi bir nn.Module) ağırlıklarını, gradyan YOK,
sadece forward-pass + ödül/ceza + sigma-adaptasyonu + plato-tespiti
ile öğrenir.

Bu, QUANN v0'daki mekanizmanın (QWeight / QNetwork) bir nn.Module
ağırlıklarına vektörel uyarlamasıdır:

  Her ağırlık elemanı i için bir arama aralığı [low[i], high[i]] vardır.
  - Bir adım: aralıktan sigma ölçekli bir aday (candidate) üretilir.
  - Adayla forward -> hata (loss) ölçülür.
  - Hata azaldı   -> ÖDÜL : değer kabul edilir, sigma küçülür (istismar).
  - Hata arttı    -> CEZA : değer geri alınır, yön ters çevrilir,
                            aralık aday tarafı daraltılır, sigma büyür (keşif).
  - Hata değişmedi -> yön rastgele çevrilir (keşif).
  - Plato (hata uzun süredir düşmüyor) -> sigma açılır (yeniden keşif).

"Backprop'suz" anlamı: hiçbir yerde .backward() yok, gradyan yok.
Tek maliyet: aday başına 1 forward pass.

Stokastik: her adımda ağırlıkların rastgele bir alt kümesi (K tanesi)
daraltılır — tıpkı SGD'in mini-batch seçmesi gibi. Bu, tam ağırlık
kümeyi her adımda daraltmaktan çok daha uygulanabilir kılar.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_loss(model: nn.Module, inputs: torch.Tensor,
                  targets: torch.Tensor) -> float:
    """Varsayılan LM hata fonksiyonu: ortalama cross-entropy.
    inputs: [T] veya [B, T]; targets: [] veya [B]."""
    with torch.no_grad():
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)
        mask = inputs != 0
        logits, _, _ = model(inputs, mask)
        loss = F.cross_entropy(logits, targets)
    return loss.item()


@dataclass
class QUANNConfig:
    """QUANN arama hiper-parametreleri."""
    sigma0: float = 0.25          # başlangıç adım ölçeği (aralık birimi)
    sigma_floor: float = 0.02     # sigma alt sınırı
    sigma_cap: float = 0.6        # sigma üst sınırı
    shrink: float = 0.82          # ödülde sigma çarpanı (küçült)
    grow: float = 1.0 / 0.82      # cezada sigma çarpanı (büyüt)
    eps: float = 1e-4             # ödül/ceza eşiği (hata farkı)
    k_weights: int = 60           # adımda daraltılacak ağırlık sayısı
    minibatch: int = 8            # ödül sinyali için mini-batch boyutu
    min_width: float = 1e-3       # aralık bu kadar daraltırsa "yakınsadı"
    plateau_window: int = 40      # plato penceresi (adım)
    plateau_patience: float = 1e-3  # pencerede bu kadar düşüş yoksa plato
    plateau_grow: float = 1.5     # platonunda sigma çarpanı
    seed: int = 0


class QUANN:
    """
    Bir nn.Module'un ağırlıklarını backprop'suz öğrenen QUANN motoru.

    Kullanım:
        quann = QUANN(model, device, cfg)
        quann.fit(examples, epochs=...)   # examples: list[(inputs, targets)]
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 cfg: Optional[QUANNConfig] = None,
                 loss_fn: Optional[Callable] = None):
        """
        Args:
            model: nn.Module (ağırlıkları daraltılacak ağ)
            device: hesaplama cihazı
            cfg: QUANNConfig
            loss_fn: callable(model, inputs, targets) -> float.
                     None ise varsayılan LM cross-entropy kullanılır.
        """
        self.model = model
        self.device = device
        self.cfg = cfg or QUANNConfig()
        self.rng = random.Random(self.cfg.seed)
        self.loss_fn = loss_fn or _default_loss
        self._init_state()

    # ------------------------------------------------------------------
    # QUANN durumu (her ağırlık için aralık + yön + sigma)
    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        self.low: List[torch.Tensor] = []
        self.high: List[torch.Tensor] = []
        self.direction: List[torch.Tensor] = []
        self.sigma: List[float] = []
        self._params: List[nn.Parameter] = []
        self._flat_sizes: List[int] = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # QUANN gradyan kullanmaz -> in-place güncelleme için grad kapat
            p.requires_grad_(False)
            v = p.data
            std = v.std().item() if v.numel() > 1 else 1.0
            # Aralığı ağırlığın doğal ölçeğine göre başlat:
            #   geniş (keşif için) ama ağırlık ölçeğine uygun
            radius = max(0.05, 6.0 * std)
            low = v - radius
            high = v + radius
            self._params.append(p)
            self.low.append(low.clone())
            self.high.append(high.clone())
            # Yön: her eleman için ±1 (rastgele)
            signs = torch.where(torch.rand_like(v) < 0.5,
                                torch.ones_like(v), -torch.ones_like(v))
            self.direction.append(signs)
            self._flat_sizes.append(v.numel())
            self.sigma.append(self.cfg.sigma0)

        self._total_params = sum(self._flat_sizes)
        self._cum = [0]
        for s in self._flat_sizes:
            self._cum.append(self._cum[-1] + s)

    # ------------------------------------------------------------------
    # Ağırlık seçimi
    # ------------------------------------------------------------------
    def _select(self, k: int) -> List[Tuple[int, int]]:
        """K rastgele ağırlık elemanı seç (param_idx, flat_idx)."""
        k = min(k, self._total_params)
        chosen = self.rng.sample(range(self._total_params), k)
        out = []
        for g in chosen:
            pi = self._bisect_param(g)
            base = self._cum[pi]
            out.append((pi, g - base))
        return out

    def _bisect_param(self, g: int) -> int:
        lo, hi = 0, len(self._params) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum[mid + 1] <= g:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------
    # Hata fonksiyonu (model-agnostik)
    # ------------------------------------------------------------------
    def _loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """loss_fn(model, inputs, targets) -> float (grad yok)."""
        self.model.eval()
        return self.loss_fn(self.model, inputs, targets)

    # ------------------------------------------------------------------
    # TEK ADIM: bir mini-batch üzerinde K ağırlık daralt
    # ------------------------------------------------------------------
    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        cfg = self.cfg
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # base_err: mini-batch üzerinde ölçülür (gürültü azaltma)
        base_err = self._loss(inputs, targets)

        stat = {'rewarded': 0, 'censured': 0, 'neutral': 0, 'converged': 0}
        sel = self._select(cfg.k_weights)

        for pi, fi in sel:
            p = self._params[pi].view(-1)
            low = self.low[pi].view(-1)
            high = self.high[pi].view(-1)
            d = self.direction[pi].view(-1)

            lo = low[fi].item()
            hi = high[fi].item()
            width = hi - lo
            if width < cfg.min_width:
                stat['converged'] += 1
                continue

            cur = p[fi].item()
            # sigma ölçekli, yön bazlı adım (aralık içinde)
            step = d[fi].item() * self.sigma[pi] * width
            cand = cur + step
            cand = min(hi, max(lo, cand))
            if abs(cand - cur) < 1e-9:
                stat['neutral'] += 1
                continue

            # --- adayla hata (aynı mini-batch, 1 forward) ---
            old = cur
            p[fi] = cand
            err = self._loss(inputs, targets)

            if err < base_err - cfg.eps:
                # ÖDÜL: kabul et, base_err'ı güncelle (stale ödül önlenir), sigma küçült
                p[fi] = cand
                base_err = err
                stat['rewarded'] += 1
                self.sigma[pi] = max(cfg.sigma_floor, self.sigma[pi] * cfg.shrink)
            elif err > base_err + cfg.eps:
                # CEZA: geri al, yön çevir, aralığı aday tarafı daralt, sigma büyüt
                p[fi] = old
                d[fi] = -d[fi]
                mid = (old + cand) / 2
                if cand > old:
                    high[fi] = mid
                else:
                    low[fi] = mid
                stat['censured'] += 1
                self.sigma[pi] = min(cfg.sigma_cap, self.sigma[pi] * cfg.grow)
            else:
                # NÖTR: geri al, yönü rastgele çevir (keşif)
                p[fi] = old
                if self.rng.random() < 0.5:
                    d[fi] = -d[fi]
                stat['neutral'] += 1

        return {'base_err': base_err, **stat}

    # ------------------------------------------------------------------
    # EĞİTİM
    # ------------------------------------------------------------------
    def fit(self, examples: List[Tuple[torch.Tensor, torch.Tensor]],
            epochs: int = 1, eval_fn: Optional[Callable] = None,
            verbose: bool = True) -> dict:
        cfg = self.cfg
        history = []
        best = float('inf')
        plateau = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            order = list(range(len(examples)))
            self.rng.shuffle(order)

            ep_rewarded = ep_censured = ep_neutral = ep_converged = 0
            ep_loss_sum = 0.0
            ep_n_ex = 0
            recent: List[float] = []
            B = max(1, cfg.minibatch)

            for b in range(0, len(order), B):
                idxs = order[b:b + B]
                inps = torch.stack([examples[i][0] for i in idxs])
                tgts = torch.stack([examples[i][1] for i in idxs])
                r = self.step(inps, tgts)
                ep_rewarded += r['rewarded']
                ep_censured += r['censured']
                ep_neutral += r['neutral']
                ep_converged += r['converged']
                ep_loss_sum += r['base_err'] * len(idxs)
                ep_n_ex += len(idxs)
                recent.append(r['base_err'])
                if len(recent) > cfg.plateau_window:
                    recent.pop(0)

                # Plato tespiti: pencerede anlamlı düşüş yoksa sigma aç
                if len(recent) == cfg.plateau_window:
                    if (max(recent) - min(recent)) < cfg.plateau_patience:
                        plateau += 1
                        if plateau % 5 == 0:
                            for i in range(len(self.sigma)):
                                self.sigma[i] = min(cfg.sigma_cap,
                                                    self.sigma[i] * cfg.plateau_grow)
                            plateau = 0
                    else:
                        plateau = 0

            avg_loss = ep_loss_sum / max(ep_n_ex, 1)
            elapsed = time.time() - t0
            rec = {
                'epoch': epoch, 'avg_loss': avg_loss,
                'ppl': math.exp(min(avg_loss, 20)),
                'rewarded': ep_rewarded, 'censured': ep_censured,
                'neutral': ep_neutral, 'converged': ep_converged,
                'sigma_mean': sum(self.sigma) / len(self.sigma),
                'elapsed': elapsed,
            }
            history.append(rec)
            if verbose:
                print(f"  E{epoch:3d} | loss={avg_loss:.4f} ppl={rec['ppl']:.2f} "
                      f"| R={ep_rewarded} C={ep_censured} N={ep_neutral} "
                      f"conv={ep_converged} | σ={rec['sigma_mean']:.3f} | {elapsed:.0f}s",
                      flush=True)

            if eval_fn is not None:
                v = eval_fn()
                rec['eval'] = v
                if verbose:
                    print(f"        eval: {v}")
                if v < best:
                    best = v

        return {'history': history, 'best_eval': best}

    # ------------------------------------------------------------------
    # İstatistik
    # ------------------------------------------------------------------
    def info(self) -> dict:
        widths = []
        for lo, hi in zip(self.low, self.high):
            w = (hi - lo).flatten()
            widths.extend(w.tolist())
        widths = torch.tensor(widths)
        return {
            'total_params': self._total_params,
            'n_param_tensors': len(self._params),
            'mean_width': widths.mean().item(),
            'pct_converged': (widths < self.cfg.min_width).float().mean().item(),
            'sigma_mean': sum(self.sigma) / len(self.sigma),
            'sigma_min': min(self.sigma),
            'sigma_max': max(self.sigma),
        }
