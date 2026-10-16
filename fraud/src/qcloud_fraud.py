"""QCloudFraudClassifier — the research's point-cloud architecture, for fraud.

Reuses the canonical QUANN-Cloud primitives (NOT a plain MLP):
    * WordToND        (cloudlm)     : context/emb vector -> D-dim cloud point
    * CloudEncoderND  (cloudlm)     : point CLOUD (set) -> cloud state
                                        (PointNet: point-MLP + max-pool, and
                                         pair-geometry [diff,dist,cos-angle] +
                                         mean-pool; permutation-invariant)
    * QCloudWeight    (quann_cloud) : one D-dim point per token identity, with
                                        [low, high] intervals + sigma — the
                                        0th-order (no-backprop) search state

Adaptation for a tabular classification task:
    * input  : a transaction = K points (one per quantized feature), a SET
               (no positional encoding / no sequence transformer — it is a
               cloud, not a sentence)
    * head   : class_head  cloud_state -> 1 logit  (fraud)   [was: word_head->vocab]
    * loss   : binary cross-entropy                     [was: next-token CE]
    * QUANN  : update_token_weights narrows each (feature,bucket) identity
               point by reward/censure  (0th-order, no .backward())

Three training regimes (all share this model + the same quantized input):
    pure_quann : network params frozen (random init); QUANN moves the points.
    hybrid     : network (cloud_encoder+class_head) backprop + QUANN points.
    backprop   : points from a learned emb->to_nd; everything backprop.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config  # noqa: F401  (sets up repo-root import path)

from cloudlm import WordToND, CloudEncoderND
from quann_cloud.src.qcloud_weight import QCloudWeight


class QCloudFraudClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_cloud: int = 128,
        d_cloud_dim: int = 8,
        hidden: int = 64,
        convergence_threshold: float = 0.05,
        expansion_rate: float = 0.1,
    ):
        super().__init__()
        self.D = d_cloud_dim
        self.d_model = d_model
        self.d_cloud = d_cloud
        self.vocab_size = vocab_size
        self.eos_id = 1
        self.convergence_threshold = convergence_threshold
        self.expansion_rate = expansion_rate

        # representation path (used by the backprop baseline)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.to_nd = WordToND(d_model, d_cloud_dim)
        self.cloud_encoder = CloudEncoderND(d_cloud_dim, d_cloud, hidden=hidden)

        # classification head (fraud) — replaces the LM word_head
        self.class_head = nn.Sequential(
            nn.Linear(d_cloud, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # QUANN identity points: token_id -> QCloudWeight (lazy)
        self.token_weights: Dict[int, QCloudWeight] = {}

    # ------------------------------------------------------------------ points
    def get_weight(self, token_id: int) -> QCloudWeight:
        w = self.token_weights.get(token_id)
        if w is None:
            w = QCloudWeight(
                D=self.D,
                convergence_threshold=self.convergence_threshold,
                expansion_rate=self.expansion_rate,
            )
            self.token_weights[token_id] = w
        return w

    def points_for(self, tokens: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, K, D) — QUANN identity points; pad/eos positions are 0.

        Vectorized over unique token ids (one lookup each) instead of a
        B*K Python loop, so it stays fast for large eval batches.
        """
        B, K = tokens.shape
        ids = tokens.reshape(-1)
        uniq, inv = torch.unique(ids, return_inverse=True)
        vals = torch.stack([
            torch.tensor(self.get_weight(int(t)).value,
                         device=tokens.device, dtype=torch.float32)
            for t in uniq.tolist()
        ]) if len(uniq) else torch.zeros(0, self.D,
                                         device=tokens.device, dtype=torch.float32)
        pts = vals[inv].view(B, K, self.D)
        if mask is not None:
            pts = pts.masked_fill(~mask.unsqueeze(-1), 0.0)
        return pts

    # ------------------------------------------------------------------ forward
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                points_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if points_override is not None:
            points = points_override
        else:
            x = self.embedding(tokens) * math.sqrt(self.d_model)
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
            points = self.to_nd(x)
        cloud_state = self.cloud_encoder(points, mask)
        class_logits = self.class_head(cloud_state)  # (B, 1)
        return class_logits, points

    def predict_proba(self, tokens: torch.Tensor, use_quann_points: bool = True,
                      batch: int = 50000) -> torch.Tensor:
        """(B,) fraud probability, chunked for large eval sets.

        use_quann_points=True -> QUANN identity points; False -> learned emb->to_nd.
        """
        self.eval()
        B = tokens.shape[0]
        outs = []
        with torch.no_grad():
            for s in range(0, B, batch):
                t = tokens[s:s + batch]
                msk = (t != 0) & (t != self.eos_id)
                if use_quann_points:
                    pts = self.points_for(t, msk)
                    logits, _ = self.forward(t, msk, points_override=pts)
                else:
                    logits, _ = self.forward(t, msk)
                outs.append(torch.sigmoid(logits).squeeze(-1))
        return torch.cat(outs)

    # ------------------------------------------------------------------ error
    def compute_error(self, tokens: torch.Tensor, mask: torch.Tensor,
                      target: int, points_override: Optional[torch.Tensor] = None) -> float:
        with torch.no_grad():
            logits, _ = self.forward(tokens, mask, points_override=points_override)
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), torch.full((1,), float(target), device=tokens.device))
        return loss.item()

    # ------------------------------------------------- QUANN 0th-order update
    def update_token_weights(self, tokens: torch.Tensor, mask: torch.Tensor,
                             target: int, base_points: Optional[torch.Tensor] = None,
                             search_rule: str = "classic",
                             pos_weight: Optional[torch.Tensor] = None) -> dict:
        """Narrow each (feature,bucket) point by reward/censure. No backprop.

        Batched: base + one candidate per valid position in a single forward
        (pass 1); failures get their opposite in a second forward (pass 2).
        `pos_weight` (scalar tensor) re-weights the positive class in the reward
        BCE so the search stays balanced on an imbalanced pool.
        """
        stats = {"rewarded": 0, "censured": 0, "unchanged": 0}
        with torch.no_grad():
            K = tokens.size(1)
            if base_points is None:
                base_points = self.points_for(tokens, mask)

            valid = [(i, int(tokens[0, i])) for i in range(K)
                     if bool(mask[0, i]) and int(tokens[0, i]) not in (0, self.eos_id)]

            variants = [base_points]
            candidates: Dict[int, dict] = {}
            for i, t in valid:
                w = self.get_weight(t)
                c = torch.tensor(w.sample(), device=tokens.device, dtype=torch.float32)
                vp = base_points.clone()
                for j in range(K):
                    if int(tokens[0, j]) == t:
                        vp[0, j] = c
                candidates[i] = {"cand": c, "opp": None}
                variants.append(vp)

            tgt = torch.full((len(variants),), float(target),
                             device=tokens.device)
            var_tok = tokens.expand(len(variants), -1)
            var_mask = mask.expand(len(variants), -1)
            var_pts = torch.stack(variants).squeeze(1)  # (len, K, D)
            var_logits, _ = self.forward(var_tok, var_mask, points_override=var_pts)
            var_err = F.binary_cross_entropy_with_logits(
                var_logits.squeeze(-1), tgt, reduction="none", pos_weight=pos_weight)
            base_error = var_err[0].item()
            final_error = base_error
            delta_sum = (var_err[1:] - base_error).sum().item()
            delta_n = len(valid)

            def _scatter(new_val: torch.Tensor, src_pts: torch.Tensor, t: int):
                for j in range(K):
                    if int(tokens[0, j]) == t:
                        src_pts[0, j] = new_val

            failed = []
            for k, (i, t) in enumerate(valid):
                err_c = var_err[k + 1].item()
                w = self.get_weight(t)
                if err_c < base_error - 1e-6:
                    self._apply_narrow(w, candidates[i]["cand"].numpy(), True, search_rule)
                    w.update_sigma(rewarded=True)
                    stats["rewarded"] += 1
                    _scatter(torch.tensor(w.value, dtype=torch.float32, device=tokens.device),
                             base_points, t)
                else:
                    failed.append((i, t, err_c))

            if failed:
                opp_variants = []
                for i, t, _ in failed:
                    w = self.get_weight(t)
                    opp = torch.tensor(w.opposite(w.value), device=tokens.device,
                                       dtype=torch.float32)
                    vp = base_points.clone()
                    _scatter(opp, vp, t)
                    candidates[i]["opp"] = opp
                    opp_variants.append(vp)
                o_tok = tokens.expand(len(opp_variants), -1)
                o_mask = mask.expand(len(opp_variants), -1)
                o_logits, _ = self.forward(o_tok, o_mask,
                                           points_override=torch.stack(opp_variants).squeeze(1))
                o_err = F.binary_cross_entropy_with_logits(
                    o_logits.squeeze(-1),
                    torch.full((len(opp_variants),), float(target), device=tokens.device),
                    reduction="none", pos_weight=pos_weight)
                for k, (i, t, err_c) in enumerate(failed):
                    w = self.get_weight(t)
                    err_o = o_err[k].item()
                    if err_o < base_error - 1e-6:
                        self._apply_narrow(w, candidates[i]["opp"].numpy(), True, search_rule)
                        w.update_sigma(rewarded=True)
                        stats["rewarded"] += 1
                        _scatter(torch.tensor(w.value, dtype=torch.float32, device=tokens.device),
                                 base_points, t)
                    elif err_c > base_error + 1e-6:
                        self._apply_narrow(w, candidates[i]["cand"].numpy(), False, search_rule)
                        w.update_sigma(rewarded=False)
                        stats["censured"] += 1
                        _scatter(torch.tensor(w.value, dtype=torch.float32, device=tokens.device),
                                 base_points, t)
                    else:
                        stats["unchanged"] += 1
                    final_error = min(final_error, err_c, err_o)

        stats["base_error"] = base_error
        stats["final_error"] = final_error
        stats["mean_delta"] = delta_sum / delta_n if delta_n > 0 else 0.0
        return stats

    @staticmethod
    def _apply_narrow(w: QCloudWeight, new_point, reward: bool, search_rule: str):
        if search_rule == "search":
            w.narrow_search(new_point, reward)
        else:
            w.narrow(new_point, reward)

    # ------------------------------------------------------------- introspection
    def all_converged(self) -> bool:
        if not self.token_weights:
            return False
        return all(w.is_converged() for w in self.token_weights.values())

    def convergence_info(self) -> dict:
        info = {"total_tokens": len(self.token_weights), "converged_tokens": 0,
                "all_converged": False, "mean_width": float("nan")}
        if self.token_weights:
            widths = [float(w.error_width().mean()) for w in self.token_weights.values()]
            info["mean_width"] = float(np.mean(widths))
        for tid, w in self.token_weights.items():
            if w.is_converged():
                info["converged_tokens"] += 1
        info["all_converged"] = self.all_converged()
        return info

    # ------------------------------------------------------------- persistence
    def save_quann(self, path: str, extra: Optional[dict] = None):
        ckpt = {
            "model_state": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size, "d_model": self.d_model,
                "d_cloud": self.d_cloud, "D": self.D,
                "convergence_threshold": self.convergence_threshold,
                "expansion_rate": self.expansion_rate,
            },
            "token_weights": [
                {"token_id": tid, "low": w.low.tolist(), "high": w.high.tolist(),
                 "value": w.value.tolist(), "direction": w.direction.tolist(),
                 "sigma": w.sigma}
                for tid, w in self.token_weights.items()
            ],
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load_quann(cls, path: str, device: torch.device) -> "QCloudFraudClassifier":
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        model = cls(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            d_cloud=cfg["d_cloud"], d_cloud_dim=cfg["D"],
            convergence_threshold=cfg["convergence_threshold"],
            expansion_rate=cfg["expansion_rate"],
        )
        model.load_state_dict(ckpt["model_state"])
        for tw in ckpt["token_weights"]:
            w = QCloudWeight(D=cfg["D"],
                             convergence_threshold=cfg["convergence_threshold"],
                             expansion_rate=cfg["expansion_rate"])
            w.low = np.array(tw["low"], dtype=np.float64)
            w.high = np.array(tw["high"], dtype=np.float64)
            w.value = np.array(tw["value"], dtype=np.float64)
            w.direction = np.array(tw["direction"], dtype=int)
            w.sigma = tw["sigma"]
            model.token_weights[tw["token_id"]] = w
        return model.to(device)

    def __repr__(self):
        return (f"QCloudFraudClassifier(D={self.D}, d_model={self.d_model}, "
                f"d_cloud={self.d_cloud}, vocab={self.vocab_size}, "
                f"quann_points={len(self.token_weights)})")
