"""
QCloudNetwork — CloudLM + QUANN Eğitim
=======================================

Mevcut CloudLMFull mimarisine QUANN eğitim döngüsü eklenir:
- context_encoder: Transformer (4 katman, d_model=256)
- to_nd: WordToND (256 → D boyutlu projeksiyon)
- cloud_encoder: CloudEncoderND (nokta bulut kodlayıcı)
- word_head: Tahmin modülü (cloud_state → logits)
- point_head: Yardımcı nokta tahmini (cloud_state → next_point)
- token_weights: list[QCloudWeight] (her token için QUANN narrownning)
- bidirectional: bool (karşıt yön deneme aktif mi?)

**Fark (standart CloudLM vs QCloudNetwork):**
- Standart: Backprop ile ağırlık güncelleme
- QCloudNetwork: Reward/ceza ile token'ların bulut noktalarının aralığı daraltma

**Backprop'suz eğitim:**
- Gradient YOK
- Credit assignment: Token-level reward
- Search mechanism: Reward/cense + narrownning
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

try:
    from .qcloud_weight import QCloudWeight
except ImportError:
    from qcloud_weight import QCloudWeight


try:
    from cloudlm import WordToND, CloudEncoderND
except ImportError:
    # quann_cloud/src altından doğrudan çalıştırıldığında repo kökünü ekle
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from cloudlm import WordToND, CloudEncoderND


class QCloudNetwork(nn.Module):
    """
    CloudLM + QUANN Eğitim: Backprop'suz Nokta Bulut Modeli.
    
    Attributes:
        context_encoder: nn.TransformerEncoder (bağlam kodlayıcı)
        to_nd: WordToND (256 → D boyutlu projeksiyon)
        cloud_encoder: CloudEncoderND (nokta bulut kodlayıcı)
        word_head: nn.Sequential (cloud_state → logits)
        point_head: nn.Sequential (cloud_state → next_point)
        token_weights: list[QCloudWeight] (her token için QUANN narrownning)
        bidirectional: bool (karşıt yön deneme aktif mi?)
        D: int (bulut boyutu)
        d_model: int (bağlam boyutu)
        d_cloud: int (bulut kodlayıcı boyutu)
        vocab_size: int (vocab boyutu)
        eos_id: int (EOS token ID)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_cloud: int = 512,
        d_cloud_dim: int = 16,
        n_layers: int = 4,
        bidirectional: bool = True,
        convergence_threshold: float = 0.01,
        expansion_rate: float = 0.1,
    ):
        super().__init__()
        self.D = d_cloud_dim
        self.d_model = d_model
        self.d_cloud = d_cloud
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.convergence_threshold = convergence_threshold
        self.expansion_rate = expansion_rate
        self.eos_id = 1
        
        # CloudLM components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=d_model // 16,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
        self.to_nd = WordToND(d_model, d_cloud_dim)
        self.cloud_encoder = CloudEncoderND(d_cloud_dim, d_cloud)
        
        self.word_head = nn.Sequential(
            nn.Linear(d_cloud, d_cloud), nn.GELU(),
            nn.Linear(d_cloud, vocab_size)
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_cloud, 128), nn.GELU(),
            nn.Linear(128, d_cloud_dim), nn.Tanh()
        )
        
        # QUANN components
        # Her TOKEN KİMLİĞİ için QCloudWeight (lazy): token_id → nokta aralığı.
        # "Her kelime → ND uzayda bir nokta" — cümle bulutu, token'ların
        # kimlik bazlı noktalarından kurulur (konum bazlı DEĞİL).
        self.token_weights: Dict[int, QCloudWeight] = {}
    
    def get_weight(self, token_id: int) -> QCloudWeight:
        """Token'ın QCloudWeight'ini döndür (yoksa rastgele başlat)."""
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
        """
        (B, T, D): her token'ın QUANN noktası (kimlik bazlı).
        Padding/EOS pozisyonları 0.
        """
        B, T = tokens.shape
        pts = torch.zeros(B, T, self.D, device=tokens.device)
        for b in range(B):
            for i in range(T):
                t = tokens[b, i].item()
                if t == 0 or t == self.eos_id:
                    continue
                if mask is not None and not mask[b, i].item():
                    continue
                w = self.get_weight(t)
                pts[b, i] = torch.tensor(w.value.tolist(), device=tokens.device)
        return pts
    
    # =========================================================================
    # FORWARD PASS (CloudLM ile aynı)
    # =========================================================================
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        points_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CloudLM forward pass (backprop yok, sadece forward).
        
        Args:
            tokens: torch.Tensor[B, T] — Token ID'leri
            mask: Optional[torch.Tensor[B, T]] — Padding mask
            points_override: Optional[torch.Tensor[B, T, D]] — QUANN noktaları.
                Verilirse to_nd projeksiyonu bypass edilir; bulut doğrudan
                bu noktalardan kurulur (padding pozisyonları 0 olmalı).
            
        Returns:
            word_logits: torch.Tensor[B, vocab_size] — Next token logits
            next_point: torch.Tensor[B, D] — Next token point
            points: torch.Tensor[B, T, D] — Current token points
        """
        seq_len = tokens.size(1)
        
        if points_override is not None:
            # QUANN noktaları verildi → encoder/to_nd gereksiz (hız)
            points = points_override
        else:
            x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
            x = x + self.pos_enc[:, :seq_len, :]
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
            x = self.context_encoder(x, src_key_padding_mask=~mask if mask is not None else None)
            x = self.norm(x)
            points = self.to_nd(x)
        cloud_state = self.cloud_encoder(points, mask)
        word_logits = self.word_head(cloud_state)
        next_point = self.point_head(cloud_state)
        
        return word_logits, next_point, points
    
    # =========================================================================
    # GENERATE (CloudLM ile aynı)
    # =========================================================================
    
    def generate(
        self,
        tokens: torch.Tensor,
        max_new: int = 20,
        temperature: float = 0.8,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Metin üretimi. token_weights doluysa QUANN noktaları kullanılır
        (pozisyon < len(token_weights)); ötesi için to_nd fallback.
        """
        self.eval()
        generated = tokens.clone()
        if eos_id is None:
            eos_id = self.eos_id
        
        with torch.no_grad():
            for _ in range(max_new):
                mask = (generated != 0) & (generated != eos_id)
                T = generated.size(1)
                
                word_logits = None
                if self.token_weights:
                    # QUANN noktaları (kimlik bazlı); bilinmeyen tokenlar 0
                    points = self.points_for(generated, mask)
                    word_logits, _, _ = self.forward(generated, mask, points_override=points)
                else:
                    word_logits, _, _ = self.forward(generated, mask)
                
                probs = F.softmax(word_logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if (nxt == eos_id).all():
                    break
                generated = torch.cat([generated, nxt], dim=1)
        
        return generated
    
    # =========================================================================
    # QUANN EĞİTİM DÖNGÜSÜ (Backprop'suz)
    # =========================================================================
    
    def compute_error(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        target_token: int,
        points_override: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Verilen noktalarla hata hesapla.
        
        Args:
            tokens: (1, T) Token ID'leri
            mask: (1, T) Padding mask
            target_token: Next token ID
            points_override: (1, T, D) — QUANN noktaları (None → to_nd)
            
        Returns:
            float: Cross-entropy loss
        """
        with torch.no_grad():
            word_logits, _, _ = self.forward(tokens, mask, points_override=points_override)
            loss = F.cross_entropy(
                word_logits,
                torch.tensor([target_token], device=word_logits.device),
            )
        return loss.item()
    
    def update_token_weights(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        target_token: int,
        base_points: Optional[torch.Tensor] = None,
        search_rule: str = "classic",
    ) -> dict:
        """
        Her token'ın QCloudWeight'ini reward/cense ile güncelle (backprop yok).
        
        search_rule:
          - "classic": orijinal v0 narrow() (ödül→genişlet, ceza→kötü tarafa daralt,
            value her durumda yeni noktaya)
          - "search": narrow_search() (ödül→segmente daralt, ceza→kötü tarafı kes,
            value cezada sabit) — hibrit mod için
        
        Aday noktalar forward pass'e points_override ile GERÇEKTEN verilir.
        Batch'li değerlendirme: base + tüm adaylar tek forward'da, başarısız
        olanların karşıt yönleri ikinci forward'da ölçülür.
        
        Args:
            tokens: (1, T) Token ID'leri
            mask: (1, T) bool padding mask
            target_token: Next token ID
            base_points: (1, T, D) — Mevcut QUANN noktaları
                (token_weights value'larından; kabul edildikçe güncellenir)
            
        Returns:
            dict: {'rewarded', 'censured', 'unchanged', 'base_error', 'final_error',
                   'mean_delta'}
        """
        stats = {'rewarded': 0, 'censured': 0, 'unchanged': 0}
        
        with torch.no_grad():
            T = tokens.size(1)
            if base_points is None:
                base_points = self.points_for(tokens, mask)
            
            # Geçerli pozisyonlar + token kimlikleri
            valid = [(i, tokens[0, i].item()) for i in range(T)
                     if mask[0, i].item()
                     and tokens[0, i].item() not in (0, self.eos_id)]
            
            # PASS 1: base + her pozisyon için birer aday (tek batch'li forward)
            # Aday, o token'ın TÜM görünümlerini değiştirir (aynı kelime)
            candidates = {}
            variants = [base_points]
            for i, t in valid:
                w = self.get_weight(t)
                c = w.sample()
                vp = base_points.clone()
                for j in range(T):
                    if tokens[0, j].item() == t:
                        vp[0, j] = torch.tensor(c.tolist(), device=tokens.device)
                candidates[(i, t)] = {'cand': c, 'opp': None}
                variants.append(vp)
            
            var_tok = tokens.expand(len(variants), -1)
            var_mask = mask.expand(len(variants), -1)
            var_logits, _, _ = self.forward(var_tok, var_mask,
                                            points_override=torch.stack(variants).squeeze(1))
            var_err = F.cross_entropy(
                var_logits,
                torch.full((len(variants),), target_token, device=tokens.device),
                reduction='none'
            )
            base_error = var_err[0].item()
            final_error = base_error
            delta_sum = (var_err[1:] - base_error).sum().item()
            delta_n = len(valid)
            
            def _scatter(new_val, src_pts, token_id):
                for j in range(T):
                    if tokens[0, j].item() == token_id:
                        src_pts[0, j] = torch.tensor(new_val.tolist(), device=tokens.device)
            
            # Ödül adaylarını uygula; başarısızları karşıt yöne ilet
            failed = []
            for k, (i, t) in enumerate(valid):
                err_c = var_err[k + 1].item()
                w = self.get_weight(t)
                if err_c < base_error - 1e-6:
                    self._apply_narrow(w, candidates[(i, t)]['cand'], True, search_rule)
                    w.update_sigma(rewarded=True)
                    stats['rewarded'] += 1
                    _scatter(w.value, base_points, t)
                else:
                    failed.append((i, t, err_c))
            
            # PASS 2: başarısız pozisyonlar için karşıt yön (tek batch'li forward)
            if failed and self.bidirectional:
                opp_variants = []
                for i, t, _ in failed:
                    w = self.get_weight(t)
                    opp = w.opposite(w.value)
                    vp = base_points.clone()
                    _scatter(opp, vp, t)
                    candidates[(i, t)]['opp'] = opp
                    opp_variants.append(vp)
                
                var_logits, _, _ = self.forward(
                    var_tok[:len(opp_variants)], var_mask[:len(opp_variants)],
                    points_override=torch.stack(opp_variants).squeeze(1)
                )
                var_err = F.cross_entropy(
                    var_logits,
                    torch.full((len(opp_variants),), target_token, device=tokens.device),
                    reduction='none'
                )
                
                for k, (i, t, err_c) in enumerate(failed):
                    w = self.get_weight(t)
                    err_o = var_err[k].item()
                    if err_o < base_error - 1e-6:
                        # Karşıt yönde iyileşme → doğru yön bulundu
                        self._apply_narrow(w, candidates[(i, t)]['opp'], True, search_rule)
                        w.update_sigma(rewarded=True)
                        stats['rewarded'] += 1
                        _scatter(w.value, base_points, t)
                    elif err_c > base_error + 1e-6:
                        # Her iki yön de kötü → ceza
                        self._apply_narrow(w, candidates[(i, t)]['cand'], False, search_rule)
                        w.update_sigma(rewarded=False)
                        stats['censured'] += 1
                        _scatter(w.value, base_points, t)
                    else:
                        stats['unchanged'] += 1
            elif failed:
                for i, t, _ in failed:
                    w = self.get_weight(t)
                    self._apply_narrow(w, candidates[(i, t)]['cand'], False, search_rule)
                    w.update_sigma(rewarded=False)
                    stats['censured'] += 1
                    _scatter(w.value, base_points, t)

            # Measure the actual post-update state, not the best isolated
            # candidate (which may not represent the combined updated cloud).
            final_logits, _, _ = self.forward(
                tokens, mask, points_override=base_points
            )
            final_error = F.cross_entropy(
                final_logits,
                torch.tensor([target_token], device=final_logits.device),
            ).item()
        
        stats['base_error'] = base_error
        stats['final_error'] = final_error
        if delta_n > 0:
            stats['mean_delta'] = delta_sum / delta_n
        return stats
    
    @staticmethod
    def _apply_narrow(w, new_point, reward, search_rule):
        if search_rule == "search":
            w.narrow_search(new_point, reward)
        else:
            w.narrow(new_point, reward)
    
    def all_converged(self) -> bool:
        """Tüm token_weights yakınsadı mı?"""
        if not self.token_weights:
            return False
        return all(w.is_converged() for w in self.token_weights.values())
    
    def convergence_info(self) -> dict:
        """Tüm token_weights'in yakınsama bilgisi."""
        info = {
            'total_tokens': len(self.token_weights),
            'converged_tokens': 0,
            'all_converged': False,
            'token_info': [],
        }
        
        for tid, w in self.token_weights.items():
            info['token_info'].append({
                'token_id': tid,
                'converged': w.is_converged(),
                'width_mean': w.error_width().mean(),
                'sigma': w.sigma,
            })
            if w.is_converged():
                info['converged_tokens'] += 1
        
        info['all_converged'] = self.all_converged()
        return info
    
    def training_stats(self) -> dict:
        """Eğitim istatistiklerini özetle."""
        total_rewarded = 0
        total_censured = 0
        
        for w in self.token_weights.values():
            for val, direction, outcome in w.history:
                if outcome == 'reward':
                    total_rewarded += 1
                elif outcome == 'censure':
                    total_censured += 1
        
        total = total_rewarded + total_censured
        return {
            'total_rewards': total_rewarded,
            'total_censures': total_censured,
            'total_iterations': total,
            'reward_ratio': total_rewarded / total if total > 0 else 0,
        }
    
    def reset_token_weights(self) -> None:
        """Tüm token_weights'i sıfırla (yeni eğitim için)."""
        for w in self.token_weights.values():
            w.reset()
    
    def __repr__(self) -> str:
        bidir = "bidir" if self.bidirectional else "simple"
        return (
            f"QCloudNetwork(D={self.D}, d_model={self.d_model}, "
            f"d_cloud={self.d_cloud}, layers={self.context_encoder.num_layers}, "
            f"mode={bidir}, tokens={len(self.token_weights)})"
        )


# =========================================================================
# TEST
# =========================================================================

if __name__ == "__main__":
    print("=== QCloudNetwork Test ===\n")
    
    vocab_size = 100  # Küçük vocab (test için)
    model = QCloudNetwork(
        vocab_size=vocab_size,
        d_model=64,  # Küçük d_model (test için)
        d_cloud=128,  # Küçük d_cloud (test için)
        d_cloud_dim=3,  # D=3 (test için)
        n_layers=2,  # Küçük n_layers (test için)
        bidirectional=True,
    )
    
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Basit test: 5 token'lık cümle
    tokens = torch.tensor([[1, 2, 3, 4, 5]])  # Batch=1, Seq=5
    mask = (tokens != 0) & (tokens != 1)
    
    # Token weights: lazy — first forward/update oluşturur
    print(f"Token weights: lazy (token kimliği bazlı)\n")

    # Forward pass test
    print("--- Forward Pass Test ---")
    word_logits, next_point, points = model(tokens, mask)
    print(f"Word logits shape: {word_logits.shape}")  # (1, vocab_size)
    print(f"Next point shape: {next_point.shape}")  # (1, D=3)
    print(f"Points shape: {points.shape}")  # (1, 5, D=3)
    print(f"Points: {points[0].detach().cpu().numpy()}\n")
    
    # Generate test
    print("--- Generate Test ---")
    generated = model.generate(tokens, max_new=3)
    print(f"Generated tokens: {generated[0].tolist()}")  # 5 + 3 = 8 token
    print(f"Generated shape: {generated.shape}\n")
    
    # QUANN training test
    print("--- QUANN Training Test ---")
    target_token = 5
    base_error = model.compute_error(tokens, mask, target_token)
    print(f"Base error: {base_error:.4f}\n")
    
    # Birkaç token'ı güncelle (base_points=None → iç points_for ile hesaplanır)
    for step in range(5):
        stats = model.update_token_weights(tokens, mask, target_token)
        print(f"Step {step+1}: {stats}")
    print(f"Token weights created: {len(model.token_weights)}")
    
    print(f"\nConvergence info: {model.convergence_info()}")
    print(f"Training stats: {model.training_stats()}")
    
    print("\n=== Test Passed ===")
