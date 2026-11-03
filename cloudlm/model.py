"""
CloudLM — Cümle Bulutu Dil Modeli (kanonik ND mimari)
======================================================
Token → [Transformer] → [WordToND] → [CloudEncoderND] → [word_head + point_head]

- Her kelime → D boyutlu uzayda bir nokta (Tanh ile [-1, 1])
- Cümle → nokta bulutu; anlam → bulut geometrisi (mesafe + açı)
- Tahmin → bulut konfigürasyonundan bir sonraki kelime (+ nokta)

Sıralama: set (unordered) — nokta çiftleri üzerinden pozisyonsuz.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordToND(nn.Module):
    """Bağlam vektörünü (d_model) → D boyutlu bulut noktasına projekte eder."""

    def __init__(self, d_model: int, d_cloud_dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_cloud_dim), nn.Tanh(),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.proj(ctx)  # (B, N, D)


class CloudEncoderND(nn.Module):
    """
    ND nokta bulutunu (B, N, D) → küme durumu (B, d_cloud) olarak kodlar.

    PointNet mantığı:
      a) nokta MLP (paylaşılan) → max/attention-pool
      b) çift geometri MLP: [diff(D), dist, cos_açı, |i|, |j|] → mean-pool
      c) fusion
    Sıralama YOK — küme olarak işler; mask ile padding çiftleri sıfırlanır.
    """

    def __init__(self, d_cloud_dim: int, d_cloud: int = 512, hidden: int = 256,
                 pool: str = "max", cloud_attn: bool = False,
                 cloud_attn_heads: int = 8, attn_scale_init: float = 1.0):
        super().__init__()
        self.d_cloud_dim = d_cloud_dim  # Store for load_state_dict compatibility
        self.pool = pool
        self.cloud_attn = cloud_attn
        self.cloud_attn_heads = cloud_attn_heads
        self.point_mlp = nn.Sequential(
            nn.Linear(d_cloud_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_cloud),
        )
        # Pair input: diff(D) + dist(1) + cos(1) + |i|(D) + |j|(D) = 3*D + 2
        pair_input_dim = 3 * d_cloud_dim + 2
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_cloud),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_cloud * 2, d_cloud * 2), nn.GELU(),
            nn.Linear(d_cloud * 2, d_cloud),
            nn.LayerNorm(d_cloud),
        )
        self.attn_proj = nn.Linear(d_cloud, 1)
        
        if cloud_attn:
            self.cloud_self_attn = nn.MultiheadAttention(
                embed_dim=d_cloud, num_heads=cloud_attn_heads, batch_first=True,
                dropout=0.0, bias=True
            )
            self.attn_scale = nn.Parameter(torch.ones(1) * attn_scale_init)

    def _fix_legacy_pair_mlp_weight(self, state_dict, prefix: str) -> None:
        """Fix legacy pair_mlp weight for backward compatibility.
        
        FIXED: Does NOT mutate the original state_dict; works on a copy.
        """
        d = self.d_cloud_dim
        old_pair_in = d + 4  # Legacy bug: used d_cloud_dim + 4
        new_pair_in = 3 * d + 2  # Correct: 3*d_cloud_dim + 2
        
        pair_weight_key = prefix + "pair_mlp.0.weight"
        if pair_weight_key not in state_dict:
            return
        
        old_w = state_dict[pair_weight_key]
        if old_w.shape[1] != old_pair_in:
            return  # Not a legacy checkpoint
        
        # Create new weight (don't mutate original)
        new_w = torch.zeros(old_w.shape[0], new_pair_in, device=old_w.device, dtype=old_w.dtype)
        new_w[:, :old_pair_in] = old_w
        # For new |i|(D) and |j|(D) dimensions, reuse the scalar weights
        new_w[:, d + 2:2 * d + 2] = old_w[:, d + 2].unsqueeze(1).expand(-1, d)
        new_w[:, 2 * d + 2:3 * d + 2] = old_w[:, d + 3].unsqueeze(1).expand(-1, d)
        state_dict[pair_weight_key] = new_w

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for legacy checkpoints.
        
        FIXED: Works on a copy of state_dict to avoid mutating the caller's dict.
        """
        # Work on a copy to avoid mutating the original state_dict
        state_dict = dict(state_dict)
        prefix = "cloud."
        self._fix_legacy_pair_mlp_weight(state_dict, prefix)
        
        # Legacy: old max-pool checkpoints predate attn_proj and cloud_self_attn
        if self.pool == "max" and not self.cloud_attn:
            if prefix + "attn_proj.weight" not in state_dict:
                state_dict[prefix + "attn_proj.weight"] = torch.zeros(1, self.d_cloud)
            if prefix + "attn_proj.bias" not in state_dict:
                state_dict[prefix + "attn_proj.bias"] = torch.zeros(1)
        
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = points.shape

        pf = self.point_mlp(points)
        
        # R2: Cloud attention (nokta-üstü self-attention, pooling öncesi)
        if self.cloud_attn:
            # key_padding_mask: True = padding (ignore), False = keep
            # Our mask: True = keep, False = padding
            # So we need to invert it
            pf, _ = self.cloud_self_attn(pf, pf, pf, key_padding_mask=~mask)
            # Scale attention output for stability
            pf = pf * self.attn_scale
        
        if self.pool == "attn":
            scores = self.attn_proj(pf).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)
            w = F.softmax(scores, dim=1)
            point_pooled = (pf * w.unsqueeze(-1)).sum(dim=1)
        else:
            if mask is not None:
                pf = pf.masked_fill(~mask.unsqueeze(-1), torch.finfo(pf.dtype).min)
            point_pooled = pf.max(dim=1).values
            if mask is not None:
                point_pooled = point_pooled.masked_fill(~mask.any(dim=1, keepdim=True), 0)

        diff = points.unsqueeze(2) - points.unsqueeze(1)
        dist = diff.norm(dim=-1)
        norm_i = points.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_ang = (points.unsqueeze(2) * points.unsqueeze(1)).sum(-1) / \
                  (norm_i * norm_i.transpose(1, 2)).clamp(min=1e-8)

        # Expand norms to match diff dimension (D)
        norm_i_exp = norm_i.unsqueeze(2).expand(-1, -1, N, D)
        norm_j_exp = norm_i.unsqueeze(1).expand(-1, N, -1, D)

        pair_in = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1),
                             norm_i_exp, norm_j_exp], dim=-1)
        pair_feats = self.pair_mlp(pair_in)

        if mask is not None:
            pmask = mask.unsqueeze(2) & mask.unsqueeze(1)
            pair_feats = pair_feats.masked_fill(~pmask.unsqueeze(-1), 0)
            pair_count = pmask.float().sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)  # (B, 1)
        else:
            pair_count = torch.ones(B, 1, device=points.device) * N * N
        pair_pooled = pair_feats.sum(dim=(1, 2)) / pair_count

        return self.fusion(torch.cat([point_pooled, pair_pooled], dim=-1))


class PositionalEncoding(nn.Module):
    """Dinamik sine/cosine positional encoding - her sequence uzunluğu için çalışır."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Shape: (max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D) + positional encoding"""
        # pe shape: (max_len, D), need to index [0:N, :] then add to x
        return x + self.pe[:x.size(1), :]


class CloudLM(nn.Module):
    """
    Cümle Bulutu Dil Modeli.

    Çıktılar (forward):
      word_logits: (B, vocab)  — bir sonraki kelime
      next_point:  (B, D)      — bir sonraki kelimenin bulut konumu (aux)
      points:      (B, N, D)   — mevcut cümlenin bulut noktaları
    """

    def __init__(self, vocab_size: int, d_model: int = 256, d_cloud: int = 512,
                 d_cloud_dim: int = 16, n_layers: int = 8,
                 nhead: Optional[int] = None, max_pos_len: int = 2048,
                 hidden: int = 256, proj_hidden: Optional[int] = None,
                 point_hidden: int = 128, pool: str = "max",
                 cloud_attn: bool = False, cloud_attn_heads: int = 8,
                 attn_scale_init: float = 1.0):
        """
        hidden:      cloud MLP gizli boyutu (nokta+çift MLP) — vars. 256
        proj_hidden: WordToND projeksiyon gizli boyutu — None → d_model
        point_hidden: point_head gizli boyutu — vars. 128
        n_layers:    Transformer katman sayısı — vars. 8 (önceki: 4)
        max_pos_len: Maksimum pozisyon uzunluğu (vars. 2048, eskiden 256)
        cloud_attn_heads: Cloud self-attention head sayısı — vars. 8 (önceki: 4)
        attn_scale_init: Cloud attention scale başlangıç değeri — vars. 1.0
        """
        super().__init__()
        self.d_cloud_dim = d_cloud_dim
        self.vocab_size = vocab_size
        nhead = nhead or max(1, d_model // 16)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_pos_len)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.to_nd = WordToND(d_model, d_cloud_dim, hidden=proj_hidden)
        self.cloud = CloudEncoderND(d_cloud_dim, d_cloud, hidden=hidden, pool=pool,
                                    cloud_attn=cloud_attn, cloud_attn_heads=cloud_attn_heads,
                                    attn_scale_init=attn_scale_init)

        self.word_head = nn.Sequential(
            nn.Linear(d_cloud, d_cloud), nn.GELU(),
            nn.Linear(d_cloud, vocab_size),
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_cloud, point_hidden), nn.GELU(),
            nn.Linear(point_hidden, d_cloud_dim), nn.Tanh(),
        )

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for legacy checkpoints.
        
        FIXED: Works on a copy of state_dict to avoid mutating the caller's dict.
        Uses shared _fix_legacy_pair_mlp_weight helper.
        """
        # Work on a copy to avoid mutating the original state_dict
        state_dict = dict(state_dict)
        cloud_prefix = "cloud."
        self.cloud._fix_legacy_pair_mlp_weight(state_dict, cloud_prefix)
        
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        x = self.norm(x)
        points = self.to_nd(x)
        cloud_state = self.cloud(points, mask)
        return self.word_head(cloud_state), self.point_head(cloud_state), points

    def generate(self, tokens: torch.Tensor, max_new: int = 20, temperature: float = 0.8,
                 eos_id: int = 1, top_k: int = 50) -> torch.Tensor:
        self.eval()
        generated = tokens.clone()
        with torch.no_grad():
            for _ in range(max_new):
                mask = (generated != 0) & (generated != eos_id)
                logits, _, _ = self.forward(generated, mask)
                if top_k and top_k < logits.size(-1):
                    indices = torch.argsort(logits, dim=-1, descending=True)[:, :top_k]
                    mask_k = torch.ones_like(logits, dtype=torch.bool)
                    mask_k.scatter_(1, indices, False)
                    logits.masked_fill_(mask_k, float('-inf'))
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if (nxt == eos_id).all():
                    break
                generated = torch.cat([generated, nxt], dim=1)
        return generated

    def get_cloud(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Cümlenin bulut noktaları (numpy, (N, D))."""
        self.eval()
        with torch.no_grad():
            _, _, points = self.forward(tokens, mask)
        return points[0].cpu().numpy()
