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
                 cloud_attn_heads: int = 8):
        super().__init__()
        self.pool = pool
        self.cloud_attn = cloud_attn
        self.cloud_attn_heads = cloud_attn_heads
        self.point_mlp = nn.Sequential(
            nn.Linear(d_cloud_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_cloud),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_cloud_dim + 4, hidden), nn.GELU(),
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
            # Initialize attention weights carefully to avoid NaN
            self.cloud_self_attn = nn.MultiheadAttention(
                embed_dim=d_cloud, num_heads=cloud_attn_heads, batch_first=True,
                dropout=0.0, bias=True  # Dropout=0 for stability initially
            )
            # Scale attention output
            self.attn_scale = nn.Parameter(torch.ones(1) * 0.1)  # Small initial scale

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for legacy checkpoints."""
        # Legacy checkpoint compatibility: old max-pool checkpoints predate
        # attn_proj and cloud_self_attn. Fill missing keys with zeros.
        if self.pool == "max" and not self.cloud_attn:
            prefix = "cloud."
            # Use actual d_cloud dimension from the model
            actual_d_cloud = self.point_mlp[-1].out_features
            if prefix + "attn_proj.weight" not in state_dict:
                state_dict[prefix + "attn_proj.weight"] = torch.zeros(1, actual_d_cloud)
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

        norm_i_exp = norm_i.unsqueeze(2).expand(-1, -1, N, -1)
        norm_j_exp = norm_i.unsqueeze(1).expand(-1, N, -1, -1)

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
                 nhead: Optional[int] = None, pos_len: int = 256,
                 hidden: int = 256, proj_hidden: Optional[int] = None,
                 point_hidden: int = 128, pool: str = "max",
                 cloud_attn: bool = False, cloud_attn_heads: int = 8):
        """
        hidden:      cloud MLP gizli boyutu (nokta+çift MLP) — vars. 256
        proj_hidden: WordToND projeksiyon gizli boyutu — None → d_model
        point_hidden: point_head gizli boyutu — vars. 128
        n_layers:    Transformer katman sayısı — vars. 8 (önceki: 4)
        cloud_attn_heads: Cloud self-attention head sayısı — vars. 8 (önceki: 4)
        """
        super().__init__()
        self.d_cloud_dim = d_cloud_dim
        self.vocab_size = vocab_size
        nhead = nhead or max(1, d_model // 16)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, pos_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.to_nd = WordToND(d_model, d_cloud_dim, hidden=proj_hidden)
        self.cloud = CloudEncoderND(d_cloud_dim, d_cloud, hidden=hidden, pool=pool,
                                    cloud_attn=cloud_attn, cloud_attn_heads=cloud_attn_heads)

        self.word_head = nn.Sequential(
            nn.Linear(d_cloud, d_cloud), nn.GELU(),
            nn.Linear(d_cloud, vocab_size),
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_cloud, point_hidden), nn.GELU(),
            nn.Linear(point_hidden, d_cloud_dim), nn.Tanh(),
        )

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for legacy checkpoints."""
        # Legacy checkpoint compatibility: old max-pool checkpoints predate
        # attn_proj and cloud_self_attn in CloudEncoderND. Fill missing keys.
        cloud_prefix = "cloud."
        if cloud_prefix + "attn_proj.weight" not in state_dict:
            # Get the correct shape from the model
            cloud = self.cloud
            if cloud.pool == "max" and not cloud.cloud_attn:
                state_dict[cloud_prefix + "attn_proj.weight"] = torch.zeros(1, cloud.attn_proj.weight.shape[1])
                state_dict[cloud_prefix + "attn_proj.bias"] = torch.zeros(1)
        super().load_state_dict(state_dict, strict=strict)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_enc[:, :tokens.size(1), :]
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
