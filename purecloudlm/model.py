"""
PureCloudLM — Transformer-Free, O(N) Geometry-Only Language Model
===================================================================

Temel Fikir:
- Token → Embedding → ND Nokta → Global Geometry → Tahmin
- Transformer YOK (O(N²) attention yok)
- Pair computation YOK veya APPROXIMATE (O(N) veya O(N log N))
- Sadece küme istatistikleri (centroid, variance, max, min)

Mimari:
  Token → Embedding (d_model) → WordToND (D) → Global Pooling → Tahmin
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sine/cosine positional encoding - sequence order'ı korumak için"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D)"""
        return x + self.pe[:, :x.size(1), :]  # (B, N, D)


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


class GlobalGeometryEncoder(nn.Module):
    """
    ND nokta bulutunu (B, N, D) → küme durumu (B, d_cloud) olarak kodlar.
    
    O(N) yaklaşımı:
      a) Global pooling (max, min, mean, std) - O(N)
      b) Centroid + variance - O(N)
      c) Pair computation YOK (veya approximate)
    
    Sıralama YOK — küme olarak işler.
    """

    def __init__(self, d_cloud_dim: int, d_cloud: int = 512, hidden: int = 256,
                 pool_mode: str = "all", use_pair_approx: bool = False,
                 num_sampled_pairs: int = 64, use_positional: bool = True,
                 max_len: int = 256):
        super().__init__()
        self.d_cloud_dim = d_cloud_dim
        self.pool_mode = pool_mode
        self.use_pair_approx = use_pair_approx
        self.num_sampled_pairs = num_sampled_pairs
        self.use_positional = use_positional
        
        # Positional encoding (sine/cosine) - sequence order'ı korumak için
        if use_positional:
            self.pos_encoder = PositionalEncoding(d_cloud_dim, max_len=max_len)
        
        # Optional: approximate pair statistics (sampled pairs only)
        if use_pair_approx:
            self.pair_sampler = PairSampler(num_sampled_pairs)
            # Sampled pair features: [diff, dist, cos] for K pairs
            # K × (D + 1 + 1) = K × (D + 2)
            pair_feat_dim = num_sampled_pairs * (d_cloud_dim + 2)
            self.pair_mlp = nn.Sequential(
                nn.Linear(pair_feat_dim, hidden), nn.GELU(),
                nn.Linear(hidden, d_cloud),
            )
        
        # Global statistics: mean, max, min, std (varying sizes based on pool_mode)
        if self.pool_mode == "mean":
            stats_dim = d_cloud_dim * 2  # mean + std
        elif self.pool_mode == "max":
            stats_dim = d_cloud_dim * 2  # max + std
        elif self.pool_mode == "min":
            stats_dim = d_cloud_dim * 2  # min + std
        elif self.pool_mode == "all":
            stats_dim = d_cloud_dim * 4  # mean + max + min + std
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
        
        # Optional: approximate pair statistics adds to stats_dim
        if use_pair_approx:
            stats_dim += d_cloud
        
        # Final fusion - stats_dim dinamik olarak hesaplanıyor
        self.fusion = nn.Sequential(
            nn.Linear(stats_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_cloud),
            nn.LayerNorm(d_cloud),
        )

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: (B, N, D) - nokta bulutu
            mask: (B, N) - True = valid, False = padding
        
        Returns:
            cloud_state: (B, d_cloud)
        """
        B, N, D = points.shape
        
        # O(N) global statistics
        if mask is not None:
            # Masked pooling
            masked_points = points.masked_fill(~mask.unsqueeze(-1), 0)
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            
            # Mean (B, D)
            mean = masked_points.sum(dim=1) / counts
            
            # Max (only over valid points)
            max_val = points.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            max_pool = max_val.max(dim=1).values
            
            # Min (only over valid points)
            min_val = points.masked_fill(~mask.unsqueeze(-1), float('inf'))
            min_pool = min_val.min(dim=1).values
            
            # Std (only over valid points)
            mean_expanded = mean.unsqueeze(1).expand(-1, N, -1)
            diff = points - mean_expanded
            variance = (diff ** 2).masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / counts
            # Keep the derivative finite when a masked sequence has one valid
            # point (zero variance), while preserving an exact zero std value.
            epsilon = torch.finfo(variance.dtype).eps
            std = (variance + epsilon).sqrt() - math.sqrt(epsilon)
        else:
            mean = points.mean(dim=1)
            max_pool = points.max(dim=1).values
            min_pool = points.min(dim=1).values
            std = points.std(dim=1)
        
        # pool_mode'a göre global statistics seçimi
        if self.pool_mode == "mean":
            global_stats = torch.cat([mean, std], dim=-1)
        elif self.pool_mode == "max":
            global_stats = torch.cat([max_pool, std], dim=-1)
        elif self.pool_mode == "min":
            global_stats = torch.cat([min_pool, std], dim=-1)
        elif self.pool_mode == "all":
            global_stats = torch.cat([mean, max_pool, min_pool, std], dim=-1)
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")
        
        # Optional: approximate pair statistics (sampled pairs only)
        if self.use_pair_approx:
            pair_features = self.pair_sampler(points, mask)
            pair_embed = self.pair_mlp(pair_features)
            cloud_state = self.fusion(torch.cat([global_stats, pair_embed], dim=-1))
        else:
            cloud_state = self.fusion(global_stats)
        
        return cloud_state


class PairSampler(nn.Module):
    """
    O(N log N) veya O(N) approximate pair sampling.
    
    Tüm çiftler yerine:
      - Rastgele örneklem (O(K), K << N²)
      - veya centroid'e en yakın K nokta
      - veya random projection ile clustering
    """

    def __init__(self, num_sampled_pairs: int, sampling_method: str = "random"):
        super().__init__()
        self.num_sampled_pairs = num_sampled_pairs
        self.sampling_method = sampling_method

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample K pairs from N points.
        
        Args:
            points: (B, N, D)
            mask: (B, N)
        
        Returns:
            pair_features: (B, K × (D + 2))
        """
        B, N, D = points.shape
        K = self.num_sampled_pairs
        
        if self.sampling_method == "random":
            # Random sampling: O(K)
            valid_mask = mask if mask is not None else torch.ones(B, N, dtype=torch.bool, device=points.device)
            
            pair_features = []
            for b in range(B):
                valid_indices = torch.where(valid_mask[b])[0]
                if len(valid_indices) < 2:
                    # Not enough points, repeat last point
                    idx_i = torch.full((K,), valid_indices[-1].item(), device=points.device)
                    idx_j = torch.full((K,), valid_indices[-1].item(), device=points.device)
                else:
                    # Sample K pairs
                    idx_i = torch.randint(0, len(valid_indices), (K,), device=points.device)
                    idx_j = torch.randint(0, len(valid_indices), (K,), device=points.device)
                    idx_i = valid_indices[idx_i]
                    idx_j = valid_indices[idx_j]
                
                # Extract pair features: [diff, dist, cos]
                p_i = points[b, idx_i]  # (K, D)
                p_j = points[b, idx_j]  # (K, D)
                
                diff = p_i - p_j  # (K, D)
                dist = diff.norm(dim=-1)  # (K,)
                norm_i = p_i.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (K, 1)
                norm_j = p_j.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (K, 1)
                cos_ang = (p_i * p_j).sum(dim=-1) / (norm_i.squeeze(-1) * norm_j.squeeze(-1))  # (K,)
                
                pair_feat = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1)], dim=-1)
                pair_features.append(pair_feat)
            
            result = torch.stack(pair_features, dim=0)  # (B, K, D+2)
            return result.view(B, -1)  # (B, K × (D + 2))
        
        elif self.sampling_method == "centroid":
            # Sample pairs near centroid: O(N + K)
            centroid = points.mean(dim=1)  # (B, D)
            
            pair_features = []
            for b in range(B):
                # Compute distances to centroid
                dist_to_centroid = (points[b] - centroid[b]).norm(dim=-1)  # (N,)
                
                # Select K closest points to centroid
                _, closest_indices = torch.topk(dist_to_centroid, min(K, N), largest=False)
                
                # Form pairs from closest points
                if len(closest_indices) >= 2:
                    idx_i = closest_indices[:K]
                    idx_j = closest_indices[:K]
                else:
                    idx_i = closest_indices
                    idx_j = closest_indices
                
                p_i = points[b, idx_i]
                p_j = points[b, idx_j]
                
                diff = p_i - p_j
                dist = diff.norm(dim=-1)
                norm_i = p_i.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (K, 1)
                norm_j = p_j.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (K, 1)
                cos_ang = (p_i * p_j).sum(dim=-1) / (norm_i.squeeze(-1) * norm_j.squeeze(-1))  # (K,)
                
                pair_feat = torch.cat([diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1)], dim=-1)
                pair_features.append(pair_feat)
            
            result = torch.stack(pair_features, dim=0)  # (B, K, D+2)
            return result.view(B, -1)  # (B, K × (D + 2))
        
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")


class PureCloudLM(nn.Module):
    """
    PureCloudLM — Transformer-Free Language Model
    
    forward çıktısı:
      word_logits: (B, vocab)  — bir sonraki kelime
      next_point:  (B, D)      — bir sonraki kelimenin bulut konumu (aux)
      points:      (B, N, D)   — mevcut cümlenin bulut noktaları
    """

    def __init__(self, vocab_size: int, d_model: int = 256, d_cloud: int = 512,
                 d_cloud_dim: int = 16, n_layers: int = 1,
                 pos_len: int = 256, hidden: int = 256,
                 proj_hidden: Optional[int] = None, point_hidden: int = 128,
                 pool_mode: str = "all", use_pair_approx: bool = False,
                 num_sampled_pairs: int = 64):
        """
        d_cloud_dim: Nokta boyutu (D) — 16, 32, 64
        use_pair_approx: Pair computation kullan (O(K) yerine O(N²))
        num_sampled_pairs: Kaç pair örneklemesi (K)
        """
        super().__init__()
        self.d_cloud_dim = d_cloud_dim
        self.vocab_size = vocab_size
        
        # Embedding + Dynamic Positional Encoding (NO Transformer!)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=2048)
        
        # Word to ND projection
        self.to_nd = WordToND(d_model, d_cloud_dim, hidden=proj_hidden)
        
        # Global Geometry Encoder (O(N))
        self.cloud = GlobalGeometryEncoder(
            d_cloud_dim=d_cloud_dim,
            d_cloud=d_cloud,
            hidden=hidden,
            pool_mode=pool_mode,
            use_pair_approx=use_pair_approx,
            num_sampled_pairs=num_sampled_pairs
        )
        
        # Prediction heads
        self.word_head = nn.Sequential(
            nn.Linear(d_cloud, d_cloud), nn.GELU(),
            nn.Linear(d_cloud, vocab_size),
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_cloud, point_hidden), nn.GELU(),
            nn.Linear(point_hidden, d_cloud_dim), nn.Tanh(),
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        PureCloudLM forward pass (NO Transformer!)
        
        Steps:
          1. Embedding + Dynamic Positional Encoding
          2. Word → ND projection
          3. Global Geometry Encoding (O(N))
          4. Prediction
        """
        # Step 1: Embedding + Dynamic Positional Encoding
        x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        
        # Optional: Mask padding
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        
        # Step 2: Word → ND projection (NO Transformer!)
        points = self.to_nd(x)  # (B, N, D)
        
        # Step 3: Global Geometry Encoding (O(N))
        cloud_state = self.cloud(points, mask)
        
        # Step 4: Prediction
        word_logits = self.word_head(cloud_state)
        next_point = self.point_head(cloud_state)
        
        return word_logits, next_point, points

    def generate(self, tokens: torch.Tensor, max_new: int = 20, temperature: float = 0.8,
                 eos_id: int = 1, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation"""
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
        """Cümlenin bulut noktaları (numpy, (N, D))"""
        self.eval()
        with torch.no_grad():
            _, _, points = self.forward(tokens, mask)
        return points[0].cpu().numpy()
