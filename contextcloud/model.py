"""Transformer-free contextual language modeling with sentence and window clouds."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from purecloudlm.model import GlobalGeometryEncoder, WordToND


logger = logging.getLogger(__name__)


class SentenceBulutEncoder(nn.Module):
    """Encode each token sequence as one 512D sentence-cloud state."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_cloud_dim: int = 16,
        d_cloud: int = 512,
        proj_hidden: Optional[int] = None,
        hidden: int = 256,
        max_tokens: int = 2048,
        geometry_max_tokens: int = 256,
        padding_idx: int = 0,
    ) -> None:
        """Build the embedding, WordToND projection, and O(N) geometry encoder."""
        super().__init__()
        self.padding_idx = padding_idx
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_tokens)
        self.to_nd = WordToND(d_model, d_cloud_dim, hidden=proj_hidden)
        self.sentence_encoder = GlobalGeometryEncoder(
            d_cloud_dim=d_cloud_dim,
            d_cloud=d_cloud,
            hidden=hidden,
            pool_mode="all",
            use_pair_approx=False,
            use_positional=True,
            max_len=geometry_max_tokens,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return sentence states for tokens shaped ``(B, T)``."""
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape (batch, tokens), got {tuple(tokens.shape)}")
        if tokens.size(1) > self.max_tokens:
            raise ValueError(
                f"token length {tokens.size(1)} exceeds configured max_tokens={self.max_tokens}"
            )
        if tokens.size(1) == 0:
            raise ValueError("each sentence must contain at least one token position")

        token_mask = mask if mask is not None else tokens.ne(self.padding_idx)
        if token_mask.shape != tokens.shape:
            raise ValueError("mask must have the same shape as tokens")
        token_mask = token_mask.to(dtype=torch.bool, device=tokens.device)

        embedded = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_enc(embedded)
        embedded = embedded.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        points = self.to_nd(embedded)

        # GlobalGeometryEncoder's masked statistics require at least one valid point.
        # Empty/padded sentences receive a finite temporary point and are masked by
        # ContextCloudLM before sentence-level pooling.
        safe_mask = token_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        states = self.sentence_encoder(points, safe_mask)
        logger.debug("Encoded %d sentence clouds", tokens.size(0))
        return states

    def load_purecloudlm_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        strict: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        """Transfer shape-compatible PureCloudLM embedding/cloud weights into this encoder."""
        raw_state: Any = checkpoint
        if isinstance(checkpoint, Mapping):
            for key in ("model_state", "model_state_dict", "state_dict"):
                candidate = checkpoint.get(key)
                if isinstance(candidate, Mapping):
                    raw_state = candidate
                    break
        if not isinstance(raw_state, Mapping):
            raise TypeError("checkpoint must be a state dict or contain model_state/model_state_dict")

        own_state = self.state_dict()
        transferred: dict[str, torch.Tensor] = {}
        for source_key, value in raw_state.items():
            if not isinstance(source_key, str) or not isinstance(value, torch.Tensor):
                continue
            key = source_key.removeprefix("module.")
            if key.startswith("cloud."):
                key = "sentence_encoder." + key.removeprefix("cloud.")
            target = own_state.get(key)
            if target is not None and target.shape == value.shape:
                transferred[key] = value

        if not transferred:
            raise ValueError("no shape-compatible PureCloudLM weights found in checkpoint")
        logger.info("Transferred %d PureCloudLM tensors into SentenceBulutEncoder", len(transferred))
        return self.load_state_dict(transferred, strict=strict)


class WindowBulutEncoder(nn.Module):
    """Create overlapping fixed-size sentence windows and encode each as a cloud."""

    def __init__(
        self,
        sentence_dim: int = 512,
        window_size: int = 5,
        stride: int = 2,
        d_window_dim: int = 32,
        d_cloud: int = 512,
        hidden: int = 256,
        geometry_max_len: int = 256,
    ) -> None:
        """Build the sentence projection and O(window-size) geometry encoder."""
        super().__init__()
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self.window_size = window_size
        self.stride = stride
        self.sentence_proj = nn.Linear(sentence_dim, d_window_dim)
        self.window_encoder = GlobalGeometryEncoder(
            d_cloud_dim=d_window_dim,
            d_cloud=d_cloud,
            hidden=hidden,
            pool_mode="all",
            use_pair_approx=False,
            use_positional=True,
            max_len=geometry_max_len,
        )

    def forward(
        self,
        sentence_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return window states ``(B, W, D)`` and their validity mask ``(B, W)``."""
        if sentence_states.ndim != 3:
            raise ValueError("sentence_states must have shape (batch, sentences, dim)")
        batch_size, sentence_count, _ = sentence_states.shape
        if sentence_count == 0:
            raise ValueError("at least one sentence position is required")
        if mask is None:
            mask = torch.ones(
                batch_size, sentence_count, dtype=torch.bool, device=sentence_states.device
            )
        elif mask.shape != (batch_size, sentence_count):
            raise ValueError("sentence mask must have shape (batch, sentences)")
        else:
            mask = mask.to(dtype=torch.bool, device=sentence_states.device)

        windows = sliding_window(sentence_states, self.window_size, self.stride)
        windowed_mask = sliding_window(mask.unsqueeze(-1), self.window_size, self.stride).squeeze(-1)
        valid_windows = windowed_mask.any(dim=-1)
        if window_mask is not None:
            if window_mask.shape != valid_windows.shape:
                raise ValueError("window_mask must have shape (batch, number_of_windows)")
            valid_windows = valid_windows & window_mask.to(
                dtype=torch.bool, device=sentence_states.device
            )

        batch_size, num_windows, _, sentence_dim = windows.shape
        projected = self.sentence_proj(windows)
        flat_points = projected.reshape(batch_size * num_windows, self.window_size, -1)
        flat_mask = windowed_mask.reshape(batch_size * num_windows, self.window_size)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        encoded = self.window_encoder(flat_points, safe_mask)
        encoded = encoded.reshape(batch_size, num_windows, -1)
        encoded = encoded.masked_fill(~valid_windows.unsqueeze(-1), 0.0)
        logger.debug("Encoded %d sliding sentence windows", num_windows)
        return encoded, valid_windows


class WindowGeometryEncoder(nn.Module):
    """Summarize inter-window distances and angles in linear time."""

    def __init__(self, window_dim: int = 512, centroid_dim: int = 128, hidden: int = 256) -> None:
        """Build a per-window geometry encoder relative to the global centroid."""
        super().__init__()
        self.centroid_mlp = nn.Linear(window_dim, centroid_dim)
        # Features: local centroid, displacement, distance, and cosine angle.
        pair_input_dim = centroid_dim * 2 + 2
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, window_dim),
        )

    def forward(
        self,
        window_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a global geometry state without constructing a window-pair matrix."""
        if window_states.ndim != 3:
            raise ValueError("window_states must have shape (batch, windows, dim)")
        batch_size, num_windows, _ = window_states.shape
        if num_windows == 0:
            raise ValueError("at least one window position is required")
        if mask is None:
            mask = torch.ones(
                batch_size, num_windows, dtype=torch.bool, device=window_states.device
            )
        elif mask.shape != (batch_size, num_windows):
            raise ValueError("window mask must have shape (batch, windows)")
        else:
            mask = mask.to(dtype=torch.bool, device=window_states.device)

        centroids = self.centroid_mlp(window_states)
        weights = mask.unsqueeze(-1).to(dtype=centroids.dtype)
        global_centroid = (centroids * weights).sum(dim=1, keepdim=True)
        global_centroid = global_centroid / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        displacement = centroids - global_centroid
        distance = displacement.norm(dim=-1, keepdim=True)
        angle = F.cosine_similarity(centroids, global_centroid.expand_as(centroids), dim=-1)
        features = torch.cat([centroids, displacement, distance, angle.unsqueeze(-1)], dim=-1)
        geometry = self.pair_mlp(features)
        geometry = geometry.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = geometry.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(geometry.dtype)
        logger.debug("Encoded inter-window geometry for %d windows", num_windows)
        return pooled


class ContextCloudLM(nn.Module):
    """Transformer-free language model combining sentence and window clouds."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        d_sentence: int = 512,
        d_window: int = 512,
        d_cloud_dim: int = 16,
        d_window_dim: int = 32,
        proj_hidden: Optional[int] = None,
        window_size: int = 5,
        window_stride: int = 2,
        hidden: int = 256,
        padding_idx: int = 0,
        max_tokens: int = 2048,
    ) -> None:
        """Create the two-level cloud model and next-token prediction head."""
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.sentence_encoder = SentenceBulutEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_cloud_dim=d_cloud_dim,
            d_cloud=d_sentence,
            proj_hidden=proj_hidden,
            hidden=hidden,
            max_tokens=max_tokens,
            padding_idx=padding_idx,
        )
        self.window_encoder = WindowBulutEncoder(
            sentence_dim=d_sentence,
            window_size=window_size,
            stride=window_stride,
            d_window_dim=d_window_dim,
            d_cloud=d_window,
            hidden=hidden,
        )
        self.window_geometry = WindowGeometryEncoder(window_dim=d_window, hidden=hidden)
        self.combined_head = nn.Sequential(
            nn.Linear(d_sentence + d_window, 1024),
            nn.GELU(),
            nn.Linear(1024, vocab_size),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        sentence_mask: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next-token logits from ``(B, S, T)`` or single-sentence ``(B, T)``."""
        single_sentence = tokens.ndim == 2
        if single_sentence:
            tokens = tokens.unsqueeze(1)
            if sentence_mask is not None:
                if sentence_mask.ndim == 2:
                    sentence_mask = sentence_mask.unsqueeze(1)
                elif sentence_mask.ndim != 3:
                    raise ValueError("sentence_mask must be (B,T), (B,S), or (B,S,T)")
        elif tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, sentences, tokens)")

        batch_size, num_sentences, num_tokens = tokens.shape
        if sentence_mask is None:
            token_mask = tokens.ne(self.padding_idx)
        elif sentence_mask.shape == tokens.shape:
            token_mask = sentence_mask.to(dtype=torch.bool, device=tokens.device)
        elif sentence_mask.shape == (batch_size, num_sentences):
            sentence_valid = sentence_mask.to(dtype=torch.bool, device=tokens.device)
            token_mask = tokens.ne(self.padding_idx) & sentence_valid.unsqueeze(-1)
        else:
            raise ValueError("sentence_mask must have shape (B,S) or (B,S,T)")

        flat_tokens = tokens.reshape(batch_size * num_sentences, num_tokens)
        flat_mask = token_mask.reshape(batch_size * num_sentences, num_tokens)
        sentence_states = self.sentence_encoder(flat_tokens, flat_mask).reshape(
            batch_size, num_sentences, -1
        )
        sentence_valid = token_mask.any(dim=-1)
        sentence_states = sentence_states.masked_fill(~sentence_valid.unsqueeze(-1), 0.0)
        sentence_weights = sentence_valid.unsqueeze(-1).to(sentence_states.dtype)
        sentence_state = (sentence_states * sentence_weights).sum(dim=1)
        sentence_state = sentence_state / sentence_weights.sum(dim=1).clamp_min(1.0)

        window_states, valid_windows = self.window_encoder(
            sentence_states, mask=sentence_valid, window_mask=window_mask
        )
        window_mean_state = window_states.sum(dim=1) / valid_windows.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(window_states.dtype)
        geometry_state = self.window_geometry(window_states, valid_windows)
        window_state = 0.5 * (window_mean_state + geometry_state)
        logits = self.combined_head(torch.cat([sentence_state, window_state], dim=-1))
        logger.debug("Produced next-token logits for batch size %d", batch_size)
        return logits

    def generate(
        self,
        tokens: torch.Tensor,
        max_new: int = 20,
        temperature: float = 0.8,
        eos_id: int = 1,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressively sample tokens from 2D or sentence-context 3D input.

        For 2D input, generated tokens are appended to the token sequence. For
        3D input, they are appended to the final sentence's token sequence.
        Generation stops when every batch item samples ``eos_id`` or when
        ``max_new`` tokens have been sampled.
        """
        if tokens.ndim not in (2, 3):
            raise ValueError("tokens must have shape (batch, tokens) or (batch, sentences, tokens)")
        if max_new < 0:
            raise ValueError("max_new cannot be negative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if tokens.ndim == 3 and tokens.size(1) == 0:
            raise ValueError("3D input must contain at least one sentence")

        self.eval()
        generated = tokens.clone()
        with torch.no_grad():
            for _ in range(max_new):
                logits = self(generated)

                if top_k and top_k < logits.size(-1):
                    top_indices = torch.argsort(logits, dim=-1, descending=True)[:, :top_k]
                    excluded = torch.ones_like(logits, dtype=torch.bool)
                    excluded.scatter_(1, top_indices, False)
                    logits = logits.masked_fill(excluded, float("-inf"))

                probabilities = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                if (next_token == eos_id).all():
                    break

                if generated.ndim == 2:
                    generated = torch.cat((generated, next_token), dim=1)
                else:
                    additions = generated.new_full(
                        (*generated.shape[:2], 1), self.padding_idx
                    )
                    additions[:, -1, 0] = next_token.squeeze(1)
                    generated = torch.cat((generated, additions), dim=2)

        return generated

    def load_purecloudlm_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        strict: bool = False,
    ) -> nn.modules.module._IncompatibleKeys:
        """Initialize the sentence encoder from a legacy PureCloudLM checkpoint."""
        return self.sentence_encoder.load_purecloudlm_checkpoint(checkpoint, strict=strict)


def sliding_window(tensor: torch.Tensor, size: int = 5, stride: int = 2) -> torch.Tensor:
    """Return zero-padded overlapping windows from ``(B, N, ...)`` in linear time."""
    if tensor.ndim < 2:
        raise ValueError("tensor must have batch and sequence dimensions")
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")
    sequence_length = tensor.size(1)
    if sequence_length == 0:
        raise ValueError("cannot create windows from an empty sequence")

    if sequence_length <= size:
        starts = [0]
    else:
        last_start = sequence_length - size
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start and starts[-1] + size < sequence_length:
            starts.append(last_start)
    windows = []
    for start in starts:
        window = tensor[:, start : start + size]
        missing = size - window.size(1)
        if missing:
            pad_shape = list(window.shape)
            pad_shape[1] = missing
            window = torch.cat([window, window.new_zeros(pad_shape)], dim=1)
        windows.append(window)
    return torch.stack(windows, dim=1)


class _PositionalEncoding(nn.Module):
    """Sine/cosine positions for sentence tokens, with PureCloudLM key names."""

    def __init__(self, d_model: int, max_len: int) -> None:
        """Precompute and register a non-trainable positional encoding table."""
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``(B, T, D)`` embeddings."""
        return inputs + self.pe[:, : inputs.size(1), :]
