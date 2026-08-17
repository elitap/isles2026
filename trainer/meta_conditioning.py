"""
MetaConditioner: encodes (days_norm, chronicity, site_idx) and injects the
conditioning vector at the nnUNet bottleneck via channel concatenation
followed by a Conv1×1×1 projection back to the original channel width.

Architecture:
    x  →  stage(x)  →  [B, C, D, H, W]
    meta_vec = MetaEncoder(days, chronicity, site_idx)   [B, C_meta]
    expand meta_vec  →  [B, C_meta, D, H, W]
    cat([x, meta_exp], dim=1)  →  Conv1×1×1  →  [B, C, D, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MetaEncoder(nn.Module):
    """Maps (days_norm, chronicity, site_idx) → conditioning vector [B, c_meta]."""

    def __init__(
        self,
        n_sites: int,
        site_embed_dim: int = 8,
        c_meta: int = 16,
        n_chronicity: int = 4,
    ) -> None:
        super().__init__()
        self.n_chronicity = n_chronicity
        self.site_embed = nn.Embedding(n_sites, site_embed_dim)
        in_dim = 1 + n_chronicity + site_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, c_meta),
            nn.ReLU(inplace=True),
            nn.Linear(c_meta, c_meta),
        )

    def forward(self, days: Tensor, chronicity: Tensor, site_idx: Tensor) -> Tensor:
        """
        days:       [B] float  — sentinel –1 or min-max normalised [0, 1]
        chronicity: [B] int64  — category index (0=unknown, 1=chronic, ...)
        site_idx:   [B] int64  — site index in sorted vocab
        Returns:    [B, c_meta]
        """
        days_f = days.float().unsqueeze(1)  # [B, 1]
        chron_oh = torch.zeros(
            days.size(0), self.n_chronicity, dtype=torch.float32, device=days.device
        )
        chron_idx = chronicity.long().clamp(0, self.n_chronicity - 1).unsqueeze(1)
        chron_oh.scatter_(1, chron_idx, 1.0)
        site_emb = self.site_embed(site_idx.long())  # [B, site_embed_dim]
        return self.mlp(torch.cat([days_f, chron_oh, site_emb], dim=1))


class MetaConditionedStage(nn.Module):
    """
    Drop-in replacement for one nnUNet encoder stage (the bottleneck).

    The trainer calls set_meta() before each forward pass and clear_meta()
    after.  When no metadata is pending (e.g. at inference without sidecars)
    the module behaves identically to the wrapped stage — no side effects.
    """

    def __init__(
        self,
        stage: nn.Module,
        bottleneck_channels: int,
        n_sites: int,
        site_embed_dim: int = 8,
        c_meta: int = 16,
        n_chronicity: int = 4,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.meta_encoder = MetaEncoder(n_sites, site_embed_dim, c_meta, n_chronicity)
        self.proj = nn.Conv3d(bottleneck_channels + c_meta, bottleneck_channels, kernel_size=1)
        self._meta: dict[str, Tensor] | None = None

    def set_meta(self, days: Tensor, chronicity: Tensor, site_idx: Tensor) -> None:
        """Called by trainer before each forward pass."""
        self._meta = {"days": days, "chronicity": chronicity, "site_idx": site_idx}

    def clear_meta(self) -> None:
        self._meta = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage(x)
        if self._meta is None:
            return x
        meta_vec = self.meta_encoder(
            self._meta["days"].to(x.device),
            self._meta["chronicity"].to(x.device),
            self._meta["site_idx"].to(x.device),
        )  # [B, c_meta]
        n_spatial = x.ndim - 2
        meta_exp = meta_vec.view(meta_vec.size(0), -1, *([1] * n_spatial))
        meta_exp = meta_exp.expand(-1, -1, *x.shape[2:])
        return self.proj(torch.cat([x, meta_exp], dim=1))
