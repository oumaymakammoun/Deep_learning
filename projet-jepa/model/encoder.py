"""
Vision Transformer (ViT) Context Encoder for I-JEPA
====================================================
Encodes visible (unmasked) context patches into latent representations.

CRITICAL DIFFERENCE FROM MAE:
- MAE decoder reconstructs PIXELS. I-JEPA predicts LATENT representations.
- This forces the model to learn semantic features, not low-level textures.

Reference: Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture", CVPR 2023.
"""

import math
import numpy as np
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Converts image into patch embeddings via convolution projection."""

    def __init__(self, image_size: int = 96, patch_size: int = 8,
                 in_channels: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, embed_dim)"""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with scaled dot-product."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attention_weights)."""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x, attn_weights


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN -> MHSA -> residual -> LN -> MLP -> residual."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass; optionally returns attention weights."""
        attn_out, attn_w = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w if return_attention else None


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings (grid_size^2, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    half = embed_dim // 2
    omega = np.arange(half // 2, dtype=np.float64)
    omega = 1.0 / (10000.0 ** (2.0 * omega / half))

    pos_h = grid[1].reshape(-1)
    pos_w = grid[0].reshape(-1)
    out_h = np.einsum('m,d->md', pos_h, omega)
    out_w = np.einsum('m,d->md', pos_w, omega)
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)
    return np.concatenate([emb_h, emb_w], axis=1)


class VisionTransformerEncoder(nn.Module):
    """
    ViT Context Encoder for I-JEPA.

    Processes ONLY visible (context) patches. NO [CLS] token, NO classification
    head — this is a pure self-supervised backbone.

    I-JEPA vs MAE: both mask patches, but MAE reconstructs pixels while
    I-JEPA predicts latent representations from the target encoder.
    """

    def __init__(self, image_size: int = 96, patch_size: int = 8,
                 in_channels: int = 3, embed_dim: int = 384, depth: int = 6,
                 num_heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size

        self.patch_embed = PatchEmbedding(image_size, patch_size,
                                          in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Init positional embeddings with sinusoidal pattern
        pos = _get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: (B, C, H, W) input images
            context_mask: (B, num_patches) bool — True = keep patch
            return_attention: whether to collect attention maps
        Returns:
            (encoded_patches, attention_maps_or_None)
        """
        x = self.patch_embed(x) + self.pos_embed

        # --- I-JEPA key step: keep ONLY context patches ---
        if context_mask is not None:
            ctx_idx = context_mask[0].nonzero(as_tuple=False).squeeze(-1)
            x = x[:, ctx_idx, :]

        attn_maps: List[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, return_attention=return_attention)
            if attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)
        return x, attn_maps if return_attention else None

    def get_pos_embed(self, patch_indices: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """Get positional embeddings for specific patch positions."""
        if patch_indices is None:
            return self.pos_embed
        return self.pos_embed[:, patch_indices, :]
