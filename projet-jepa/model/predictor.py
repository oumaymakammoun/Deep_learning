"""
I-JEPA Predictor Network
=========================
A narrow transformer that predicts target patch representations from context
encoder outputs. This is the CORE of I-JEPA's predictive architecture.

KEY DESIGN CHOICES:
1. The predictor is NARROWER than the encoder (fewer params) to prevent
   trivial identity/copy solutions.
2. It receives context patch embeddings + positional tokens for target locations.
3. It outputs predicted representations at target positions.
4. Prediction is in LATENT SPACE — NOT pixel space (unlike MAE/BERT).

The predictor takes:
- Context encoder outputs (encoded visible patches)
- Positional embeddings of target (masked) patches
And produces:
- Predicted representations for target patches
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PredictorBlock(nn.Module):
    """Transformer block for the predictor (Pre-LN architecture)."""

    def __init__(self, embed_dim: int = 192, num_heads: int = 6,
                 mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Pre-LN -> MHSA -> residual -> Pre-LN -> MLP -> residual."""
        B, N, D = x.shape

        # Self-attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(out)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class Predictor(nn.Module):
    """
    I-JEPA Predictor: narrow transformer that predicts target representations.

    Architecture:
        1. Project context encoder outputs from encoder_dim -> predictor_dim
        2. Create learnable mask tokens at target positions
        3. Add positional embeddings to both context and target tokens
        4. Process through narrow transformer blocks
        5. Extract and project predictions at target positions back to encoder_dim

    WHY NARROW?
    The predictor must be narrower than the encoder to avoid learning a
    trivial copy/identity mapping. If the predictor had the same capacity as
    the encoder, it could simply memorize the target encoder's output without
    the context encoder learning useful representations.

    Args:
        num_patches: Total number of patches in the image
        encoder_embed_dim: Embedding dim of the context/target encoders
        predictor_embed_dim: Internal embedding dim of the predictor (NARROWER)
        depth: Number of transformer blocks in the predictor
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
    """

    def __init__(self, num_patches: int = 144, encoder_embed_dim: int = 384,
                 predictor_embed_dim: int = 192, depth: int = 3,
                 num_heads: int = 6, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        # --- Project FROM encoder space TO predictor space ---
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # --- Learnable mask token ---
        # This token is placed at target (masked) positions and learns to
        # aggregate information from context patches via attention.
        # Unlike MAE which uses a SINGLE shared mask token, I-JEPA uses
        # this token + positional embeddings to specify target locations.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # --- Positional embeddings for the predictor ---
        # These are SEPARATE from the encoder's positional embeddings.
        # They tell the predictor WHERE each target patch is located.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # --- Narrow transformer blocks ---
        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_embed_dim)

        # --- Project BACK from predictor space to encoder space ---
        # The output must match the target encoder's embedding dimension
        # so we can compute MSE loss between prediction and target.
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

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

    def forward(self, context_encodings: torch.Tensor,
                context_indices: torch.Tensor,
                target_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict target patch representations from context.

        This is the HEART of I-JEPA:
        - Takes encoded context patches and positional info about targets
        - Predicts what the target encoder would output at target positions
        - Prediction is in EMBEDDING SPACE (not pixel space!)

        Args:
            context_encodings: (B, N_ctx, encoder_embed_dim) — context encoder output
            context_indices: (N_ctx,) — patch indices of context patches
            target_indices: (N_tgt,) — patch indices of target patches

        Returns:
            predictions: (B, N_tgt, encoder_embed_dim) — predicted target reps
        """
        B = context_encodings.shape[0]
        N_ctx = context_indices.shape[0]
        N_tgt = target_indices.shape[0]

        # Step 1: Project context encodings to predictor dimension
        # (B, N_ctx, encoder_dim) -> (B, N_ctx, predictor_dim)
        ctx_tokens = self.input_proj(context_encodings)

        # Step 2: Add positional embeddings to context tokens
        ctx_pos = self.pos_embed[:, context_indices, :]  # (1, N_ctx, pred_dim)
        ctx_tokens = ctx_tokens + ctx_pos

        # Step 3: Create mask tokens for target positions
        # These start as the learned mask_token + positional embedding
        # The transformer will attend context->target to fill them in
        tgt_tokens = self.mask_token.expand(B, N_tgt, -1)  # (B, N_tgt, pred_dim)
        tgt_pos = self.pos_embed[:, target_indices, :]      # (1, N_tgt, pred_dim)
        tgt_tokens = tgt_tokens + tgt_pos

        # Step 4: Concatenate context + target tokens
        # The predictor sees ALL tokens (context + mask) and attends freely
        # Context tokens carry information; target tokens carry position
        x = torch.cat([ctx_tokens, tgt_tokens], dim=1)  # (B, N_ctx+N_tgt, pred_dim)

        # Step 5: Process through narrow transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Step 6: Extract ONLY the target token predictions
        # We only need predictions at masked positions
        target_predictions = x[:, N_ctx:, :]  # (B, N_tgt, predictor_dim)

        # Step 7: Project back to encoder embedding space
        # (B, N_tgt, predictor_dim) -> (B, N_tgt, encoder_dim)
        target_predictions = self.output_proj(target_predictions)

        return target_predictions
