"""
I-JEPA Target Encoder (EMA-updated)
====================================
The target encoder is a MOMENTUM-UPDATED copy of the context encoder.
It produces the ground-truth representations that the predictor tries to match.

KEY DESIGN PRINCIPLES:
1. NO GRADIENTS flow through the target encoder (stop-gradient)
2. Weights are updated via Exponential Moving Average (EMA) of the context encoder
3. This asymmetry prevents representation collapse (from BYOL/DINO literature)
4. The target encoder sees ALL patches (including masked ones) to produce targets

WHY EMA instead of gradients?
- If both encoders were trained with gradients, the model could collapse
  to a trivial constant representation (all outputs = same vector).
- The EMA creates a slowly-evolving target that stabilizes training.
- This is the same insight from BYOL, MoCo, and DINO.
"""

import copy
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from model.encoder import VisionTransformerEncoder


class TargetEncoder(nn.Module):
    """
    EMA Target Encoder for I-JEPA.

    This is a copy of the context encoder whose weights are updated via
    Exponential Moving Average. It processes ALL image patches (no masking)
    to produce the ground-truth target representations.

    The predictor's job is to match its output to this encoder's output
    at the masked (target) positions.

    Unlike MAE/BERT:
    - The target is a LEARNED representation, not raw pixels/tokens.
    - This means the target itself improves over training, leading to
      increasingly abstract and semantic prediction targets.

    Args:
        context_encoder: The context encoder to copy architecture from
    """

    def __init__(self, context_encoder: VisionTransformerEncoder) -> None:
        super().__init__()

        # Deep copy the context encoder architecture and weights
        self.encoder = copy.deepcopy(context_encoder)

        # CRITICAL: Disable gradients for the target encoder
        # The target encoder is NEVER trained with backprop
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor,
                target_indices: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Encode ALL patches and optionally return only target positions.

        Unlike the context encoder, the target encoder processes ALL patches
        (no masking). This gives it a complete view of the image, producing
        representations that capture the full context.

        Args:
            x: (B, C, H, W) input images
            target_indices: (N_tgt,) indices of target patches to return.
                          If None, return all patch representations.

        Returns:
            (B, N_tgt, embed_dim) target representations
        """
        # Process ALL patches (no context_mask = see everything)
        all_patch_repr, _ = self.encoder(x, context_mask=None,
                                         return_attention=False)
        # all_patch_repr: (B, num_patches, embed_dim)

        if target_indices is not None:
            # Return only representations at target (masked) positions
            return all_patch_repr[:, target_indices, :]

        return all_patch_repr
