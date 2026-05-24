"""
Exponential Moving Average (EMA) Utilities for I-JEPA
======================================================
The target encoder in I-JEPA is updated via EMA of the context encoder.

EMA update rule:
    θ_target = m * θ_target + (1 - m) * θ_context

where m is the momentum coefficient (typically 0.996 → 1.0 with cosine schedule).

WHY EMA?
- Prevents representation collapse (a known failure mode in self-supervised learning)
- Creates a slowly-evolving target that provides stable learning signal
- Same principle used in BYOL, MoCo v1/v2, DINO, and I-JEPA
- Without EMA (or stop-gradient), both encoders would converge to outputting
  a constant vector regardless of input → useless representations
"""

import math
from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update(
    online_model: nn.Module,
    target_model: nn.Module,
    momentum: float,
) -> None:
    """
    Update target model parameters via Exponential Moving Average.

    For each parameter pair (θ_online, θ_target):
        θ_target = momentum * θ_target + (1 - momentum) * θ_online

    High momentum (e.g., 0.999) means the target changes very slowly.
    momentum=1.0 means no update (target frozen).
    momentum=0.0 means direct copy (no momentum).

    Args:
        online_model: The context encoder (trained with gradients)
        target_model: The target encoder (updated via EMA, no gradients)
        momentum: EMA momentum coefficient [0, 1]
    """
    for online_params, target_params in zip(
        online_model.parameters(), target_model.parameters()
    ):
        # In-place EMA update
        target_params.data.mul_(momentum).add_(
            online_params.data, alpha=1.0 - momentum
        )


def cosine_momentum_schedule(
    base_momentum: float,
    final_momentum: float,
    current_step: int,
    total_steps: int,
) -> float:
    """
    Compute EMA momentum with cosine annealing schedule.

    The momentum increases from base_momentum to final_momentum following
    a cosine curve. This means:
    - Early training: lower momentum → target encoder updates faster
    - Late training: higher momentum → target encoder becomes more stable

    This schedule is crucial for I-JEPA training stability:
    - Too low momentum early on → target changes too fast → unstable
    - Too high momentum early on → target doesn't track online model → slow learning

    m(t) = final_m - (final_m - base_m) * (cos(π * t / T) + 1) / 2

    Args:
        base_momentum: Starting momentum (e.g., 0.996)
        final_momentum: Final momentum (e.g., 1.0)
        current_step: Current training step
        total_steps: Total number of training steps

    Returns:
        Current momentum value
    """
    if total_steps <= 0:
        return final_momentum

    # Cosine annealing from base to final
    progress = min(current_step / total_steps, 1.0)
    momentum = final_momentum - (final_momentum - base_momentum) * (
        (math.cos(math.pi * progress) + 1.0) / 2.0
    )
    return momentum
