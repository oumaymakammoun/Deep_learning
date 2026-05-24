"""
Multi-Block Masking Strategy for I-JEPA
========================================
Implements the masking strategy from the I-JEPA paper:
- Multiple small TARGET blocks are sampled (what we predict)
- A large CONTEXT region is formed by removing targets from all patches

I-JEPA masking differs from MAE masking:
- MAE: Random independent patches are masked uniformly
- I-JEPA: Contiguous BLOCKS are masked, encouraging spatial reasoning
- I-JEPA masks are semantically meaningful regions, not random noise

The multi-block strategy samples 4 target blocks, each covering 15-20%
of the image area, with varying aspect ratios.
"""

import math
import random
from typing import Tuple, List, Dict

import torch


def sample_block_mask(
    grid_size: int,
    scale_min: float,
    scale_max: float,
    aspect_ratio_min: float,
    aspect_ratio_max: float,
) -> Tuple[List[int], Tuple[int, int, int, int]]:
    """
    Sample a rectangular block mask from the patch grid.

    Randomly selects a rectangular region of the patch grid with:
    - Area between scale_min and scale_max (fraction of total patches)
    - Aspect ratio between aspect_ratio_min and aspect_ratio_max

    Args:
        grid_size: Size of the patch grid (e.g., 12 for 12x12 = 144 patches)
        scale_min: Minimum fraction of total patches in the block
        scale_max: Maximum fraction of total patches in the block
        aspect_ratio_min: Minimum aspect ratio (width/height)
        aspect_ratio_max: Maximum aspect ratio (width/height)

    Returns:
        Tuple of:
            - List of patch indices inside the block
            - (top, left, height, width) block coordinates
    """
    num_patches = grid_size * grid_size

    # Sample target area (as fraction of total patches)
    target_area = random.uniform(scale_min, scale_max) * num_patches

    # Sample aspect ratio
    log_ratio_min = math.log(aspect_ratio_min)
    log_ratio_max = math.log(aspect_ratio_max)
    aspect_ratio = math.exp(random.uniform(log_ratio_min, log_ratio_max))

    # Compute block dimensions
    h = int(round(math.sqrt(target_area / aspect_ratio)))
    w = int(round(math.sqrt(target_area * aspect_ratio)))

    # Clamp to grid bounds
    h = max(1, min(h, grid_size))
    w = max(1, min(w, grid_size))

    # Random top-left corner
    top = random.randint(0, grid_size - h)
    left = random.randint(0, grid_size - w)

    # Collect patch indices inside this block
    indices = []
    for row in range(top, top + h):
        for col in range(left, left + w):
            indices.append(row * grid_size + col)

    return indices, (top, left, h, w)


def generate_masks(
    grid_size: int,
    num_targets: int = 4,
    target_scale_min: float = 0.15,
    target_scale_max: float = 0.2,
    target_aspect_ratio_min: float = 0.75,
    target_aspect_ratio_max: float = 1.5,
    context_scale_min: float = 0.85,
    context_scale_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate I-JEPA multi-block masks for one image.

    Strategy:
    1. Sample `num_targets` rectangular target blocks
    2. Union all target block indices -> these are the patches to predict
    3. Context = all patches NOT in any target block
    4. Optionally subsample context to context_scale fraction

    This produces:
    - context_mask: (num_patches,) bool — True = visible to context encoder
    - target_indices: (N_tgt,) long — indices of target patches
    - context_indices: (N_ctx,) long — indices of context patches

    Args:
        grid_size: Patch grid size (e.g., 12)
        num_targets: Number of target blocks to sample
        target_scale_min: Min area fraction per target block
        target_scale_max: Max area fraction per target block
        target_aspect_ratio_min: Min aspect ratio of target blocks
        target_aspect_ratio_max: Max aspect ratio of target blocks
        context_scale_min: Min fraction of non-target patches to keep
        context_scale_max: Max fraction of non-target patches to keep

    Returns:
        Dict with keys: 'context_mask', 'target_indices', 'context_indices'
    """
    num_patches = grid_size * grid_size
    all_target_indices = set()

    # Step 1: Sample multiple target blocks
    for _ in range(num_targets):
        indices, _ = sample_block_mask(
            grid_size, target_scale_min, target_scale_max,
            target_aspect_ratio_min, target_aspect_ratio_max,
        )
        all_target_indices.update(indices)

    # Step 2: Context = everything NOT in target blocks
    all_indices = set(range(num_patches))
    context_candidates = sorted(all_indices - all_target_indices)

    # Step 3: Optionally subsample context patches
    context_scale = random.uniform(context_scale_min, context_scale_max)
    n_context = max(1, int(len(context_candidates) * context_scale))
    if n_context < len(context_candidates):
        context_candidates = sorted(random.sample(context_candidates, n_context))

    # Step 4: Build output tensors
    target_indices = torch.tensor(sorted(all_target_indices), dtype=torch.long)
    context_indices = torch.tensor(context_candidates, dtype=torch.long)

    # Context mask: True for visible patches
    context_mask = torch.zeros(num_patches, dtype=torch.bool)
    context_mask[context_indices] = True

    return {
        'context_mask': context_mask,
        'target_indices': target_indices,
        'context_indices': context_indices,
    }


def generate_batch_masks(
    batch_size: int,
    grid_size: int,
    num_targets: int = 4,
    target_scale_min: float = 0.15,
    target_scale_max: float = 0.2,
    target_aspect_ratio_min: float = 0.75,
    target_aspect_ratio_max: float = 1.5,
    context_scale_min: float = 0.85,
    context_scale_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate a SHARED mask for the entire batch.

    For training efficiency, all images in a batch share the same mask
    pattern. This allows us to use simple tensor indexing instead of
    per-sample gather operations.

    Args:
        batch_size: Number of images in the batch
        grid_size: Patch grid size
        (other args same as generate_masks)

    Returns:
        Dict with:
            'context_mask': (B, num_patches) bool
            'target_indices': (N_tgt,) long
            'context_indices': (N_ctx,) long
    """
    masks = generate_masks(
        grid_size, num_targets,
        target_scale_min, target_scale_max,
        target_aspect_ratio_min, target_aspect_ratio_max,
        context_scale_min, context_scale_max,
    )

    # Repeat the same mask for all images in batch
    masks['context_mask'] = masks['context_mask'].unsqueeze(0).expand(
        batch_size, -1
    )

    return masks
