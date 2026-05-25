"""
I-JEPA Training Loop
=====================
Complete training pipeline for I-JEPA self-supervised pretraining.

Training procedure:
1. Sample a batch of images
2. Generate multi-block masks (context + target regions)
3. Context encoder processes ONLY visible (context) patches
4. Target encoder (EMA, no grad) processes ALL patches → target representations
5. Predictor predicts target representations from context encodings
6. Loss = MSE between predicted and target representations (LATENT SPACE)
7. Update context encoder + predictor with gradients
8. Update target encoder via EMA (NO gradients)

This is fundamentally different from MAE:
- MAE loss = MSE in PIXEL space (reconstruct pixels)
- I-JEPA loss = MSE in LATENT space (predict representations)
- I-JEPA never reconstructs pixels → learns semantic features
"""

import os
import sys
import math
import time
import random
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from model.encoder import VisionTransformerEncoder
from model.predictor import Predictor
from model.target_encoder import TargetEncoder
from utils.masking import generate_batch_masks
from utils.ema import ema_update, cosine_momentum_schedule
from data.dataset import get_dataset, get_dataloader


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cosine_lr_schedule(
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    - Epochs [0, warmup_epochs): linear warmup from 0 to base_lr
    - Epochs [warmup_epochs, total_epochs): cosine decay from base_lr to min_lr

    Args:
        optimizer: PyTorch optimizer
        current_epoch: Current epoch number
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        base_lr: Peak learning rate
        min_lr: Minimum learning rate

    Returns:
        Current learning rate
    """
    if current_epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (current_epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_one_epoch(
    context_encoder: VisionTransformerEncoder,
    predictor: Predictor,
    target_encoder: TargetEncoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    total_steps: int,
    global_step: int,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        (average_loss, updated_global_step)
    """
    context_encoder.train()
    predictor.train()
    # Target encoder is always in eval mode (no dropout, etc.)
    target_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    grid_size = context_encoder.grid_size
    mask_cfg = config['masking']

    for batch_idx, batch_data in enumerate(dataloader):
        # Handle both (images, labels) and (images,) formats
        if isinstance(batch_data, (list, tuple)):
            images = batch_data[0]
        else:
            images = batch_data

        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        # --- Step 1: Generate multi-block masks ---
        masks = generate_batch_masks(
            batch_size=B,
            grid_size=grid_size,
            num_targets=mask_cfg['num_targets'],
            target_scale_min=mask_cfg['target_scale_min'],
            target_scale_max=mask_cfg['target_scale_max'],
            target_aspect_ratio_min=mask_cfg['target_aspect_ratio_min'],
            target_aspect_ratio_max=mask_cfg['target_aspect_ratio_max'],
            context_scale_min=mask_cfg['context_scale_min'],
            context_scale_max=mask_cfg['context_scale_max'],
        )

        context_mask = masks['context_mask'].to(device)
        target_indices = masks['target_indices'].to(device)
        context_indices = masks['context_indices'].to(device)

        # --- Step 2: Forward pass with mixed precision ---
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=config['training']['use_amp']):
            # Context encoder: encode ONLY visible patches
            # This is where I-JEPA differs: encoder sees incomplete input
            context_output, _ = context_encoder(
                images, context_mask=context_mask
            )

            # Predictor: predict target representations from context
            # The predictor receives:
            #   - context_output: encoded visible patches
            #   - context_indices: WHERE the visible patches are
            #   - target_indices: WHERE the target patches are (positional info)
            # It outputs PREDICTED representations at target positions
            predictions = predictor(
                context_output, context_indices, target_indices
            )

            # Target encoder: get ground-truth representations (NO GRAD!)
            # Unlike MAE where targets are raw pixels, I-JEPA targets are
            # LEARNED representations from the EMA encoder.
            with torch.no_grad():
                target_output = target_encoder(images, target_indices)

            # --- Step 3: Compute loss in LATENT SPACE ---
            # MSE between predicted and target representations
            # This is THE key difference from MAE/BERT:
            # - MAE: loss = MSE(predicted_pixels, actual_pixels)
            # - I-JEPA: loss = MSE(predicted_embeddings, target_embeddings)
            # Operating in latent space forces semantic prediction
            loss = nn.functional.mse_loss(predictions, target_output)

        # --- Step 4: Backward pass with gradient scaling ---
        scaler.scale(loss).backward()

        # Gradient clipping for stability
        if config['training']['grad_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(context_encoder.parameters()) + list(predictor.parameters()),
                config['training']['grad_clip']
            )

        scaler.step(optimizer)
        scaler.update()

        # --- Step 5: EMA update of target encoder ---
        # The target encoder is NEVER trained with gradients.
        # It slowly tracks the context encoder via momentum.
        momentum = cosine_momentum_schedule(
            base_momentum=config['ema']['momentum_base'],
            final_momentum=config['ema']['momentum_final'],
            current_step=global_step,
            total_steps=total_steps,
        )
        ema_update(context_encoder, target_encoder.encoder, momentum)

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Logging
        if batch_idx % config['training']['log_interval'] == 0:
            print(f"  [Epoch {epoch}][Step {batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Momentum: {momentum:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Ctx: {context_indices.shape[0]} | "
                  f"Tgt: {target_indices.shape[0]}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def save_checkpoint(
    context_encoder: VisionTransformerEncoder,
    predictor: Predictor,
    target_encoder: TargetEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    path: str,
) -> None:
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'context_encoder': context_encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': config,
    }, path)
    print(f"  [SAVE] Checkpoint saved to {path}")


def main() -> None:
    """Main I-JEPA training entry point."""
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")
    if device.type == 'cuda':
        print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")

    # --- Build models ---
    enc_cfg = config['encoder']
    context_encoder = VisionTransformerEncoder(
        image_size=enc_cfg['image_size'],
        patch_size=enc_cfg['patch_size'],
        in_channels=enc_cfg['in_channels'],
        embed_dim=enc_cfg['embed_dim'],
        depth=enc_cfg['depth'],
        num_heads=enc_cfg['num_heads'],
        mlp_ratio=enc_cfg['mlp_ratio'],
        dropout=enc_cfg['dropout'],
    ).to(device)

    pred_cfg = config['predictor']
    predictor = Predictor(
        num_patches=context_encoder.num_patches,
        encoder_embed_dim=enc_cfg['embed_dim'],
        predictor_embed_dim=pred_cfg['predictor_embed_dim'],
        depth=pred_cfg['depth'],
        num_heads=pred_cfg['num_heads'],
        mlp_ratio=pred_cfg['mlp_ratio'],
    ).to(device)

    target_encoder = TargetEncoder(context_encoder).to(device)

    # Print model sizes
    enc_params = sum(p.numel() for p in context_encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"[INIT] Context Encoder: {enc_params / 1e6:.1f}M params")
    print(f"[INIT] Predictor: {pred_params / 1e6:.1f}M params")
    print(f"[INIT] Grid size: {context_encoder.grid_size}x{context_encoder.grid_size} "
          f"= {context_encoder.num_patches} patches")

    # --- Dataset ---
    data_cfg = config['data']
    train_dataset = get_dataset(
        dataset_name=data_cfg['dataset'],
        data_dir=data_cfg['data_dir'],
        image_size=data_cfg['image_size'],
        is_train=True,
    )
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
    )

    # --- Optimizer ---
    # Only the context encoder and predictor are optimized.
    # The target encoder has NO learnable parameters (EMA only).
    train_cfg = config['training']
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.999),
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=train_cfg['use_amp'])

    # Training setup
    total_steps = train_cfg['epochs'] * len(train_loader)
    global_step = 0
    loss_history = []

    print(f"\n[TRAIN] Starting I-JEPA pretraining for {train_cfg['epochs']} epochs")
    print(f"[TRAIN] Total steps: {total_steps}")
    print(f"[TRAIN] Batch size: {train_cfg['batch_size']}")
    print("=" * 60)

    for epoch in range(train_cfg['epochs']):
        epoch_start = time.time()

        # Update learning rate
        lr = cosine_lr_schedule(
            optimizer, epoch, train_cfg['epochs'],
            train_cfg['warmup_epochs'], train_cfg['lr'], train_cfg['min_lr']
        )

        # Train one epoch
        avg_loss, global_step = train_one_epoch(
            context_encoder, predictor, target_encoder,
            train_loader, optimizer, scaler,
            device, epoch, config, total_steps, global_step,
        )

        elapsed = time.time() - epoch_start
        loss_history.append(avg_loss)

        print(f"[Epoch {epoch}/{train_cfg['epochs']}] "
              f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % train_cfg['save_interval'] == 0 or \
           epoch == train_cfg['epochs'] - 1:
            ckpt_path = os.path.join(
                train_cfg['checkpoint_dir'],
                f"ijepa_epoch{epoch + 1}.pt"
            )
            save_checkpoint(
                context_encoder, predictor, target_encoder,
                optimizer, scaler, epoch, avg_loss, config, ckpt_path,
            )

    # Save loss history for visualization
    os.makedirs(train_cfg['checkpoint_dir'], exist_ok=True)
    np.save(os.path.join(train_cfg['checkpoint_dir'], "loss_history.npy"),
            np.array(loss_history))

    print("\n" + "=" * 60)
    print("[DONE] I-JEPA pretraining complete!")
    print(f"[DONE] Final loss: {loss_history[-1]:.4f}")
    print(f"[DONE] Checkpoints saved in: {train_cfg['checkpoint_dir']}")


if __name__ == "__main__":
    main()
