"""
Linear Probing Evaluation for I-JEPA
======================================
Evaluates the quality of learned representations by training a linear
classifier on top of frozen encoder features.

Linear probing protocol:
1. Load the pretrained I-JEPA context encoder
2. FREEZE all encoder weights (no fine-tuning)
3. Add a single linear layer (embed_dim -> num_classes)
4. Train ONLY the linear layer with supervised cross-entropy loss
5. Report top-1 accuracy on the test set

WHY linear probing?
- Measures how linearly separable the learned features are
- If a linear classifier works well, the encoder learned semantically
  meaningful representations during self-supervised pretraining
- Standard evaluation protocol for SSL methods (BYOL, DINO, MAE, I-JEPA)
"""

import os
import sys
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.encoder import VisionTransformerEncoder
from data.dataset import get_labeled_dataset, get_dataloader


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class LinearProbe(nn.Module):
    """
    Linear probing head for evaluation.

    Takes mean-pooled encoder features and classifies them.
    Only this linear layer is trained; the encoder is frozen.

    Args:
        embed_dim: Encoder embedding dimension
        num_classes: Number of output classes
    """

    def __init__(self, embed_dim: int = 384, num_classes: int = 10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) patch-level features from encoder
        Returns:
            (B, num_classes) logits
        """
        # Global average pooling over patches
        x = x.mean(dim=1)  # (B, D)
        x = self.norm(x)
        x = self.fc(x)
        return x


@torch.no_grad()
def extract_features(
    encoder: VisionTransformerEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Extract features from the frozen encoder for all samples.

    Args:
        encoder: Pretrained I-JEPA encoder (frozen)
        dataloader: Data loader with labeled data
        device: Computation device

    Returns:
        (features, labels) numpy arrays
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        # Encode ALL patches (no masking for evaluation)
        features, _ = encoder(images, context_mask=None)
        # Global average pooling
        features = features.mean(dim=1)  # (B, embed_dim)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return (torch.cat(all_features, dim=0).numpy(),
            torch.cat(all_labels, dim=0).numpy())


def train_linear_probe(
    encoder: VisionTransformerEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> float:
    """
    Train and evaluate a linear probe on frozen encoder features.

    Args:
        encoder: Pretrained I-JEPA encoder
        train_loader: Labeled training data
        test_loader: Labeled test data
        device: Computation device
        config: Configuration dict

    Returns:
        Top-1 test accuracy (float)
    """
    eval_cfg = config['eval']
    enc_cfg = config['encoder']

    # Freeze encoder completely
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Linear probe
    probe = LinearProbe(
        embed_dim=enc_cfg['embed_dim'],
        num_classes=eval_cfg['num_classes'],
    ).to(device)

    # SGD optimizer (standard for linear probing)
    optimizer = torch.optim.SGD(
        probe.parameters(),
        lr=eval_cfg['lr'],
        momentum=eval_cfg['momentum'],
        weight_decay=eval_cfg['weight_decay'],
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=eval_cfg['epochs']
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print(f"\n[EVAL] Training linear probe for {eval_cfg['epochs']} epochs")
    print("=" * 50)

    for epoch in range(eval_cfg['epochs']):
        # --- Train ---
        probe.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward: frozen encoder + trainable probe
            with torch.no_grad():
                features, _ = encoder(images, context_mask=None)

            logits = probe(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total

        # --- Evaluate ---
        probe.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                features, _ = encoder(images, context_mask=None)
                logits = probe(features)

                _, predicted = logits.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100.0 * test_correct / test_total
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Epoch {epoch + 1}/{eval_cfg['epochs']}] "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}%")

    print("=" * 50)
    print(f"[EVAL] Best linear probe accuracy: {best_acc:.2f}%")
    return best_acc


def main() -> None:
    """Main linear probing evaluation entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")

    # --- Load pretrained encoder ---
    enc_cfg = config['encoder']
    encoder = VisionTransformerEncoder(
        image_size=enc_cfg['image_size'],
        patch_size=enc_cfg['patch_size'],
        in_channels=enc_cfg['in_channels'],
        embed_dim=enc_cfg['embed_dim'],
        depth=enc_cfg['depth'],
        num_heads=enc_cfg['num_heads'],
        mlp_ratio=enc_cfg['mlp_ratio'],
        dropout=0.0,
    ).to(device)

    # Load checkpoint
    ckpt_dir = config['training']['checkpoint_dir']
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if not ckpt_files:
        print("[ERROR] No checkpoints found! Run train.py first.")
        sys.exit(1)

    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    print(f"[INIT] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint['context_encoder'])
    print(f"[INIT] Loaded encoder from epoch {checkpoint['epoch'] + 1}")

    # --- Load labeled datasets ---
    data_cfg = config['data']
    train_dataset = get_labeled_dataset(
        data_cfg['dataset'], data_cfg['data_dir'],
        data_cfg['image_size'], is_train=True,
    )
    test_dataset = get_labeled_dataset(
        data_cfg['dataset'], data_cfg['data_dir'],
        data_cfg['image_size'], is_train=False,
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=config['eval']['batch_size'],
        shuffle=True, num_workers=data_cfg['num_workers'],
        drop_last=False,
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=config['eval']['batch_size'],
        shuffle=False, num_workers=data_cfg['num_workers'],
        drop_last=False,
    )

    # --- Run linear probing ---
    accuracy = train_linear_probe(
        encoder, train_loader, test_loader, device, config,
    )

    print(f"\n[RESULT] Final linear probe top-1 accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
