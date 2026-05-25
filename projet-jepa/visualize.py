"""
I-JEPA Visualization Tools
============================
Produces three types of visualizations:
1. t-SNE plot of learned representations (shows cluster quality)
2. Attention map visualization (shows what the encoder focuses on)
3. Training loss curve (shows convergence)

These plots help assess whether I-JEPA learned semantically meaningful
representations without ever reconstructing pixels.
"""

import os
import sys
from typing import Dict, Any, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from model.encoder import VisionTransformerEncoder
from data.dataset import get_labeled_dataset, get_dataloader


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ===================================================================
# 1. t-SNE VISUALIZATION
# ===================================================================

def plot_tsne(
    encoder: VisionTransformerEncoder,
    dataloader: DataLoader,
    device: torch.device,
    output_path: str,
    n_samples: int = 2000,
    perplexity: int = 30,
    class_names: Optional[list] = None,
) -> None:
    """
    Generate t-SNE visualization of learned representations.

    Extracts features from the frozen encoder, reduces to 2D with t-SNE,
    and plots colored by class label. Good representations should show
    distinct, well-separated clusters.

    Args:
        encoder: Pretrained I-JEPA encoder
        dataloader: Labeled data loader
        device: Computation device
        output_path: Where to save the plot
        n_samples: Number of samples to visualize
        perplexity: t-SNE perplexity parameter
        class_names: Optional list of class names for legend
    """
    from sklearn.manifold import TSNE

    print("[VIS] Extracting features for t-SNE...")
    encoder.eval()
    features_list = []
    labels_list = []
    count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if count >= n_samples:
                break
            images = images.to(device)
            feats, _ = encoder(images, context_mask=None)
            feats = feats.mean(dim=1)  # Global average pooling
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            count += images.shape[0]

    features = np.concatenate(features_list, axis=0)[:n_samples]
    labels = np.concatenate(labels_list, axis=0)[:n_samples]

    print(f"[VIS] Running t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    num_classes = len(np.unique(labels))
    cmap = plt.cm.get_cmap('tab10', num_classes)

    for c in range(num_classes):
        mask = labels == c
        label = class_names[c] if class_names else f"Class {c}"
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[cmap(c)], label=label, alpha=0.6, s=15,
                   edgecolors='none')

    ax.set_title("I-JEPA Learned Representations (t-SNE)", fontsize=16,
                 fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(loc='best', fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] t-SNE plot saved to {output_path}")


# ===================================================================
# 2. ATTENTION MAP VISUALIZATION
# ===================================================================

def plot_attention_maps(
    encoder: VisionTransformerEncoder,
    dataloader: DataLoader,
    device: torch.device,
    output_path: str,
    n_images: int = 8,
) -> None:
    """
    Visualize attention maps from the last transformer layer.

    Shows which patches attend to which, revealing what spatial
    relationships the encoder has learned.

    Args:
        encoder: Pretrained I-JEPA encoder
        dataloader: Data loader
        device: Computation device
        output_path: Where to save the plot
        n_images: Number of images to visualize
    """
    print("[VIS] Generating attention maps...")
    encoder.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:n_images].to(device)

    # Forward pass with attention extraction
    with torch.no_grad():
        _, attn_maps = encoder(images, context_mask=None, return_attention=True)

    if not attn_maps:
        print("[VIS] No attention maps available, skipping.")
        return

    # Use the last layer's attention: (B, num_heads, N, N)
    last_attn = attn_maps[-1]  # Last layer
    # Average over heads: (B, N, N)
    avg_attn = last_attn.mean(dim=1)

    grid_size = encoder.grid_size
    n_images = min(n_images, images.shape[0])

    fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 6))
    if n_images == 1:
        axes = axes.reshape(2, 1)

    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_denorm = (images * std + mean).clamp(0, 1).cpu()

    for i in range(n_images):
        # Original image
        img = images_denorm[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i}", fontsize=10)
        axes[0, i].axis('off')

        # Attention map: mean attention received by each patch
        # Sum attention over the source dimension -> how much attention
        # each patch receives from all other patches
        attn_map = avg_attn[i].mean(dim=0)  # (N,)
        attn_map = attn_map.reshape(grid_size, grid_size).cpu().numpy()

        axes[1, i].imshow(img, alpha=0.4)
        axes[1, i].imshow(attn_map, cmap='hot', alpha=0.6,
                          interpolation='bilinear',
                          extent=[0, img.shape[1], img.shape[0], 0])
        axes[1, i].set_title("Attention", fontsize=10)
        axes[1, i].axis('off')

    fig.suptitle("I-JEPA Encoder Attention Maps (Last Layer, Head Average)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Attention maps saved to {output_path}")


# ===================================================================
# 3. TRAINING LOSS CURVE
# ===================================================================

def plot_loss_curve(
    loss_history_path: str,
    output_path: str,
) -> None:
    """
    Plot training loss curve from saved history.

    Args:
        loss_history_path: Path to loss_history.npy file
        output_path: Where to save the plot
    """
    print("[VIS] Plotting training loss curve...")

    if not os.path.exists(loss_history_path):
        print(f"[VIS] Loss history not found at {loss_history_path}")
        return

    loss_history = np.load(loss_history_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color='#2196F3', linewidth=2,
            label='Training Loss (MSE in latent space)')

    # Add smoothed curve
    if len(loss_history) > 5:
        window = min(10, len(loss_history) // 3)
        smoothed = np.convolve(loss_history,
                               np.ones(window) / window,
                               mode='valid')
        smooth_epochs = epochs[window - 1:]
        ax.plot(smooth_epochs, smoothed, color='#F44336', linewidth=2,
                linestyle='--', alpha=0.8, label=f'Smoothed (window={window})')

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss (MSE in Embedding Space)", fontsize=13)
    ax.set_title("I-JEPA Training Loss\n"
                 "(Prediction in latent space, NOT pixel reconstruction)",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(loss_history))

    # Annotate final loss
    final_loss = loss_history[-1]
    ax.annotate(f'Final: {final_loss:.4f}',
                xy=(len(loss_history), final_loss),
                xytext=(-80, 20), textcoords='offset points',
                fontsize=11, color='#F44336',
                arrowprops=dict(arrowstyle='->', color='#F44336'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Loss curve saved to {output_path}")


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    """Generate all I-JEPA visualizations."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    vis_cfg = config['visualization']
    output_dir = vis_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")

    # STL-10 class names
    stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer',
                     'dog', 'horse', 'monkey', 'ship', 'truck']
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    dataset_name = config['data']['dataset']
    class_names = stl10_classes if dataset_name == 'stl10' else cifar10_classes

    # --- Load encoder ---
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
    if os.path.exists(ckpt_dir):
        ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    else:
        ckpt_files = []
    if ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
        print(f"[INIT] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['context_encoder'])
        print(f"[INIT] Loaded encoder from epoch {ckpt['epoch'] + 1}")
    else:
        print("[WARN] No checkpoint found — using randomly initialized encoder")

    # --- Load data ---
    data_cfg = config['data']
    test_dataset = get_labeled_dataset(
        data_cfg['dataset'], data_cfg['data_dir'],
        data_cfg['image_size'], is_train=False,
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=data_cfg['num_workers'], drop_last=False,
    )

    # --- Generate visualizations ---
    print("\n" + "=" * 50)
    print("Generating I-JEPA Visualizations")
    print("=" * 50)

    # 1. t-SNE
    plot_tsne(
        encoder, test_loader, device,
        output_path=os.path.join(output_dir, "tsne_representations.png"),
        n_samples=vis_cfg['tsne_n_samples'],
        perplexity=vis_cfg['tsne_perplexity'],
        class_names=class_names,
    )

    # 2. Attention maps
    plot_attention_maps(
        encoder, test_loader, device,
        output_path=os.path.join(output_dir, "attention_maps.png"),
        n_images=vis_cfg['attention_n_images'],
    )

    # 3. Loss curve
    loss_path = os.path.join(ckpt_dir, "loss_history.npy")
    plot_loss_curve(
        loss_history_path=loss_path,
        output_path=os.path.join(output_dir, "loss_curve.png"),
    )

    print("\n[DONE] All visualizations saved to: " + output_dir)


if __name__ == "__main__":
    main()
