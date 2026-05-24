# I-JEPA — Image Joint Embedding Predictive Architecture

A complete PyTorch implementation of **I-JEPA** (Image-based Joint-Embedding Predictive Architecture) from Meta AI.

> **Paper**: Assran et al., *"Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"*, CVPR 2023.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        I-JEPA ARCHITECTURE                            │
│                                                                       │
│   Input Image                                                         │
│   ┌──────────────┐                                                    │
│   │ ▓▓░░░░▓▓▓▓▓▓ │  ▓ = Context patches (visible)                    │
│   │ ▓▓░░░░▓▓▓▓▓▓ │  ░ = Target patches (masked)                     │
│   │ ▓▓▓▓▓▓▓▓░░░░ │                                                    │
│   │ ▓▓▓▓▓▓▓▓░░░░ │  Multi-block masking: 4 target blocks             │
│   └──────┬───────┘                                                    │
│          │                                                            │
│    ┌─────┴─────┐          ┌──────────────┐                            │
│    │  Context   │          │   Target     │                            │
│    │  patches   │          │   patches    │                            │
│    └─────┬─────┘          └──────┬───────┘                            │
│          │                       │                                     │
│          ▼                       ▼                                     │
│  ┌───────────────┐     ┌─────────────────┐                            │
│  │   CONTEXT     │     │    TARGET       │                            │
│  │   ENCODER     │     │    ENCODER      │  ◄── EMA copy (no grad)    │
│  │  (ViT, 6 blk)│     │   (ViT, 6 blk) │                            │
│  │   Trainable   │     │   Frozen (EMA)  │                            │
│  └───────┬───────┘     └────────┬────────┘                            │
│          │                      │                                     │
│          │    ┌──────────┐      │                                     │
│          ├───►│PREDICTOR │      │                                     │
│          │    │(3 blocks)│      │ Target representations              │
│          │    │ Narrow   │      │ (ground truth)                      │
│          │    └────┬─────┘      │                                     │
│          │         │            │                                     │
│          │         ▼            ▼                                     │
│          │    ┌─────────────────────┐                                 │
│          │    │   MSE LOSS          │  ◄── In LATENT SPACE            │
│          │    │   (embedding space) │      NOT pixel space!           │
│          │    └─────────────────────┘                                 │
│          │                                                            │
│          │    ═══════════════════════                                 │
│          │    KEY DIFFERENCE FROM MAE:                                │
│          │    • MAE: predicts PIXELS  → low-level features           │
│          │    • I-JEPA: predicts EMBEDDINGS → semantic features      │
│          │    ═══════════════════════                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
projet-jepa/
├── model/
│   ├── __init__.py           # Package docstring
│   ├── encoder.py            # ViT context encoder (trainable)
│   ├── predictor.py          # Narrow transformer predictor
│   └── target_encoder.py     # EMA target encoder (no gradients)
├── data/
│   ├── __init__.py
│   └── dataset.py            # STL-10 / CIFAR-10 loaders
├── utils/
│   ├── __init__.py
│   ├── masking.py            # Multi-block masking strategy
│   └── ema.py                # Exponential Moving Average utilities
├── train.py                  # Self-supervised pretraining loop
├── eval.py                   # Linear probing evaluation
├── visualize.py              # t-SNE, attention maps, loss curves
├── config.yaml               # All hyperparameters
└── README.md                 # This file
```

---

## Installation

### Prerequisites
- Python ≥ 3.8
- CUDA-capable GPU (recommended, 8GB+ VRAM)

### Install Dependencies

```bash
pip install torch torchvision pyyaml numpy matplotlib scikit-learn
```

That's it — no exotic dependencies. Only PyTorch ecosystem + standard scientific Python.

---

## Quick Start

### 1. Train I-JEPA (Self-Supervised Pretraining)

```bash
cd projet-jepa
python train.py
```

This will:
- Download STL-10 automatically (falls back to CIFAR-10 if needed)
- Train for 100 epochs with AdamW + cosine LR schedule
- Save checkpoints every 10 epochs to `./checkpoints/`

**Custom config:**
```bash
python train.py path/to/custom_config.yaml
```

### 2. Evaluate with Linear Probing

```bash
python eval.py
```

This will:
- Load the best checkpoint
- Freeze the encoder weights
- Train a linear classifier on top
- Report top-1 accuracy

### 3. Generate Visualizations

```bash
python visualize.py
```

This produces in `./figures/`:
- `tsne_representations.png` — t-SNE of learned features
- `attention_maps.png` — Encoder attention heatmaps
- `loss_curve.png` — Training loss over epochs

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Encoder** | ViT-Small | patch=8, dim=384, depth=6, heads=6 |
| **Predictor** | Narrow ViT | dim=192, depth=3, heads=6 |
| **Image size** | 96×96 | STL-10 native resolution |
| **Patches** | 12×12 = 144 | 96 / 8 = 12 |
| **Target blocks** | 4 | scale 15–20% each |
| **EMA momentum** | 0.996 → 1.0 | Cosine schedule |
| **Optimizer** | AdamW | lr=1.5e-4, wd=0.05 |
| **Batch size** | 256 | Reduce if OOM |
| **Epochs** | 100 | ~45 min on A100 |
| **AMP** | Enabled | Mixed precision (FP16) |

---

## Expected Results

### Linear Probe Accuracy (Top-1)

| Dataset  | Epochs | Accuracy |
|----------|--------|----------|
| STL-10   | 100    | ~60–68%  |
| CIFAR-10 | 100    | ~70–78%  |

> **Note**: These are results for a small-scale ViT-Small trained for 100
> epochs. The original paper uses ViT-Huge on ImageNet for 300+ epochs,
> achieving 70%+ linear probe and 83%+ fine-tuned accuracy.

### Training Tips for Better Results
- **More epochs**: Training for 300+ epochs significantly improves quality
- **Larger batch size**: Use 512 or 1024 if GPU memory allows
- **Gradient accumulation**: Simulate larger batches on smaller GPUs
- **Larger model**: Increase `embed_dim` to 768 and `depth` to 12

---

## What Makes I-JEPA Special?

### vs MAE (Masked Autoencoders)
| Aspect | MAE | I-JEPA |
|--------|-----|--------|
| Prediction target | Raw pixels | Learned embeddings |
| Loss space | Pixel space | Latent space |
| Decoder | Heavy pixel decoder | Narrow predictor |
| Target encoder | N/A | EMA momentum encoder |
| Augmentations needed | Strong | Minimal |
| Feature quality | Low-level biased | Semantic |

### vs Contrastive Methods (SimCLR, BYOL, DINO)
| Aspect | Contrastive | I-JEPA |
|--------|-------------|--------|
| Requires augmentations | Heavy (crop, color, blur) | Minimal |
| Invariance | To augmentations | To masking |
| Training signal | Image-level | Patch-level |
| Collapse prevention | Negatives / EMA / centering | EMA + prediction |

---

## GPU Memory Tips

If you encounter OOM errors:

1. **Reduce batch size** in `config.yaml`: `batch_size: 128` or `64`
2. **Disable AMP** if on CPU: `use_amp: false`
3. **Reduce model size**: lower `embed_dim` to 256 or `depth` to 4

---

## Citation

```bibtex
@inproceedings{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding
         Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and
          Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael
          and LeCun, Yann and Ballas, Nicolas},
  booktitle={CVPR},
  year={2023}
}
```

---

## License

This implementation is for educational and research purposes.
The I-JEPA method is by Meta AI (FAIR).
