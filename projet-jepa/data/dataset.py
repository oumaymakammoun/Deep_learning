"""
Dataset Loading for I-JEPA
===========================
Provides data loaders for STL-10 (preferred) and CIFAR-10 (fallback).

STL-10 is ideal for I-JEPA because:
- Higher resolution (96x96) gives more patches to mask
- Has a large unlabeled split (100k images) perfect for self-supervised learning
- 10 classes for linear probe evaluation

CIFAR-10 is used as fallback (32x32 → resized to 96x96).

Augmentation strategy for I-JEPA:
- MINIMAL augmentations (unlike contrastive methods like SimCLR/MoCo)
- I-JEPA does NOT rely on data augmentation for invariance learning
- Only basic: random horizontal flip + normalization
- This is a KEY advantage of I-JEPA: it learns without augmentation tricks
"""

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T


def get_transforms(image_size: int = 96, is_train: bool = True
                   ) -> T.Compose:
    """
    Get image transforms for I-JEPA.

    I-JEPA uses MINIMAL augmentations compared to contrastive methods.
    - No color jittering, no Gaussian blur, no solarization
    - Just random crop + flip for training; center crop for eval
    This is because I-JEPA learns from prediction, not invariance.

    Args:
        image_size: Target image resolution
        is_train: If True, apply training augmentations

    Returns:
        Composed transform pipeline
    """
    if is_train:
        return T.Compose([
            T.Resize(image_size),
            T.RandomCrop(image_size, padding=image_size // 8),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


def get_dataset(
    dataset_name: str = "stl10",
    data_dir: str = "./datasets",
    image_size: int = 96,
    is_train: bool = True,
) -> Dataset:
    """
    Load STL-10 or CIFAR-10 dataset.

    For self-supervised pretraining:
    - STL-10: uses 'train+unlabeled' split (105k images)
    - CIFAR-10: uses 'train' split (50k images)

    For evaluation (linear probing):
    - STL-10: 'train' (5k) for training probe, 'test' (8k) for eval
    - CIFAR-10: 'train' (50k) for training probe, 'test' (10k) for eval

    Args:
        dataset_name: "stl10" or "cifar10"
        data_dir: Directory to download/store data
        image_size: Target image resolution
        is_train: Training or evaluation split

    Returns:
        PyTorch Dataset
    """
    transform = get_transforms(image_size, is_train)

    if dataset_name.lower() == "stl10":
        try:
            if is_train:
                # For SSL pretraining: use train+unlabeled (105k images)
                dataset = torchvision.datasets.STL10(
                    root=data_dir, split='train+unlabeled',
                    download=True, transform=transform,
                )
            else:
                dataset = torchvision.datasets.STL10(
                    root=data_dir, split='test',
                    download=True, transform=transform,
                )
            print(f"[DATA] Loaded STL-10 ({'train+unlabeled' if is_train else 'test'}): "
                  f"{len(dataset)} images")
            return dataset
        except Exception as e:
            print(f"[DATA] STL-10 failed ({e}), falling back to CIFAR-10")
            dataset_name = "cifar10"

    # CIFAR-10 fallback
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=is_train,
        download=True, transform=transform,
    )
    print(f"[DATA] Loaded CIFAR-10 ({'train' if is_train else 'test'}): "
          f"{len(dataset)} images")
    return dataset


def get_labeled_dataset(
    dataset_name: str = "stl10",
    data_dir: str = "./datasets",
    image_size: int = 96,
    is_train: bool = True,
) -> Dataset:
    """
    Load LABELED dataset for linear probing evaluation.

    Unlike get_dataset() which may use unlabeled data for pretraining,
    this always returns labeled data for supervised evaluation.

    Args:
        dataset_name: "stl10" or "cifar10"
        data_dir: Directory for data
        image_size: Target resolution
        is_train: Train or test split

    Returns:
        Labeled PyTorch Dataset
    """
    transform = get_transforms(image_size, is_train=False)  # no augment for eval

    if dataset_name.lower() == "stl10":
        try:
            split = 'train' if is_train else 'test'
            dataset = torchvision.datasets.STL10(
                root=data_dir, split=split,
                download=True, transform=transform,
            )
            print(f"[DATA] Labeled STL-10 ({split}): {len(dataset)} images")
            return dataset
        except Exception:
            dataset_name = "cifar10"

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=is_train,
        download=True, transform=transform,
    )
    print(f"[DATA] Labeled CIFAR-10 ({'train' if is_train else 'test'}): "
          f"{len(dataset)} images")
    return dataset


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with optimal settings.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch (important for batch norm)

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
