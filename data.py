"""MNIST train/val dataloaders for autoencoder experiments."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = "data"
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
TRAIN_VAL_SPLIT = (55_000, 5_000)
_SPLIT_SEED = 42

_train_set = None
_val_set = None


def _load_splits(data_dir: str = DATA_DIR):
    global _train_set, _val_set
    if _train_set is not None and _val_set is not None:
        return _train_set, _val_set

    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,  # no-op if data/MNIST/raw/ already exists
        transform=transforms.ToTensor(),
    )
    gen = torch.Generator().manual_seed(_SPLIT_SEED)
    _train_set, _val_set = random_split(full_train, TRAIN_VAL_SPLIT, generator=gen)
    return _train_set, _val_set


def get_train_loader(
    batch_size: int = TRAIN_BATCH_SIZE,
    data_dir: str = DATA_DIR,
    num_workers: int = 0,
) -> DataLoader:
    train_set, _ = _load_splits(data_dir)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def get_val_loader(
    batch_size: int = VAL_BATCH_SIZE,
    data_dir: str = DATA_DIR,
    num_workers: int = 0,
) -> DataLoader:
    _, val_set = _load_splits(data_dir)
    return DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    x, _ = next(iter(train_loader))
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"batch shape: {tuple(x.shape)}, range=[{x.min():.3f}, {x.max():.3f}]")


if __name__ == "__main__":
    main()
