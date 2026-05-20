"""Validation metric for MNIST autoencoder reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def recon_acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise score in [0, 1]; higher is better."""
    return (1.0 - (pred - target).abs()).clamp(min=0.0).mean()


@torch.no_grad()
def evaluation(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Average recon_acc over the validation set."""
    model.eval()
    total = 0.0
    n = 0
    for batch, _ in val_loader:
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)
        score = recon_acc(pred, batch).item()
        bs = batch.size(0)
        total += score * bs
        n += bs
    mean = total / max(n, 1)
    if math.isnan(mean) or math.isinf(mean):
        return float("nan")
    return mean
