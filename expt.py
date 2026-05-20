"""Minimal MNIST autoencoder experiment."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import DATA_DIR, get_train_loader, get_val_loader
from eval import evaluation, recon_acc

TIME_BUDGET = 60.0  # seconds of wall-clock training (excludes startup)
TOTAL_TIME_LIMIT = 120.0  # seconds total wall time before treating as failure
RESULTS_PATH = "results.tsv"
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 256


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view_as(x)


def git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def expts_dir() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expts")
    os.makedirs(path, exist_ok=True)
    return path


def append_results(
    expt_id: str,
    commit: str,
    metric: str,
    memory_gb: str,
    status: str,
    description: str,
) -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_PATH)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("expt_id\tcommit\tmetric\tmemory_gb\tstatus\tdescription\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{expt_id}\t{commit}\t{metric}\t{memory_gb}\t{status}\t{description}\n")


@torch.no_grad()
def epoch_metrics(model, train_loader, val_loader, device, criterion):
    model.eval()
    train_loss = train_acc = train_n = 0.0
    for xb, _ in train_loader:
        xb = xb.to(device)
        pred = model(xb)
        bs = xb.size(0)
        train_loss += criterion(pred, xb).item() * bs
        train_acc += recon_acc(pred, xb).item() * bs
        train_n += bs

    val_loss = val_acc = val_n = 0.0
    for xb, _ in val_loader:
        xb = xb.to(device)
        pred = model(xb)
        bs = xb.size(0)
        val_loss += criterion(pred, xb).item() * bs
        val_acc += recon_acc(pred, xb).item() * bs
        val_n += bs

    return (
        train_loss / train_n,
        val_loss / val_n,
        train_acc / train_n,
        val_acc / val_n,
    )


def plot_learning_curve(expt_id: str, history: dict[str, list[float]], total_epochs: int) -> None:
    if total_epochs <= 0:
        return
    frac = [e / total_epochs for e in history["epochs"]]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(frac, history["train_loss"], label="train loss")
    ax1.plot(frac, history["val_loss"], label="val loss")
    ax1.set_xlabel("fraction of trained epochs")
    ax1.set_ylabel("MSE loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(frac, history["train_recon_acc"], color="green", alpha=0.7, label="train recon_acc")
    ax2.plot(frac, history["val_recon_acc"], color="darkgreen", label="val recon_acc")
    ax2.set_ylabel("recon_acc")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(expts_dir(), f"learning_curve_{expt_id}.png"), dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_reconstruction(model, val_loader, device, expt_id: str, n: int = 8) -> None:
    model.eval()
    xb, _ = next(iter(val_loader))
    xb = xb[:n].to(device)
    pred = model(xb).cpu()
    orig = xb.cpu()

    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(orig[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstructed")
    fig.tight_layout()
    fig.savefig(os.path.join(expts_dir(), f"reconstruction_{expt_id}.png"), dpi=120)
    plt.close(fig)


def print_summary(metric_val, training_seconds, total_seconds, peak_vram_mb, num_params_m):
    print("---")
    print(f"metric: {metric_val:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"num_params_M: {num_params_m:.1f}")
    print("---")


def fail_run(
    expt_id: str,
    commit: str,
    description: str,
    training_seconds: float,
    script_t0: float,
    peak_vram_mb: float,
    num_params_m: float,
    reason: str,
) -> None:
    append_results(expt_id, commit, "NA", "NA", "crash", f"{description}: {reason}")
    print_summary(float("nan"), training_seconds, time.perf_counter() - script_t0, peak_vram_mb, num_params_m)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python expt.py <expt_id>", file=sys.stderr)
        sys.exit(2)
    expt_id = sys.argv[1]
    script_t0 = time.perf_counter()
    description = os.environ.get("EXPERIMENT_DESC", "mnist_ae_baseline")
    commit = git_commit_short()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_train_loader(batch_size=TRAIN_BATCH_SIZE, data_dir=DATA_DIR)
    val_loader = get_val_loader(batch_size=VAL_BATCH_SIZE, data_dir=DATA_DIR)

    model = Autoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_t0 = time.perf_counter()
    training_seconds = 0.0
    peak_vram_mb = 0.0
    epoch = 0
    history: dict[str, list] = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_recon_acc": [],
        "val_recon_acc": [],
    }

    try:
        while training_seconds < TIME_BUDGET:
            if time.perf_counter() - script_t0 >= TOTAL_TIME_LIMIT:
                fail_run(expt_id, commit, description, training_seconds, script_t0, peak_vram_mb, num_params_m, "total time limit")

            epoch += 1
            model.train()
            for xb, _ in train_loader:
                if time.perf_counter() - train_t0 >= TIME_BUDGET:
                    break
                if time.perf_counter() - script_t0 >= TOTAL_TIME_LIMIT:
                    fail_run(expt_id, commit, description, training_seconds, script_t0, peak_vram_mb, num_params_m, "total time limit")

                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(xb), xb)
                loss.backward()
                opt.step()

            training_seconds = time.perf_counter() - train_t0
            if training_seconds >= TIME_BUDGET:
                break

            tr_loss, va_loss, tr_acc, va_acc = epoch_metrics(
                model, train_loader, val_loader, device, criterion
            )
            history["epochs"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_recon_acc"].append(tr_acc)
            history["val_recon_acc"].append(va_acc)
            if device.type == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        if time.perf_counter() - script_t0 >= TOTAL_TIME_LIMIT:
            fail_run(expt_id, commit, description, training_seconds, script_t0, peak_vram_mb, num_params_m, "total time limit")

        metric_val = evaluation(model, val_loader, device)
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        if history["train_loss"]:
            plot_learning_curve(expt_id, history, epoch)
        plot_reconstruction(model, val_loader, device, expt_id)

    except Exception:
        append_results(expt_id, commit, "NA", "NA", "crash", description)
        print_summary(float("nan"), training_seconds, time.perf_counter() - script_t0, peak_vram_mb, num_params_m)
        raise

    training_seconds = time.perf_counter() - train_t0
    total_seconds = time.perf_counter() - script_t0

    if math.isnan(metric_val):
        append_results(expt_id, commit, "NA", "NA", "crash", description)
        print_summary(float("nan"), training_seconds, total_seconds, peak_vram_mb, num_params_m)
        sys.exit(1)

    memory_gb = f"{round(peak_vram_mb / 1024.0, 1):.1f}" if peak_vram_mb > 0 else "0.0"
    append_results(expt_id, commit, f"{metric_val:.6f}", memory_gb, "keep", description)
    print_summary(metric_val, training_seconds, total_seconds, peak_vram_mb, num_params_m)


if __name__ == "__main__":
    main()
