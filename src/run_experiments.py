#!/usr/bin/env python3
"""Run reproducible 2D generative-model experiments for the final project."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"


DATASETS = ["gaussian_mixture", "ring", "two_moons", "spiral"]
MODEL_ORDER = ["kde", "gmm", "vae", "ddpm"]
MODEL_LABELS = {"kde": "KDE", "gmm": "GMM", "vae": "VAE", "ddpm": "DDPM"}
DATASET_LABELS = {
    "gaussian_mixture": "Gaussian Mixture",
    "ring": "Ring",
    "two_moons": "Two Moons",
    "spiral": "Spiral",
}

PALETTE = {
    "real": "#39424e",
    "sample": "#2f7f9f",
    "sample2": "#d98c3f",
    "grid": "#d9e2df",
    "background": "#fbfcfa",
    "accent": "#5aa469",
    "bad": "#cc6b5a",
}


@dataclass
class ExperimentConfig:
    n_train: int = 1200
    n_test: int = 900
    n_generate: int = 900
    seeds: tuple[int, ...] = (0, 1, 2)
    vae_epochs: int = 320
    ddpm_epochs: int = 420
    ddpm_steps: int = 80
    device: str = "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("final_project/results"))
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=900)
    parser.add_argument("--n-generate", type=int, default=900)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--vae-epochs", type=int, default=320)
    parser.add_argument("--ddpm-epochs", type=int, default=420)
    parser.add_argument("--ddpm-steps", type=int, default=80)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_gaussian_mixture(n: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.array(
        [
            [-2.4, -1.8],
            [-1.0, 1.8],
            [1.1, -1.5],
            [2.3, 1.5],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    probs = np.array([0.20, 0.18, 0.22, 0.18, 0.22])
    covs = [
        np.array([[0.10, 0.03], [0.03, 0.18]]),
        np.array([[0.16, -0.04], [-0.04, 0.10]]),
        np.array([[0.13, 0.05], [0.05, 0.12]]),
        np.array([[0.18, -0.02], [-0.02, 0.15]]),
        np.array([[0.08, 0.00], [0.00, 0.08]]),
    ]
    comp = rng.choice(len(centers), size=n, p=probs)
    x = np.zeros((n, 2), dtype=np.float64)
    for k in range(len(centers)):
        idx = comp == k
        if idx.any():
            x[idx] = rng.multivariate_normal(centers[k], covs[k], size=idx.sum())
    return x


def make_ring(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi, size=n)
    radius = rng.normal(2.0, 0.10, size=n)
    x = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    x += rng.normal(0, 0.025, size=x.shape)
    return x


def make_two_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    seed = int(rng.integers(0, 2**31 - 1))
    x, _ = make_moons(n_samples=n, noise=0.055, random_state=seed)
    x[:, 0] = 2.1 * x[:, 0] - 1.0
    x[:, 1] = 2.2 * x[:, 1] - 0.2
    return x.astype(np.float64)


def make_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    arm = rng.integers(0, 2, size=n)
    t = rng.uniform(0.35, 3.7 * np.pi, size=n)
    r = 0.18 * t
    angle = t + arm * np.pi
    x = np.c_[r * np.cos(angle), r * np.sin(angle)]
    x += rng.normal(0, 0.065, size=x.shape)
    return x


GENERATORS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "gaussian_mixture": make_gaussian_mixture,
    "ring": make_ring,
    "two_moons": make_two_moons,
    "spiral": make_spiral,
}


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    return (train - mean) / std, (test - mean) / std, {
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    }


def inverse_standardize(x: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(stats["mean"])[None, :]
    std = np.asarray(stats["std"])[None, :]
    return x * std + mean


def generate_dataset(name: str, seed: int, n_train: int, n_test: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    raw_train = GENERATORS[name](n_train, rng)
    raw_test = GENERATORS[name](n_test, rng)
    train, test, stats = standardize(raw_train, raw_test)
    return {
        "name": name,
        "train": train.astype(np.float32),
        "test": test.astype(np.float32),
        "raw_train": raw_train.astype(np.float32),
        "raw_test": raw_test.astype(np.float32),
        "standardization": stats,
    }


def choose_kde(train: np.ndarray, seed: int) -> tuple[KernelDensity, dict[str, object]]:
    tr, val = train_test_split(train, test_size=0.22, random_state=seed)
    bandwidths = [0.045, 0.065, 0.09, 0.13, 0.18, 0.26, 0.36, 0.50]
    rows = []
    best_model = None
    best_nll = float("inf")
    for bw in bandwidths:
        model = KernelDensity(kernel="gaussian", bandwidth=bw)
        model.fit(tr)
        nll = -float(model.score(val) / len(val))
        rows.append({"bandwidth": bw, "val_nll": nll})
        if nll < best_nll:
            best_nll = nll
            best_model = KernelDensity(kernel="gaussian", bandwidth=bw).fit(train)
    assert best_model is not None
    return best_model, {"selected_bandwidth": best_model.bandwidth, "sweep": rows}


def choose_gmm(train: np.ndarray, seed: int) -> tuple[GaussianMixture, dict[str, object]]:
    components = [1, 2, 4, 6, 8, 12, 16, 20]
    rows = []
    best_model = None
    best_bic = float("inf")
    for k in components:
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-5,
            random_state=seed,
            max_iter=600,
            n_init=2,
        )
        model.fit(train)
        bic = float(model.bic(train))
        nll = -float(model.score(train))
        rows.append({"components": k, "bic": bic, "train_nll": nll})
        if bic < best_bic:
            best_bic = bic
            best_model = model
    assert best_model is not None
    return best_model, {"selected_components": int(best_model.n_components), "sweep": rows}


class VAE(nn.Module):
    def __init__(self, hidden: int = 96, latent: int = 2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-7, 5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.dec(z), mu, logvar

    def sample(self, n: int, device: str) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, 2, device=device)
            return self.dec(z).cpu().numpy()


def train_vae(train: np.ndarray, seed: int, epochs: int, device: str) -> tuple[VAE, dict[str, object]]:
    set_seed(seed)
    model = VAE().to(device)
    x = torch.as_tensor(train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    history = []
    beta = 0.025
    for epoch in range(1, epochs + 1):
        model.train()
        recon, mu, logvar = model(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 8) == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": float(loss.item()), "recon": float(recon_loss.item()), "kl": float(kl.item())})
    return model, {"epochs": epochs, "beta": beta, "history": history}


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        args = t[:, None].float() / freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class Denoiser(nn.Module):
    def __init__(self, hidden: int = 128, time_dim: int = 32):
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time(t)], dim=1))


def diffusion_schedule(steps: int, device: str) -> dict[str, torch.Tensor]:
    beta = torch.linspace(1e-4, 0.035, steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return {"beta": beta, "alpha": alpha, "alpha_bar": alpha_bar}


def train_ddpm(train: np.ndarray, seed: int, epochs: int, steps: int, device: str) -> tuple[Denoiser, dict[str, object]]:
    set_seed(seed)
    model = Denoiser().to(device)
    x0 = torch.as_tensor(train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.3e-3, weight_decay=1e-4)
    sched = diffusion_schedule(steps, device)
    history = []
    n = len(train)
    batch_size = min(512, n)
    gen = torch.Generator(device=device).manual_seed(seed + 1337) if device != "cpu" else torch.Generator().manual_seed(seed + 1337)
    for epoch in range(1, epochs + 1):
        idx = torch.randint(0, n, (batch_size,), generator=gen, device=device)
        xb = x0[idx]
        t = torch.randint(0, steps, (batch_size,), generator=gen, device=device)
        noise = torch.randn(xb.shape, generator=gen, device=device)
        ab = sched["alpha_bar"][t][:, None]
        xt = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * noise
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 8) == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": float(loss.item())})
    return model, {"epochs": epochs, "steps": steps, "history": history}


def sample_ddpm(model: Denoiser, n: int, steps: int, seed: int, device: str) -> np.ndarray:
    set_seed(seed)
    sched = diffusion_schedule(steps, device)
    model.eval()
    x = torch.randn(n, 2, device=device)
    with torch.no_grad():
        for t_idx in reversed(range(steps)):
            t = torch.full((n,), t_idx, dtype=torch.long, device=device)
            beta_t = sched["beta"][t_idx]
            alpha_t = sched["alpha"][t_idx]
            alpha_bar_t = sched["alpha_bar"][t_idx]
            eps = model(x, t)
            mean = (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_t)
            if t_idx > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * z
            else:
                x = mean
    return x.cpu().numpy()


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    sample = z[np.random.default_rng(0).choice(len(z), size=min(600, len(z)), replace=False)]
    d = pairwise_sq_dists(sample, sample)
    sigma2 = np.median(d[d > 0])
    sigma2 = float(sigma2 if sigma2 > 1e-9 else 1.0)
    kxx = np.exp(-pairwise_sq_dists(x, x) / (2 * sigma2)).mean()
    kyy = np.exp(-pairwise_sq_dists(y, y) / (2 * sigma2)).mean()
    kxy = np.exp(-pairwise_sq_dists(x, y) / (2 * sigma2)).mean()
    return float(max(kxx + kyy - 2 * kxy, 0.0))


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, seed: int, n_proj: int = 128) -> float:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_proj, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    vals = []
    for v in directions:
        px = np.sort(x @ v)
        py = np.sort(y @ v)
        m = min(len(px), len(py))
        vals.append(np.mean(np.abs(px[:m] - py[:m])))
    return float(np.mean(vals))


def precision_coverage(real: np.ndarray, gen: np.ndarray) -> tuple[float, float, float]:
    tree_real = cKDTree(real)
    tree_gen = cKDTree(gen)
    kth = min(6, len(real))
    dists, _ = tree_real.query(real, k=kth)
    radius = float(np.median(dists[:, -1]) * 1.5)
    precision = float(np.mean(tree_real.query(gen, k=1)[0] <= radius))
    coverage = float(np.mean(tree_gen.query(real, k=1)[0] <= radius))
    return precision, coverage, radius


def support_bins(name: str, x: np.ndarray, bins: int = 32) -> np.ndarray:
    if name in {"ring", "spiral"}:
        ang = np.arctan2(x[:, 1], x[:, 0])
        val = (ang + np.pi) / (2 * np.pi)
    elif name == "two_moons":
        val = (x[:, 0] - x[:, 0].min()) / (x[:, 0].ptp() + 1e-9)
    else:
        val = (np.arctan2(x[:, 1], x[:, 0]) + np.pi) / (2 * np.pi)
    idx = np.clip((val * bins).astype(int), 0, bins - 1)
    occ = np.zeros(bins, dtype=bool)
    occ[idx] = True
    return occ


def mode_coverage(name: str, real: np.ndarray, gen: np.ndarray) -> float:
    real_occ = support_bins(name, real)
    gen_occ = support_bins(name, gen)
    return float(np.sum(real_occ & gen_occ) / max(np.sum(real_occ), 1))


def evaluate_samples(
    dataset: str,
    model_name: str,
    test: np.ndarray,
    generated: np.ndarray,
    seed: int,
    nll: float | None,
    train_time: float,
    model_info: dict[str, object],
) -> dict[str, object]:
    precision, coverage, radius = precision_coverage(test, generated)
    return {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "mmd": mmd_rbf(test, generated),
        "sliced_wasserstein": sliced_wasserstein(test, generated, seed=seed),
        "precision": precision,
        "coverage": coverage,
        "support_coverage": mode_coverage(dataset, test, generated),
        "nll": nll,
        "radius": radius,
        "train_time_sec": train_time,
        "model_info": model_info,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = ["mmd", "sliced_wasserstein", "precision", "coverage", "support_coverage", "train_time_sec"]
    out = []
    for dataset in DATASETS:
        for model in MODEL_ORDER:
            group = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            if not group:
                continue
            item: dict[str, object] = {"dataset": dataset, "model": model, "n": len(group)}
            for key in metrics:
                vals = np.asarray([float(r[key]) for r in group], dtype=float)
                item[f"{key}_mean"] = float(vals.mean())
                item[f"{key}_std"] = float(vals.std(ddof=0))
            nll_vals = [r["nll"] for r in group if r["nll"] is not None]
            item["nll_mean"] = float(np.mean(nll_vals)) if nll_vals else None
            item["nll_std"] = float(np.std(nll_vals, ddof=0)) if nll_vals else None
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key != "model_info":
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def scatter_panel(ax: plt.Axes, real: np.ndarray, gen: np.ndarray | None, title: str) -> None:
    ax.set_facecolor(PALETTE["background"])
    if gen is None:
        ax.scatter(real[:, 0], real[:, 1], s=6, c=PALETTE["real"], alpha=0.62, linewidths=0)
    else:
        ax.scatter(real[:, 0], real[:, 1], s=5, c="#c9d1d3", alpha=0.42, linewidths=0, label="test")
        ax.scatter(gen[:, 0], gen[:, 1], s=6, c=PALETTE["sample"], alpha=0.62, linewidths=0, label="generated")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    for spine in ax.spines.values():
        spine.set_color("#d4d9d6")


def plot_hero(samples: dict[tuple[str, str], np.ndarray], tests: dict[str, np.ndarray], fig_dir: Path) -> None:
    cols = ["real"] + MODEL_ORDER
    fig, axes = plt.subplots(len(DATASETS), len(cols), figsize=(12.5, 9.2), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        real = tests[dataset]
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if col == "real":
                scatter_panel(ax, real, None, f"{DATASET_LABELS[dataset]}\nReal test")
            else:
                scatter_panel(ax, real, samples[(dataset, col)], MODEL_LABELS[col])
    fig.suptitle("Generated samples expose model bias across four 2D distributions", fontsize=15, fontweight="bold")
    fig.savefig(fig_dir / "fig1_hero_grid.png", dpi=240)
    fig.savefig(fig_dir / "fig1_hero_grid.pdf")
    plt.close(fig)


def plot_data_overview(tests: dict[str, np.ndarray], fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.4), constrained_layout=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        x = tests[dataset]
        scatter_panel(ax, x, None, DATASET_LABELS[dataset])
    fig.suptitle("Four evaluation distributions", fontsize=14, fontweight="bold")
    fig.savefig(fig_dir / "fig2_data_overview.png", dpi=240)
    fig.savefig(fig_dir / "fig2_data_overview.pdf")
    plt.close(fig)


def plot_metric_heatmap(agg: list[dict[str, object]], fig_dir: Path) -> None:
    metric = "mmd_mean"
    mat = np.zeros((len(DATASETS), len(MODEL_ORDER)))
    for i, dataset in enumerate(DATASETS):
        vals = []
        for model in MODEL_ORDER:
            row = next(r for r in agg if r["dataset"] == dataset and r["model"] == model)
            vals.append(float(row[metric]))
        vals_arr = np.asarray(vals)
        mat[i] = (vals_arr - vals_arr.min()) / (vals_arr.ptp() + 1e-12)
    cmap = LinearSegmentedColormap.from_list("rank", ["#4f9f87", "#f5e6a6", "#cf6f5f"])
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_yticks(range(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS])
    ax.set_title("Relative MMD rank within each dataset (lower is better)")
    for i in range(len(DATASETS)):
        for j in range(len(MODEL_ORDER)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(fig_dir / "fig5_metric_heatmap.png", dpi=240)
    fig.savefig(fig_dir / "fig5_metric_heatmap.pdf")
    plt.close(fig)


def plot_hyperparams(meta: dict[str, object], fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), constrained_layout=True)
    for col, dataset in enumerate(DATASETS):
        kde = meta[dataset]["kde"]["sweep"]
        gmm = meta[dataset]["gmm"]["sweep"]
        axes[0, col].plot([r["bandwidth"] for r in kde], [r["val_nll"] for r in kde], marker="o", color=PALETTE["sample"])
        axes[0, col].set_title(f"{DATASET_LABELS[dataset]} KDE")
        axes[0, col].set_xlabel("bandwidth")
        axes[0, col].set_ylabel("val NLL")
        axes[1, col].plot([r["components"] for r in gmm], [r["bic"] for r in gmm], marker="o", color=PALETTE["sample2"])
        axes[1, col].set_title(f"{DATASET_LABELS[dataset]} GMM")
        axes[1, col].set_xlabel("components")
        axes[1, col].set_ylabel("BIC")
    fig.savefig(fig_dir / "fig6_hyperparameter_sweep.png", dpi=240)
    fig.savefig(fig_dir / "fig6_hyperparameter_sweep.pdf")
    plt.close(fig)


def robustness_experiment(seed: int, config: ExperimentConfig, out_dir: Path) -> list[dict[str, object]]:
    rows = []
    rng = np.random.default_rng(seed + 2026)
    for dataset in DATASETS:
        clean = generate_dataset(dataset, seed + 10, config.n_train, config.n_test)
        test = clean["test"]
        for rate in [0.0, 0.03, 0.08, 0.15]:
            train = np.array(clean["train"], copy=True)
            n_out = int(len(train) * rate)
            if n_out > 0:
                outliers = rng.uniform(-3.2, 3.2, size=(n_out, 2)).astype(np.float32)
                train = np.vstack([train, outliers])
            for model_name in ["kde", "gmm"]:
                if model_name == "kde":
                    model, _ = choose_kde(train, seed)
                    gen = model.sample(config.n_generate, random_state=seed)
                    nll = -float(model.score(test) / len(test))
                else:
                    model, _ = choose_gmm(train, seed)
                    gen, _ = model.sample(config.n_generate)
                    nll = -float(model.score(test))
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "outlier_rate": rate,
                        "mmd": mmd_rbf(test, gen),
                        "sliced_wasserstein": sliced_wasserstein(test, gen, seed),
                        "nll": nll,
                    }
                )
    write_csv(out_dir / "robustness.csv", rows)
    return rows


def plot_robustness(rows: list[dict[str, object]], fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for model in ["kde", "gmm"]:
            group = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            ax.plot([r["outlier_rate"] for r in group], [r["mmd"] for r in group], marker="o", label=MODEL_LABELS[model])
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("outlier rate")
        ax.set_ylabel("MMD")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(fig_dir / "fig8_robustness.png", dpi=240)
    fig.savefig(fig_dir / "fig8_robustness.pdf")
    plt.close(fig)


def fmt(mean: object, std: object | None = None, digits: int = 3) -> str:
    if mean is None:
        return "--"
    if std is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def write_tex_tables(agg: list[dict[str, object]], out_dir: Path) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "数据集 & 模型 & MMD $\\downarrow$ & SWD $\\downarrow$ & Precision $\\uparrow$ & Coverage $\\uparrow$ & NLL $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for model in MODEL_ORDER:
            row = next(r for r in agg if r["dataset"] == dataset and r["model"] == model)
            lines.append(
                f"{DATASET_LABELS[dataset]} & {MODEL_LABELS[model]} & "
                f"{fmt(row['mmd_mean'], row['mmd_std'])} & "
                f"{fmt(row['sliced_wasserstein_mean'], row['sliced_wasserstein_std'])} & "
                f"{fmt(row['precision_mean'], row['precision_std'])} & "
                f"{fmt(row['coverage_mean'], row['coverage_std'])} & "
                f"{fmt(row['nll_mean'], row['nll_std'])} \\\\"
            )
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (table_dir / "main_results_table.tex").write_text("\n".join(lines), encoding="utf-8")

    best_lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "数据集 & 最低 MMD & 最低 SWD & 最高 Coverage \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        group = [r for r in agg if r["dataset"] == dataset]
        best_mmd = min(group, key=lambda r: float(r["mmd_mean"]))
        best_swd = min(group, key=lambda r: float(r["sliced_wasserstein_mean"]))
        best_cov = max(group, key=lambda r: float(r["coverage_mean"]))
        best_lines.append(
            f"{DATASET_LABELS[dataset]} & {MODEL_LABELS[best_mmd['model']]} & "
            f"{MODEL_LABELS[best_swd['model']]} & {MODEL_LABELS[best_cov['model']]} \\\\"
        )
    best_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (table_dir / "best_models_table.tex").write_text("\n".join(best_lines), encoding="utf-8")


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        n_train=args.n_train,
        n_test=args.n_test,
        n_generate=args.n_generate,
        seeds=tuple(args.seeds),
        vae_epochs=90 if args.quick else args.vae_epochs,
        ddpm_epochs=120 if args.quick else args.ddpm_epochs,
        ddpm_steps=40 if args.quick else args.ddpm_steps,
    )
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    sample_dir = out_dir / "samples"
    for path in [out_dir, fig_dir, sample_dir, out_dir / "tables"]:
        path.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    all_samples: dict[tuple[str, str], np.ndarray] = {}
    tests_for_plot: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {}
    start_all = time.time()

    for dataset_name in DATASETS:
        meta[dataset_name] = {}
        for seed in config.seeds:
            data = generate_dataset(dataset_name, seed, config.n_train, config.n_test)
            train = data["train"]
            test = data["test"]
            if seed == config.seeds[0]:
                tests_for_plot[dataset_name] = test

            start = time.time()
            kde, kde_info = choose_kde(train, seed)
            kde_samples = kde.sample(config.n_generate, random_state=seed + 77)
            kde_nll = -float(kde.score(test) / len(test))
            all_rows.append(evaluate_samples(dataset_name, "kde", test, kde_samples, seed, kde_nll, time.time() - start, kde_info))
            if seed == config.seeds[0]:
                all_samples[(dataset_name, "kde")] = kde_samples
                meta[dataset_name]["kde"] = kde_info

            start = time.time()
            gmm, gmm_info = choose_gmm(train, seed)
            gmm_samples, _ = gmm.sample(config.n_generate)
            gmm_nll = -float(gmm.score(test))
            all_rows.append(evaluate_samples(dataset_name, "gmm", test, gmm_samples, seed, gmm_nll, time.time() - start, gmm_info))
            if seed == config.seeds[0]:
                all_samples[(dataset_name, "gmm")] = gmm_samples
                meta[dataset_name]["gmm"] = gmm_info

            start = time.time()
            vae, vae_info = train_vae(train, seed, config.vae_epochs, config.device)
            vae_samples = vae.sample(config.n_generate, config.device)
            all_rows.append(evaluate_samples(dataset_name, "vae", test, vae_samples, seed, None, time.time() - start, vae_info))
            if seed == config.seeds[0]:
                all_samples[(dataset_name, "vae")] = vae_samples
                meta[dataset_name]["vae"] = vae_info

            start = time.time()
            ddpm, ddpm_info = train_ddpm(train, seed, config.ddpm_epochs, config.ddpm_steps, config.device)
            ddpm_samples = sample_ddpm(ddpm, config.n_generate, config.ddpm_steps, seed + 99, config.device)
            all_rows.append(evaluate_samples(dataset_name, "ddpm", test, ddpm_samples, seed, None, time.time() - start, ddpm_info))
            if seed == config.seeds[0]:
                all_samples[(dataset_name, "ddpm")] = ddpm_samples
                meta[dataset_name]["ddpm"] = ddpm_info

    agg = aggregate(all_rows)
    write_csv(out_dir / "metrics_by_seed.csv", all_rows)
    write_csv(out_dir / "metrics_summary.csv", agg)
    save_json(out_dir / "metrics_by_seed.json", all_rows)
    save_json(out_dir / "metrics_summary.json", agg)
    save_json(out_dir / "model_metadata.json", meta)
    write_tex_tables(agg, out_dir)

    for (dataset, model), samples in all_samples.items():
        np.save(sample_dir / f"{dataset}_{model}_samples.npy", samples)
    for dataset, test in tests_for_plot.items():
        np.save(sample_dir / f"{dataset}_test.npy", test)

    plot_hero(all_samples, tests_for_plot, fig_dir)
    plot_data_overview(tests_for_plot, fig_dir)
    plot_metric_heatmap(agg, fig_dir)
    plot_hyperparams(meta, fig_dir)
    robust_rows = robustness_experiment(config.seeds[0], config, out_dir)
    plot_robustness(robust_rows, fig_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "elapsed_sec": time.time() - start_all,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "notes": "Data are generated from fixed synthetic mechanisms because no TA data files were present in the project directory.",
    }
    save_json(out_dir / "run_manifest.json", manifest)

    print(f"Wrote results to {out_dir}")
    print(f"Elapsed: {manifest['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
