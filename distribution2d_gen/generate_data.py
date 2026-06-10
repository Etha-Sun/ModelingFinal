#!/usr/bin/env python3
"""Generate datasets for the final project on 2D generative modeling.

The script creates four public distributions and one hidden test split:

0. Gaussian Mixture
1. Ring
2. Two Moons
3. Spiral

Default output files:

data/train.npy
data/test.npy
data/train_label.npy
data/test_label.npy
data/hidden_test.npy
data/hidden_test_label.npy
data/metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ["gaussian_mixture", "ring", "two_moons", "spiral"]


def make_gaussian_mixture(n: int, rng: np.random.Generator) -> np.ndarray:
    num_modes = 8
    radius = 2.6
    noise = 0.16

    angles = np.linspace(0.0, 2.0 * np.pi, num_modes, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    mode_ids = rng.integers(0, num_modes, size=n)
    return centers[mode_ids] + rng.normal(0.0, noise, size=(n, 2))


def make_ring(n: int, rng: np.random.Generator) -> np.ndarray:
    radius = 2.2
    radial_noise = 0.13
    tangential_noise = 0.03

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = radius + rng.normal(0.0, radial_noise, size=n)
    points = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return points + rng.normal(0.0, tangential_noise, size=(n, 2))


def make_two_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    noise = 0.08
    n_upper = n // 2
    n_lower = n - n_upper

    theta_upper = rng.uniform(0.0, np.pi, size=n_upper)
    upper = np.stack([np.cos(theta_upper), np.sin(theta_upper)], axis=1)

    theta_lower = rng.uniform(0.0, np.pi, size=n_lower)
    lower = np.stack([1.0 - np.cos(theta_lower), 0.5 - np.sin(theta_lower)], axis=1)

    points = np.concatenate([upper, lower], axis=0)
    points += rng.normal(0.0, noise, size=points.shape)
    points *= 1.75
    points -= np.array([0.9, 0.25])
    rng.shuffle(points, axis=0)
    return points


def make_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    noise = 0.08
    n_arm1 = n // 2
    n_arm2 = n - n_arm1

    def arm(m: int, phase: float) -> np.ndarray:
        t = rng.uniform(0.25, 4.0 * np.pi, size=m)
        r = 0.22 * t
        return np.stack([r * np.cos(t + phase), r * np.sin(t + phase)], axis=1)

    points = np.concatenate([arm(n_arm1, 0.0), arm(n_arm2, np.pi)], axis=0)
    points += rng.normal(0.0, noise, size=points.shape)
    rng.shuffle(points, axis=0)
    return points


GENERATORS = [make_gaussian_mixture, make_ring, make_two_moons, make_spiral]


def make_split(n_per_class: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for label, generator in enumerate(GENERATORS):
        points = generator(n_per_class, rng).astype(np.float32)
        xs.append(points)
        ys.append(np.full(n_per_class, label, dtype=np.int64))

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order]


def save_preview(
    output_dir: Path,
    train: np.ndarray,
    train_label: np.ndarray,
    test: np.ndarray,
    test_label: np.ndarray,
    hidden: np.ndarray,
    hidden_label: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skip preview.png because matplotlib is not installed.")
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (train, train_label),
        "test": (test, test_label),
        "hidden_test": (hidden, hidden_label),
        "all_splits": (
            np.concatenate([train, test, hidden], axis=0),
            np.concatenate([train_label, test_label, hidden_label], axis=0),
        ),
    }

    for split_name, (x, y) in splits.items():
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        for label, ax in enumerate(axes.ravel()):
            points = x[y == label]
            ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.55, linewidths=0)
            ax.set_title(f"{label}: {CLASS_NAMES[label]}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(-4.0, 4.0)
            ax.grid(alpha=0.2)

        fig.suptitle(split_name, y=0.995)
        fig.tight_layout()
        fig.savefig(figure_dir / f"preview_{split_name}.png", dpi=180)
        if split_name == "train":
            fig.savefig(output_dir / "preview.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--test-per-class", type=int, default=2000)
    parser.add_argument("--hidden-per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--plot", action="store_true", help="save data/preview.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, train_label = make_split(args.train_per_class, np.random.default_rng(args.seed))
    test, test_label = make_split(args.test_per_class, np.random.default_rng(args.seed + 1))
    hidden, hidden_label = make_split(args.hidden_per_class, np.random.default_rng(args.seed + 2))

    np.save(args.output_dir / "train.npy", train)
    np.save(args.output_dir / "test.npy", test)
    np.save(args.output_dir / "train_label.npy", train_label)
    np.save(args.output_dir / "test_label.npy", test_label)
    np.save(args.output_dir / "hidden_test.npy", hidden)
    np.save(args.output_dir / "hidden_test_label.npy", hidden_label)

    metadata = {
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "label_mapping": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "hidden_test_shape": list(hidden.shape),
        "coordinate_range_note": "Samples are designed to lie mostly in [-4, 4] x [-4, 4].",
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    if args.plot:
        save_preview(args.output_dir, train, train_label, test, test_label, hidden, hidden_label)

    print(f"Saved dataset to {args.output_dir.resolve()}")
    print(f"train: {train.shape}, test: {test.shape}, hidden_test: {hidden.shape}")


if __name__ == "__main__":
    main()
