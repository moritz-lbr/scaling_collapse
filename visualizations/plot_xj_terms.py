from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_time_series(npy_path: Path) -> np.ndarray:
    series = np.load(npy_path)
    if series.ndim != 2:
        raise ValueError(
            f"Expected a 2D array with shape (T, N), got {series.shape}"
        )
    return np.asarray(series, dtype=np.float32)


def plot_time_series(
    series: np.ndarray,
    output_path: Path,
    title: str,
    save_loss_frequency: int,
) -> None:
    if save_loss_frequency < 1:
        raise ValueError(
            f"save_loss_frequency must be >= 1, got {save_loss_frequency}"
        )

    steps = np.arange(series.shape[0], dtype=np.int64) * save_loss_frequency
    positive_mask = steps > 0
    if not np.any(positive_mask):
        raise ValueError("Need at least one positive training step for a log-scaled x-axis.")

    steps = steps[positive_mask]
    series = series[positive_mask]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, series.shape[1]))

    for idx in range(series.shape[1]):
        ax.plot(steps, series[:, idx], color=colors[idx % len(colors)], linewidth=0.9, alpha=0.9)
        # ax.plot(steps, np.log(np.abs(series[:, idx])), color=colors[idx % len(colors)], linewidth=0.9, alpha=0.9)

    ax.plot(steps, np.mean(series, axis=1), color="black", linewidth=1.5, label="Mean")
    ax.set_xscale("log", base=10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot all time series stored in an xj_terms_sample_*.npy file."
    )
    parser.add_argument(
        "--xj-path",
        type=Path,
        default=repo_root / "output_cov" / "xj_terms_sample_3.npy",
        help="Path to a 2D .npy file with shape (T, N).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to figures_xj_terms/<file-stem>.png",
    )
    parser.add_argument(
        "--save-loss-frequency",
        type=int,
        default=1,
        help="Spacing in true training steps between consecutive saved rows in the .npy file.",
    )

    args = parser.parse_args()
    xj_path = args.xj_path.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else repo_root / "figures_xj_terms" / f"{xj_path.stem}.png"
    )

    series = load_time_series(xj_path)
    plot_time_series(series, output_path, xj_path.stem, args.save_loss_frequency)


if __name__ == "__main__":
    main()
