from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_covariances(npy_path: Path) -> np.ndarray:
    covariances = np.load(npy_path)
    if covariances.ndim != 3:
        raise ValueError(
            f"Expected covariance array with shape (T, N, N), got {covariances.shape}"
        )
    if covariances.shape[1] != covariances.shape[2]:
        raise ValueError(
            f"Expected square covariance matrices, got {covariances.shape[1:]}"
        )
    return covariances


def save_covariance_snapshots(
    covariances: np.ndarray,
    output_dir: Path,
    *,
    cmap: str = "coolwarm",
    frame_stride: int = 1,
    save_loss_frequency: int = 1,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if save_loss_frequency < 1:
        raise ValueError(
            f"save_loss_frequency must be >= 1, got {save_loss_frequency}"
        )

    n_snapshots = covariances.shape[0]
    start_idx = frame_stride - 1
    selected_indices = list(range(start_idx, n_snapshots, frame_stride))
    if not selected_indices:
        raise ValueError("No covariance snapshots selected for plotting.")

    vmin = float(np.min(covariances))
    vmax = float(np.max(covariances))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    frame_paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = ax.imshow(
        covariances[selected_indices[0]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance")
    ax.set_xlabel("Variable j")
    ax.set_ylabel("Variable i")

    for frame_idx, t in enumerate(selected_indices):
        image.set_data(covariances[t])
        true_step = (t + 1) * save_loss_frequency
        ax.set_title(f"Covariance Heatmap (step={true_step})")
        frame_path = output_dir / f"covariance_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=150)
        frame_paths.append(frame_path)

    plt.close(fig)
    return frame_paths


def build_gif(frame_paths: list[Path], gif_path: Path, *, duration_ms: int = 80) -> None:
    if not frame_paths:
        raise ValueError("No frame paths were provided for GIF creation.")

    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            frames.append(img.convert("RGB"))

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    for frame in frames:
        frame.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Create covariance heatmap snapshots and a GIF from an .npy file."
    )
    parser.add_argument(
        "--cov-path",
        type=Path,
        default=repo_root / "cov_Dense_0.npy",
        help="Path to covariance snapshots in .npy format with shape (T, N, N).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures_cov",
        help="Directory for saved PNG snapshots and GIF.",
    )
    parser.add_argument(
        "--gif-name",
        type=str,
        default="cov_Dense_0.gif",
        help="Filename of the generated GIF.",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=80,
        help="Duration of each GIF frame in milliseconds.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Use every n-th covariance snapshot when creating frames/GIF "
            "(n, 2n, 3n, ... in 1-based indexing)."
        ),
    )
    parser.add_argument(
        "--save-loss-frequency",
        type=int,
        default=1,
        help=(
            "Step multiplier for title labels from simulation_config.yaml "
            "(true_step = (snapshot_index + 1) * save_loss_frequency)."
        ),
    )
    args = parser.parse_args()

    covariances = load_covariances(args.cov_path)
    frame_paths = save_covariance_snapshots(
        covariances,
        args.output_dir,
        frame_stride=args.frame_stride,
        save_loss_frequency=args.save_loss_frequency,
    )
    build_gif(frame_paths, args.output_dir / args.gif_name, duration_ms=args.frame_duration_ms)


if __name__ == "__main__":
    main()
