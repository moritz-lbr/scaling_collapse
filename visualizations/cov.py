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


def infer_spectral_radius_path(cov_path: Path) -> Path:
    if cov_path.name.startswith("cov_"):
        return cov_path.with_name(f"spec_rad_{cov_path.name[len('cov_'):]}")
    return cov_path.with_name(f"{cov_path.stem}_spec_rad{cov_path.suffix}")


def load_spectral_radii(npy_path: Path, expected_length: int) -> np.ndarray:
    spectral_radii = np.load(npy_path)
    if spectral_radii.ndim != 1:
        raise ValueError(
            f"Expected spectral radius array with shape (T,), got {spectral_radii.shape}"
        )
    if spectral_radii.shape[0] != expected_length:
        raise ValueError(
            "Expected spectral radius array to have one value per covariance "
            f"snapshot, got {spectral_radii.shape[0]} and {expected_length}."
        )
    return spectral_radii


def save_covariance_snapshots(
    covariances: np.ndarray,
    spectral_radii: np.ndarray,
    delta_t: int,
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
    if delta_t < 1:
        raise ValueError(f"delta_t must be >= 1, got {delta_t}")

    n_snapshots = covariances.shape[0]
    if spectral_radii.shape != (n_snapshots,):
        raise ValueError(
            "Expected spectral_radii to have one value per covariance snapshot, "
            f"got {spectral_radii.shape} for {n_snapshots} snapshots."
        )
    selected_indices = list(range(0, n_snapshots, frame_stride))
    if not selected_indices:
        raise ValueError("No covariance snapshots selected for plotting.")

    # vmin = float(np.min(covariances))
    # vmax = float(np.max(covariances))
    # if np.isclose(vmin, vmax):
    #     vmax = vmin + 1e-12
    vmin, vmax = np.quantile(covariances[selected_indices[0]].flatten(), [0.01, 0.99])
    colorbar_range = max(abs(vmin), abs(vmax))
    # colorbar_range = 1e-1

    frame_paths: list[Path] = []
    fig = plt.figure(figsize=(7.0, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 0.06],
        wspace=0.08,
        hspace=0.08,
        height_ratios=[4.0, 1.25],
    )
    ax = fig.add_subplot(grid[0, 0])
    spec_ax = fig.add_subplot(grid[1, 0])
    cbar_ax = fig.add_subplot(grid[0, 1])
    spacer_ax = fig.add_subplot(grid[1, 1])
    spacer_ax.axis("off")
    ax.set_anchor("W")
    spec_ax.set_anchor("W")
    image = ax.imshow(
        covariances[selected_indices[0]],
        cmap=cmap,
        vmin=-colorbar_range,
        vmax=colorbar_range,
        origin="lower",
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, cax=cbar_ax)
    # cbar.set_label(r"$Cov_t(\tilde{X}_i, \tilde{X}_j)$")
    ax.set_xlabel(r"$\tilde{X}_i$", fontsize=15)
    ax.set_ylabel(r"$\tilde{X}_j$", fontsize=15)

    training_steps = np.arange(n_snapshots, dtype=np.int64) * save_loss_frequency
    positive_mask = training_steps > 0
    if not np.any(positive_mask):
        raise ValueError("Need at least one positive training step for a log-scaled x-axis.")

    visible_training_steps = training_steps[positive_mask]
    visible_spectral_radii = spectral_radii[positive_mask]
    spec_ax.plot(visible_training_steps, visible_spectral_radii, color="black", linewidth=1.3)
    spec_ax.hlines(1.0, visible_training_steps[0], visible_training_steps[-1], colors="black", linestyles="--")
    x_min = float(visible_training_steps[0])
    left_window_line = spec_ax.axvline(x_min, color="red", linewidth=1.8)
    right_window_line = spec_ax.axvline(
        max((delta_t - 1) * save_loss_frequency, x_min),
        color="red",
        linewidth=1.8,
    )
    ymin, ymax = np.quantile(spectral_radii.flatten(), [0.01, 0.99])
    # spec_ax.set_ylim(0, ymax)
    x_max = max(
        float(visible_training_steps[-1]),
        float((n_snapshots + delta_t - 2) * save_loss_frequency),
    )
    if x_max <= x_min:
        x_max = x_min * 10.0
    spec_ax.set_xlim(x_min, x_max)
    spec_ax.set_xlabel("Training step", fontsize=12)
    spec_ax.set_ylabel(r"$\lambda_{\max}/N$", fontsize=12)
    spec_ax.set_title("Spectral radius", fontsize=12)
    spec_ax.set_xscale("log")
    spec_ax.grid(True, alpha=0.25)

    cov_log_path = output_dir / "cov_log_frames"
    cov_log_path.mkdir(parents=True, exist_ok=True)

    for frame_idx, t in enumerate(selected_indices):
        image.set_data(covariances[t])
        # true_step = (t + 1) * save_loss_frequency
        window_start = (t) * save_loss_frequency
        window_end = (t + delta_t-1) * save_loss_frequency
        visible_window_start = max(window_start, x_min)
        visible_window_end = max(window_end, x_min)
        left_window_line.set_xdata([visible_window_start, visible_window_start])
        right_window_line.set_xdata([visible_window_end, visible_window_end])
        ax.set_title(r"$Cov(log \, \tilde{X}_i, log \, \tilde{X}_j)$" + f" (Training Step: {window_start}-{window_end})", fontsize=15)
        frame_path = cov_log_path / f"cov_log_{frame_idx:03d}.png"
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
        "--spec-rad-path",
        type=Path,
        default=None,
        help=(
            "Path to spectral radius values in .npy format with shape (T,). "
            "Defaults to spec_rad_<layer>.npy next to --cov-path."
        ),
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        required=True,
        help="Time window for computing running covariances.",
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
        default="corr_Dense_0_log.gif",
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
    spec_rad_path = (
        args.spec_rad_path
        if args.spec_rad_path is not None
        else infer_spectral_radius_path(args.cov_path)
    )
    spectral_radii = load_spectral_radii(spec_rad_path, covariances.shape[0])
    frame_paths = save_covariance_snapshots(
        covariances,
        spectral_radii,
        args.delta_t,
        args.output_dir,
        frame_stride=args.frame_stride,
        save_loss_frequency=args.save_loss_frequency,
    )
    build_gif(frame_paths, args.output_dir / args.gif_name, duration_ms=args.frame_duration_ms)


if __name__ == "__main__":
    main()
