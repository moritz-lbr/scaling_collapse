from __future__ import annotations

import argparse
import os
from pathlib import Path

_CACHE_DIR = Path(".cache/matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISTRIBUTION_WINDOW_SIZE = 10


def load_time_series(npy_path: Path) -> np.ndarray:
    series = np.load(npy_path)
    if series.ndim != 2:
        raise ValueError(
            f"Expected a 2D array with shape (T, N), got {series.shape}"
        )
    return np.asarray(series, dtype=np.float32)


def infer_layer_label(xj_path: Path, layer: str | None) -> str:
    if layer is not None:
        return layer

    stem = xj_path.stem
    if stem.startswith("xj_"):
        return stem[len("xj_") :]
    return stem


def is_logits_label(layer: str) -> bool:
    return layer.strip().lower().startswith("logits")


def format_layer_index(layer: str) -> str:
    if not layer:
        return "?"

    suffix = layer.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return suffix
    return layer[-1]


def build_title(layer: str, num_features: int) -> str:
    if is_logits_label(layer):
        return (
            r"Set of $\{\tilde{f}_j\}_{j=1...n_l}$ for output logits with width "
            rf"$n_l = {num_features}$"
        )

    layer_index = format_layer_index(layer)
    return (
        r"Set of $\{\tilde{X}^l_j\}_{j=1...n_l}$ for layer "
        rf"$l = {layer_index}$ of network with width "
        rf"$n_{{{layer_index}}} = {num_features}$"
    )


def build_value_label(layer: str) -> str:
    if is_logits_label(layer):
        return r"$\{\tilde{f}_j\}_{j=1...n_l}$"

    layer_index = format_layer_index(layer)
    return rf"$\{{\tilde{{X}}^{layer_index}_j\}}_{{j=1...n_{{{layer_index}}}}}$"


def build_log_label(layer: str) -> str:
    if is_logits_label(layer):
        return r"$\{\log (\tilde{f}_j)\}_{j=1...n_l}$"

    layer_index = format_layer_index(layer)
    return rf"$\{{\log (\tilde{{X}}^{layer_index}_j)\}}_{{j=1...n_{{{layer_index}}}}}$"


def build_centered_label(layer: str) -> str:
    if is_logits_label(layer):
        return (
            r"$\{\log (\tilde{f}_j) - \langle \log (\tilde{f}_j) \rangle_{n_l}\}_{j=1...n_l}$"
        )

    layer_index = format_layer_index(layer)
    return (
        rf"$\{{\log (\tilde{{X}}^{layer_index}_j) - "
        rf"\langle \log (\tilde{{X}}^{layer_index}_j) \rangle_{{n_{{{layer_index}}}}}\}}_{{j=1...n_{{{layer_index}}}}}$"
    )


def build_histogram_bins(values: np.ndarray) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot build a histogram from empty or non-finite values.")

    data_min = float(finite_values.min())
    data_max = float(finite_values.max())
    if data_min == data_max:
        span = max(abs(data_min), 1.0)
        data_min -= 0.5 * span
        data_max += 0.5 * span

    bin_count = min(80, max(20, int(np.sqrt(finite_values.size))))
    return np.linspace(data_min, data_max, bin_count + 1)


def plot_distribution_windows(
    series: np.ndarray,
    output_dir: Path,
    output_stem: str,
    layer: str,
    save_loss_frequency: int,
    window_size: int,
) -> None:
    if save_loss_frequency < 1:
        raise ValueError(
            f"save_loss_frequency must be >= 1, got {save_loss_frequency}"
        )
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    centered_series = series - np.mean(series, axis=1, keepdims=True)
    steps = np.arange(series.shape[0], dtype=np.int64) * save_loss_frequency
    num_features = series.shape[1]
    label_fontsize = 14

    output_dir.mkdir(parents=True, exist_ok=True)
    for window_index, start_idx in enumerate(
        range(0, series.shape[0], window_size),
        start=1,
    ):
        end_idx = min(start_idx + window_size, series.shape[0])
        if start_idx >= end_idx:
            continue

        window_series = series[start_idx:end_idx].reshape(-1)
        window_centered_series = centered_series[start_idx:end_idx].reshape(-1)
        window_series = window_series[np.isfinite(window_series)]
        window_centered_series = window_centered_series[np.isfinite(window_centered_series)]
        if window_series.size == 0 or window_centered_series.size == 0:
            continue

        series_bins = build_histogram_bins(window_series)
        centered_bins = build_histogram_bins(window_centered_series)
        start_step = int(steps[start_idx])
        end_step = int(steps[end_idx - 1])

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10.0, 8.0),
            constrained_layout=True,
        )
        axes[0].hist(
            window_series,
            bins=series_bins,
            density=True,
            color="C0",
            alpha=0.85,
            linewidth=1.2,
        )
        axes[1].hist(
            window_centered_series,
            bins=centered_bins,
            density=True,
            color="C1",
            alpha=0.85,
            linewidth=1.2,
        )

        axes[0].set_title(build_log_label(layer), fontsize=label_fontsize)
        axes[1].set_title(build_centered_label(layer), fontsize=label_fontsize)
        axes[0].set_ylabel("Density", fontsize=label_fontsize)
        axes[1].set_ylabel("Density", fontsize=label_fontsize)
        axes[0].set_xlabel("Value", fontsize=label_fontsize)
        axes[1].set_xlabel("Value", fontsize=label_fontsize)
        for ax in axes:
            ax.grid(True, linewidth=0.35, alpha=0.35)
            ax.tick_params(axis="both", labelsize=12)

        fig.suptitle(
            (
                f"{build_title(layer, num_features)}\n"
                f"Distribution window {window_index}: "
                f"saved rows [{start_idx}, {end_idx - 1}], "
                f"training steps [{start_step}, {end_step}]"
            ),
            fontsize=16,
        )
        output_path = output_dir / (
            f"{output_stem}_distribution_window_{window_index:03d}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot histogram distributions for all xj terms stored in an xj_*.npy, "
            "xj_terms_sample_*.npy, or xj_logits_sample_*.npy file."
        )
    )
    parser.add_argument(
        "--xj-path",
        type=Path,
        default=repo_root / "output_cov" / "xj_terms_sample_3.npy",
        help="Path to a 2D .npy file with shape (T, N).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "xj_terms_distributions",
        help="Directory where the distribution PNGs will be written.",
    )
    parser.add_argument(
        "--save-loss-frequency",
        type=int,
        default=1,
        help="Spacing in true training steps between consecutive saved rows in the .npy file.",
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        default=DISTRIBUTION_WINDOW_SIZE,
        help="Number of consecutive saved rows per non-overlapping distribution window.",
    )
    parser.add_argument(
        "--layer",
        "--title",
        dest="layer",
        type=str,
        default=None,
        help="Optional layer identifier used to build the plot title. Defaults to the xj file stem.",
    )

    args = parser.parse_args()
    xj_path = args.xj_path.resolve()
    output_dir = args.output_dir.resolve()

    series = load_time_series(xj_path)
    layer = infer_layer_label(xj_path, args.layer)
    plot_distribution_windows(
        series,
        output_dir,
        xj_path.stem,
        layer,
        args.save_loss_frequency,
        args.delta_t,
    )


if __name__ == "__main__":
    main()
