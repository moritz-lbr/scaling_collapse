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


def build_raw_mean_label(layer: str) -> str:
    if is_logits_label(layer):
        return r"$\langle \tilde{f}_j \rangle_{n_l}$"

    layer_index = format_layer_index(layer)
    return rf"$\langle \tilde{{X}}^{layer_index}_j \rangle_{{n_{{{layer_index}}}}}$"


def build_log_mean_label(layer: str) -> str:
    if is_logits_label(layer):
        return r"$\langle \log (\tilde{f}_j) \rangle_{n_l}$"

    layer_index = format_layer_index(layer)
    return rf"$\langle \log (\tilde{{X}}^{layer_index}_j) \rangle_{{n_{{{layer_index}}}}}$"


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


def build_centered_raw_label(layer: str) -> str:
    if is_logits_label(layer):
        return r"$\{\tilde{f}_j - \langle \tilde{f}_j \rangle_{n_l}\}_{j=1...n_l}$"

    layer_index = format_layer_index(layer)
    return (
        rf"$\{{\tilde{{X}}^{layer_index}_j - \langle \tilde{{X}}^{layer_index}_j \rangle_{{n_{{{layer_index}}}}}\}}"
        rf"_{{j=1...n_{{{layer_index}}}}}$"
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


def plot_time_series(
    series: np.ndarray,
    output_path: Path,
    layer: str,
    save_loss_frequency: int,
) -> None:
    if save_loss_frequency < 1:
        raise ValueError(
            f"save_loss_frequency must be >= 1, got {save_loss_frequency}"
        )

    mean_log_series = np.mean(series, axis=1, keepdims=True)
    centered_series = series - mean_log_series
    exp_series = np.exp(series.astype(np.float64))
    exp_mean_series = np.exp(mean_log_series[:, 0])
    centered_exp_series = exp_series - exp_mean_series[:, None]
    mean_log_series = mean_log_series[:, 0]
    num_features = series.shape[1]
    label_fontsize = 16

    steps = np.arange(series.shape[0], dtype=np.int64) * save_loss_frequency
    positive_mask = steps > 0
    if not np.any(positive_mask):
        raise ValueError("Need at least one positive training step for a log-scaled x-axis.")

    steps = steps[positive_mask]
    series = series[positive_mask]
    centered_series = centered_series[positive_mask]
    exp_series = exp_series[positive_mask]
    exp_mean_series = exp_mean_series[positive_mask]
    centered_exp_series = centered_exp_series[positive_mask]
    mean_log_series = mean_log_series[positive_mask]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14.0, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.cm.tab20(np.linspace(0, 1, series.shape[1]))
    plot_specs = (
        (
            axes[0, 0],
            exp_series,
            exp_mean_series,
            build_value_label(layer),
            build_raw_mean_label(layer),
            None,
        ),
        (
            axes[0, 1],
            series,
            mean_log_series,
            build_log_label(layer),
            build_log_mean_label(layer),
            None,
        ),
        (
            axes[1, 0],
            centered_exp_series,
            None,
            build_centered_raw_label(layer),
            None,
            None,
        ),
        (
            axes[1, 1],
            centered_series,
            None,
            build_centered_label(layer),
            None,
            None,
        ),
    )

    for ax, values, mean_values, ylabel, mean_label, zero_label in plot_specs:
        for idx in range(values.shape[1]):
            ax.plot(
                steps,
                values[:, idx],
                color=colors[idx % len(colors)],
                linewidth=0.9,
                alpha=0.9,
            )
        if mean_values is not None:
            ax.plot(
                steps,
                mean_values,
                color="black",
                linewidth=1.4,
                label=mean_label,
            )
            ax.legend(fontsize=label_fontsize)
        if zero_label is not None:
            ax.axhline(0.0, color="black", linewidth=1.4, label=zero_label)
            ax.legend(fontsize=label_fontsize)
        ax.set_xscale("log", base=10)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
        ax.tick_params(axis="both", labelsize=13)

    fig.suptitle(build_title(layer, num_features), fontsize=20)
    axes[1, 0].set_xlabel(r"Training Steps $t_{i}$ [log]", fontsize=label_fontsize)
    axes[1, 1].set_xlabel(r"Training Steps $t_{i}$ [log]", fontsize=label_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot all time series stored in an xj_*.npy, "
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
    output_path = (
        args.output.resolve()
        if args.output is not None
        else repo_root / "figures_xj_terms" / f"{xj_path.stem}.png"
    )

    series = load_time_series(xj_path)
    layer = infer_layer_label(xj_path, args.layer)
    plot_time_series(series, output_path, layer, args.save_loss_frequency)


if __name__ == "__main__":
    main()
