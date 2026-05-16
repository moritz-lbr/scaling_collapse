"""Spectral analysis for saved covariance time series."""

from __future__ import annotations

import argparse
import ast
import math
import os
import pprint
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

MPL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_weight_update_similarity import (
    _build_combined_legend,
    _label_from_log_path,
    _prepare_color_map,
    _scheme_from_path,
    _width_key,
    collect_files_with_ending,
    load_yaml_as_dict,
)


LAYERS = ("Dense_0", "Dense_1")
ANALYSIS_VARIABLE = "SPECTRAL_ANALYSIS"
ANALYSIS_FILENAME = "spectral_analysis.py"
QUANTITY_NAMES = (
    "largest_eigenvalue_over_rows",
    "largest_eigenvalue_explained_variance",
    "effective_degrees_of_freedom",
)
QUANTITY_LABELS = (
    r"$\lambda_{\max}/N$",
    r"$\lambda_{\max}/\sum_i \lambda_i$",
    r"$\left(\sum_i \lambda_i\right)^2 / \sum_i \lambda_i^2$",
)


def _load_save_loss_frequency(simulation_config_path: Path) -> int:
    simulation_info = load_yaml_as_dict(simulation_config_path)
    value = simulation_info.get("training", {}).get("save_loss_frequency", 1)
    if isinstance(value, str):
        if value.strip().lower() == "epoch":
            return 1
        return int(float(value))
    return int(value)


def _safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _build_training_steps(n_snapshots: int, save_loss_frequency: int) -> np.ndarray:
    return np.arange(n_snapshots, dtype=np.int64) * int(save_loss_frequency)


def _log_selected_step_set(max_step: int) -> set[int]:
    selected: set[int] = set()
    decade = 10
    while decade <= max_step:
        for multiplier in range(1, 11):
            step = multiplier * decade
            if step <= max_step:
                selected.add(step)
        decade *= 10
    return selected


def _select_log_time_indices(
    n_snapshots: int,
    save_loss_frequency: int,
) -> Tuple[np.ndarray, np.ndarray]:
    training_steps = _build_training_steps(n_snapshots, save_loss_frequency)
    positive_steps = training_steps[training_steps > 0]
    if positive_steps.size == 0:
        raise ValueError("Need at least one positive covariance training step.")

    selected_step_set = _log_selected_step_set(int(positive_steps[-1]))
    selected_mask = np.isin(training_steps, list(selected_step_set))
    selected_indices = np.flatnonzero(selected_mask)

    if selected_indices.size == 0:
        raise ValueError(
            "No covariance snapshots matched the logarithmic selection "
            f"for save_loss_frequency={save_loss_frequency}."
        )
    return selected_indices.astype(np.int64), training_steps[selected_indices]


def _load_covariances(cov_path: Path) -> np.ndarray:
    covariances = np.load(cov_path, mmap_mode="r")
    if covariances.ndim != 3:
        raise ValueError(
            f"Expected covariance array with shape (T, N, N), got {covariances.shape}"
        )
    if covariances.shape[1] != covariances.shape[2]:
        raise ValueError(f"Expected square covariance matrices, got {covariances.shape[1:]}")
    return covariances


def _compute_layer_analysis(
    cov_path: Path,
    save_loss_frequency: int,
) -> Dict[str, Any]:
    covariances = _load_covariances(cov_path)
    selected_indices, selected_steps = _select_log_time_indices(
        covariances.shape[0],
        save_loss_frequency,
    )
    num_rows = int(covariances.shape[1])

    spectra: List[List[float]] = []
    largest_over_rows: List[float | None] = []
    explained_variance_ratio: List[float | None] = []
    effective_degrees_of_freedom: List[float | None] = []

    for index in selected_indices:
        matrix = np.asarray(covariances[int(index)], dtype=np.float64)
        eigvals = np.linalg.eigvalsh(matrix)
        eigvals = np.clip(eigvals, 0.0, None)
        eigvals_desc = eigvals[::-1]
        spectra.append([float(value) for value in eigvals_desc])

        largest = float(eigvals_desc[0]) if eigvals_desc.size else 0.0
        eig_sum = float(np.sum(eigvals))
        eig_square_sum = float(np.sum(eigvals**2))

        largest_over_rows.append(_safe_float(largest / num_rows))
        explained_variance_ratio.append(
            _safe_float(largest / eig_sum) if eig_sum > 0.0 else None
        )
        effective_degrees_of_freedom.append(
            _safe_float((eig_sum**2) / eig_square_sum) if eig_square_sum > 0.0 else None
        )

    return {
        "covariance_file": cov_path.name,
        "save_loss_frequency": int(save_loss_frequency),
        "num_rows": num_rows,
        "covariance_indices": [int(index) for index in selected_indices],
        "training_steps": [int(step) for step in selected_steps],
        "eigenvalue_spectra": spectra,
        "largest_eigenvalue_over_rows": largest_over_rows,
        "largest_eigenvalue_explained_variance": explained_variance_ratio,
        "effective_degrees_of_freedom": effective_degrees_of_freedom,
    }


def _write_analysis_file(analysis_path: Path, data: Dict[str, Any]) -> None:
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = pprint.pformat(data, sort_dicts=False, width=100)
    analysis_path.write_text(
        "# Generated by visualizations/spec_analysis.py.\n"
        "# Contains selected covariance eigenspectra and spectral time series.\n"
        f"{ANALYSIS_VARIABLE} = {rendered}\n",
        encoding="utf-8",
    )


def _read_analysis_file(analysis_path: Path) -> Dict[str, Any]:
    source = analysis_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(analysis_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        has_analysis_assignment = any(
            isinstance(target, ast.Name) and target.id == ANALYSIS_VARIABLE
            for target in node.targets
        )
        if has_analysis_assignment:
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError(f"{ANALYSIS_VARIABLE} in {analysis_path} is not a dict.")
            return value
    raise ValueError(f"Could not find {ANALYSIS_VARIABLE} in {analysis_path}.")


def _analysis_inputs_are_current(
    analysis_path: Path,
    weight_metrics_dir: Path,
    layers: Sequence[str],
) -> bool:
    analysis_mtime = analysis_path.stat().st_mtime_ns
    for layer in layers:
        cov_path = weight_metrics_dir / f"cov_{layer}.npy"
        if not cov_path.exists():
            return False
        if cov_path.stat().st_mtime_ns > analysis_mtime:
            return False
    return True


def _as_float_array(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def _plot_positive_loglog(
    ax: Any,
    ranks: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    color: Any,
) -> None:

    ax.plot(ranks[:-1], eigenvalues[:-1], color=color, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def _plot_spectrum_groups(
    layer_data: Dict[str, Any],
    output_dir: Path,
    *,
    run_label: str,
    layer: str,
    group_size: int = 10,
) -> None:
    spectra = [
        np.asarray(spectrum, dtype=float)
        for spectrum in layer_data["eigenvalue_spectra"]
    ]
    training_steps = [int(step) for step in layer_data["training_steps"]]
    if not spectra:
        return

    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    for group_index, start in enumerate(range(0, len(spectra), group_size), start=1):
        group_spectra = spectra[start : start + group_size]
        group_steps = training_steps[start : start + group_size]
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
        flat_axes = list(axes.ravel())

        for ax, spectrum, step in zip(flat_axes, group_spectra, group_steps):
            ranks = np.arange(1, len(spectrum) + 1, dtype=float)
            _plot_positive_loglog(ax, ranks, spectrum, color="black")
            ax.set_title(f"t = {step}", fontsize=11)

        for ax in flat_axes[len(group_spectra) :]:
            ax.axis("off")

        for row_ax in axes[:, 0]:
            row_ax.set_ylabel("Eigenvalue")
        for col_ax in axes[-1, :]:
            col_ax.set_xlabel("Eigenvalue rank")

        fig.suptitle(
            f"{run_label} {layer}: covariance eigenspectra, "
            f"training steps {group_steps[0]}-{group_steps[-1]}",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        file_path = spectra_dir / f"spec_analysis_{group_index:02d}.png"
        fig.savefig(file_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"Saved plot to {file_path}")


def _plot_layer_quantities(
    layer_data: Dict[str, Any],
    output_dir: Path,
    *,
    run_label: str,
    layer: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_axis = np.asarray(layer_data["training_steps"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    for ax, quantity_name, ylabel in zip(axes, QUANTITY_NAMES, QUANTITY_LABELS):
        values = _as_float_array(layer_data[quantity_name])
        ax.plot(x_axis, values, color="black", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    axes[-1].set_xlabel("Training step")
    axes[0].set_title(layer, fontsize=15)
    fig.suptitle(f"{run_label}: spectral quantities", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    file_path = output_dir / "spectral_quantities.png"
    fig.savefig(file_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved plot to {file_path}")


def _series_color_map(log_paths: Sequence[Path]) -> Dict[Path, Any]:
    scheme_groups = {
        "standard": [path for path in log_paths if _scheme_from_path(path) == "standard"],
        "muP": [path for path in log_paths if _scheme_from_path(path) == "muP"],
    }
    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}
    colors: Dict[Path, Any] = {}
    for log_path in log_paths:
        scheme = _scheme_from_path(log_path)
        colors[log_path] = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1
    return colors


def _plot_combined_quantities(
    job_dir: Path,
    log_paths: Sequence[Path],
    output_root: Path,
    layers: Sequence[str],
) -> None:
    colors = _series_color_map(log_paths)
    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    fig, axes = plt.subplots(
        3,
        len(layers),
        figsize=(7.5 * len(layers), 11),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(3, len(layers))

    for col_index, layer in enumerate(layers):
        axes[0, col_index].set_title(layer, fontsize=15)

    for log_path in log_paths:
        run_dir = log_path.parent
        run_label = _label_from_log_path(log_path)
        analysis_path = run_dir / "weight_metrics" / ANALYSIS_FILENAME
        analysis = _read_analysis_file(analysis_path)
        color = colors[log_path]
        first_line = None

        for col_index, layer in enumerate(layers):
            layer_data = analysis.get(layer)
            if not isinstance(layer_data, dict):
                continue
            x_axis = np.asarray(layer_data["training_steps"], dtype=float)
            for row_index, (quantity_name, ylabel) in enumerate(
                zip(QUANTITY_NAMES, QUANTITY_LABELS)
            ):
                ax = axes[row_index, col_index]
                values = _as_float_array(layer_data[quantity_name])
                line, = ax.plot(
                    x_axis,
                    values,
                    color=color,
                    linewidth=1.4,
                    label=run_label,
                )
                if first_line is None:
                    first_line = line
                ax.set_xscale("log")
                ax.set_ylabel(ylabel)
                ax.grid(True, which="both", alpha=0.3)
                if row_index == len(QUANTITY_NAMES) - 1:
                    ax.set_xlabel("Training step")

        if first_line is not None:
            legend_entries[_scheme_from_path(log_path)].append((first_line, run_label))

    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries,
        {"standard": "Standard parametrization", "muP": "muP parametrization"},
    )
    if combined_handles:
        legend = axes[0, -1].legend(
            combined_handles,
            combined_labels,
            loc="best",
            frameon=True,
            fontsize=11,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")

    fig.suptitle(f"{job_dir.name}: covariance spectral quantities for all runs", fontsize=17)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    output_root.mkdir(parents=True, exist_ok=True)
    file_path = output_root / "spectral_quantities_all_runs.png"
    fig.savefig(file_path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(f"Saved plot to {file_path}")


def _compute_or_load_run_analysis(
    run_dir: Path,
    layers: Iterable[str],
    *,
    force: bool,
) -> Dict[str, Any]:
    weight_metrics_dir = run_dir / "weight_metrics"
    analysis_path = weight_metrics_dir / ANALYSIS_FILENAME
    requested_layers = tuple(layers)
    analysis: Dict[str, Any] = {}
    if analysis_path.exists() and not force:
        analysis = _read_analysis_file(analysis_path)
        if all(layer in analysis for layer in requested_layers) and _analysis_inputs_are_current(
            analysis_path,
            weight_metrics_dir,
            requested_layers,
        ):
            return analysis
        print(f"Recomputing stale spectral analysis at {analysis_path}")

    simulation_config_path = run_dir / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml in {run_dir}")
    save_loss_frequency = _load_save_loss_frequency(simulation_config_path)

    if force or analysis_path.exists():
        analysis = {}

    for layer in requested_layers:
        if layer in analysis:
            continue
        cov_path = weight_metrics_dir / f"cov_{layer}.npy"
        if not cov_path.exists():
            raise FileNotFoundError(f"Missing covariance file: {cov_path}")
        print(f"Computing spectral analysis for {cov_path}")
        analysis[layer] = _compute_layer_analysis(cov_path, save_loss_frequency)

    _write_analysis_file(analysis_path, analysis)
    print(f"Saved spectral quantities to {analysis_path}")
    return analysis


def run_spec_analysis(
    job_dir: Path,
    *,
    layers: Sequence[str] = LAYERS,
    output_root: Path | None = None,
    force: bool = False,
    skip_per_run_plots: bool = False,
    skip_combined_plot: bool = False,
) -> None:
    log_paths = collect_files_with_ending(job_dir, "training_log.json")
    if not log_paths:
        raise FileNotFoundError(f"No training logs found in {job_dir}")
    log_paths = sorted(log_paths, key=lambda path: tuple(_width_key(_label_from_log_path(path))))

    if output_root is None:
        output_root = Path(__file__).resolve().parent.parent / "figures_spec_analysis"
    job_output_root = output_root / job_dir.name

    for log_path in log_paths:
        run_dir = log_path.parent
        run_label = run_dir.name
        analysis = _compute_or_load_run_analysis(run_dir, layers, force=force)

        if skip_per_run_plots:
            continue
        for layer in layers:
            layer_data = analysis.get(layer)
            if not isinstance(layer_data, dict):
                continue
            layer_output_dir = job_output_root / run_label / layer
            _plot_spectrum_groups(
                layer_data,
                layer_output_dir,
                run_label=run_label,
                layer=layer,
            )
            _plot_layer_quantities(
                layer_data,
                layer_output_dir,
                run_label=run_label,
                layer=layer,
            )

    if not skip_combined_plot:
        _plot_combined_quantities(job_dir, log_paths, job_output_root, layers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot eigenspectra for Dense covariance time series."
    )
    parser.add_argument(
        "job_dir",
        nargs="?",
        type=Path,
        help="Job directory containing run folders with training_log.json files.",
    )
    parser.add_argument(
        "--job-dir",
        dest="job_dir_flag",
        type=Path,
        default=None,
        help="Alternative way to pass the job directory.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=list(LAYERS),
        help="Layers to analyze. Defaults to Dense_0 Dense_1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for figures. Defaults to figures_spec_analysis in the repo root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Recompute {ANALYSIS_FILENAME} even if it already exists.",
    )
    parser.add_argument(
        "--skip-per-run-plots",
        action="store_true",
        help="Only compute/load spectral data and create the combined all-runs plot.",
    )
    parser.add_argument(
        "--skip-combined-plot",
        action="store_true",
        help="Only create per-run plots.",
    )
    args = parser.parse_args()

    job_dir = args.job_dir_flag or args.job_dir
    if job_dir is None:
        parser.error("Provide a job directory either positionally or with --job-dir.")

    run_spec_analysis(
        job_dir,
        layers=tuple(args.layers),
        output_root=args.output_root,
        force=args.force,
        skip_per_run_plots=args.skip_per_run_plots,
        skip_combined_plot=args.skip_combined_plot,
    )


if __name__ == "__main__":
    main()
