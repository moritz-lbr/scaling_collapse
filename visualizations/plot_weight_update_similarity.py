"""Plot weight-update metrics for a single raw job or a pre-averaged avg job."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_yaml_as_dict(file_path: Path) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        return {} if data is None else {"value": data}
    return data


def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches


def load_training_log(log_path: Path) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _scheme_from_path(log_path: Path) -> str:
    directory_name = log_path.parent.name
    return "standard" if "standard" in directory_name else "muP"


def _label_from_log_path(log_path: Path) -> str:
    return log_path.parent.name.rsplit("-", maxsplit=1)[-1]


def _prepare_color_map(count: int, cmap: Any) -> Iterable[Any]:
    steps = max(count, 1)
    return cmap(np.linspace(0, 1, steps))


def _width_key(label: str) -> Sequence[int]:
    direct_split: List[int] = []
    try:
        direct_split = [int(part) for part in label.split("x")]
    except ValueError:
        direct_split = []
    if direct_split:
        return direct_split

    fallback = [int(match) for match in re.findall(r"\d+", label)]
    return fallback if fallback else [10**9]


def _sorted_entries(entries: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
    return sorted(entries, key=lambda item: tuple(_width_key(item[1])))


def _build_combined_legend(
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    legend_titles: Dict[str, str],
) -> Tuple[List[Any], List[str], List[int]]:
    handles: List[Any] = []
    labels: List[str] = []
    header_indices: List[int] = []

    for scheme in ("standard", "muP"):
        sorted_linear = _sorted_entries(legend_entries[scheme])
        if not sorted_linear:
            continue
        header = plt.Line2D([], [], linestyle="", marker="", linewidth=0)
        handles.append(header)
        labels.append(legend_titles[scheme])
        header_indices.append(len(labels) - 1)
        for handle, label in sorted_linear:
            handles.append(handle)
            labels.append(label)

    return handles, labels, header_indices


def _normalize_save_loss_frequency(save_loss_frequency: Any) -> float:
    if save_loss_frequency == "epoch":
        return 1.0
    return float(save_loss_frequency)


def _build_x_axis(
    losses: np.ndarray,
    dataset_info: Dict[str, Any],
    network_info: Dict[str, Any],
    compute_flag: bool,
) -> Tuple[np.ndarray, str, float]:
    save_loss_frequency = _normalize_save_loss_frequency(
        dataset_info.get("save_loss_frequency")
    )

    if compute_flag:
        parameters = float(network_info.get("total_params"))
        batch_size = float(dataset_info.get("batch_size"))
        x_axis = (
            np.arange(len(losses), dtype=float)
            * save_loss_frequency
            * batch_size
            * parameters
        )
        x_label = r"Training Compute $c_{i}$ [log]"
    else:
        x_axis = np.arange(len(losses), dtype=float) * save_loss_frequency
        x_label = r"Training Steps $t_{i}$ [log]"

    return x_axis, x_label, save_loss_frequency


def _load_final_metrics_block(log_path: Path) -> Dict[str, Any]:
    log_data = load_training_log(log_path)
    final_metrics = log_data.get("final_metrics")
    if final_metrics is None:
        final_metrics = log_data.get("avg_final_metrics")
    if not isinstance(final_metrics, dict):
        raise ValueError(f"No final metrics found in {log_path}")
    return final_metrics


def _load_training_metrics_block(weight_metrics_path: Path) -> Dict[str, Any]:
    weight_metrics = load_training_log(weight_metrics_path)
    training_metrics = weight_metrics.get("training_metrics")
    if training_metrics is None:
        training_metrics = weight_metrics.get("avg_training_metrics")
    if not isinstance(training_metrics, dict):
        raise ValueError(f"No training metrics found in {weight_metrics_path}")
    return training_metrics


def _zeros_or_array(values: Any, reference: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros_like(reference, dtype=float)
    return np.asarray(values, dtype=float)


def _find_weight_metrics_path(run_dir: Path, layer: str) -> Path:
    for directory_name in ("weight_metrics", "avg_final_weight_metrics"):
        candidate = run_dir / directory_name / f"{layer}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing weight metrics file for {layer} in {run_dir}")


def _has_uncertainty(std_values: np.ndarray) -> bool:
    return bool(np.any(np.nan_to_num(std_values, nan=0.0) != 0.0))


def _fit_linear_regression(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float]:
    design_matrix = np.column_stack([x_values, np.ones_like(x_values)])
    slope, intercept = np.linalg.lstsq(design_matrix, y_values, rcond=None)[0]
    return float(slope), float(intercept)


def _plot_with_uncertainty(
    ax: Any,
    x_axis: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: Any,
    label: str | None = None,
    shade: bool,
    min_lower: float | None = None,
) -> Any:
    line, = ax.plot(x_axis, mean, color=color, label=label)
    if shade:
        lower = mean - std
        upper = mean + std
        if min_lower is not None:
            lower = np.clip(lower, min_lower, None)
        ax.fill_between(x_axis, lower, upper, color=color, alpha=0.18, linewidth=0)
    return line


def _ratio_from_series(
    current: "PlotSeries",
    previous: "PlotSeries",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min(
        len(current.step_norms_mean),
        len(previous.step_norms_mean),
        len(current.x_axis),
    )
    x_axis = current.x_axis[:min_len]
    current_mean = current.step_norms_mean[:min_len]
    previous_mean = previous.step_norms_mean[:min_len]
    current_std = current.step_norms_std[:min_len]
    previous_std = previous.step_norms_std[:min_len]

    with np.errstate(divide="ignore", invalid="ignore"):
        # ratio_mean = np.divide(
        #     current_mean,
        #     previous_mean,
        #     out=np.full(min_len, np.nan, dtype=float),
        #     where=previous_mean != 0,
        # )
        res = np.log(current_mean) - np.log(2.0*previous_mean)
        ratio_mean = [np.sum(res[:i+1]) for i in range(len(res))]
        current_rel_std = np.divide(
            current_std,
            current_mean,
            out=np.zeros(min_len, dtype=float),
            where=current_mean != 0,
        )
        previous_rel_std = np.divide(
            previous_std,
            previous_mean,
            out=np.zeros(min_len, dtype=float),
            where=previous_mean != 0,
        )

    ratio_std = np.abs(ratio_mean) * np.sqrt(current_rel_std**2 + previous_rel_std**2)
    return x_axis, ratio_mean, ratio_std


def _load_compute_optimal_points(job_dir: Path) -> Dict[str, np.ndarray] | None:
    compute_path = job_dir / "pareto_frontier" / "compute_optimal_points.json"
    if not compute_path.exists():
        return None

    with compute_path.open("r", encoding="utf-8") as file:
        compute_opt_data = json.load(file)
    return {
        "opt_compute": np.asarray(compute_opt_data["opt_compute"], dtype=float),
        "min_loss": np.asarray(compute_opt_data["min_loss"], dtype=float),
        "parameters": np.asarray(compute_opt_data["parameters"], dtype=float),
    }


def _format_save_loss_frequency(save_loss_frequency: float) -> str:
    if float(save_loss_frequency).is_integer():
        return str(int(save_loss_frequency))
    return str(save_loss_frequency)


@dataclass
class PlotSeries:
    scheme: str
    label: str
    x_axis: np.ndarray
    losses_mean: np.ndarray
    losses_std: np.ndarray
    similarities_mean: np.ndarray
    similarities_std: np.ndarray
    step_norms_mean: np.ndarray
    step_norms_std: np.ndarray
    task_name: str
    dataset_info: Dict[str, Any]
    network_info: Dict[str, Any]
    save_loss_frequency: float
    num_averaged_runs: int
    has_uncertainty: bool


def _collect_plot_series(
    job_dir: Path,
    log_path: Path,
    layer: str,
    compute_flag: bool,
) -> PlotSeries:
    simulation_config_path = log_path.parent / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml next to {log_path}")

    simulation_info = load_yaml_as_dict(simulation_config_path)
    dataset_info = simulation_info.get("training")
    network_info = simulation_info.get("network")
    if not isinstance(dataset_info, dict) or not isinstance(network_info, dict):
        raise ValueError(f"Incomplete simulation config in {simulation_config_path}")

    final_metrics = _load_final_metrics_block(log_path)
    history = final_metrics.get("history", {})
    if not isinstance(history, dict):
        raise ValueError(f"Missing history in {log_path}")

    test_loss = history.get("test_loss")
    if not test_loss:
        raise ValueError(f"No test loss history found in {log_path}")

    losses_mean = np.asarray(test_loss, dtype=float)
    losses_std = _zeros_or_array(history.get("test_loss_std"), losses_mean)

    weight_metrics_path = _find_weight_metrics_path(log_path.parent, layer)
    training_metrics = _load_training_metrics_block(weight_metrics_path)

    step_norms = training_metrics.get("step_norms")
    similarities = training_metrics.get("similarities")
    if not step_norms:
        raise ValueError(f"No step norms history found in {weight_metrics_path}")
    if not similarities:
        raise ValueError(f"No similarities history found in {weight_metrics_path}")

    step_norms_mean = np.asarray(step_norms, dtype=float)
    step_norms_std = _zeros_or_array(training_metrics.get("step_norms_std"), step_norms_mean)
    similarities_mean = np.asarray(similarities, dtype=float)
    similarities_std = _zeros_or_array(
        training_metrics.get("similarities_std"), similarities_mean
    )

    x_axis, _, save_loss_frequency = _build_x_axis(
        losses_mean, dataset_info, network_info, compute_flag
    )
    task_path = dataset_info.get("training_data", {}).get("task")
    task_name = Path(task_path).name if task_path else job_dir.name
    averaging_info = simulation_info.get("averaging", {})
    num_averaged_runs = int(averaging_info.get("num_files_averaged", 1))

    return PlotSeries(
        scheme=_scheme_from_path(log_path),
        label=_label_from_log_path(log_path),
        x_axis=x_axis,
        losses_mean=losses_mean,
        losses_std=losses_std,
        similarities_mean=similarities_mean,
        similarities_std=similarities_std,
        step_norms_mean=step_norms_mean,
        step_norms_std=step_norms_std,
        task_name=task_name,
        dataset_info=dataset_info,
        network_info=network_info,
        save_loss_frequency=save_loss_frequency,
        num_averaged_runs=num_averaged_runs,
        has_uncertainty=any(
            (
                _has_uncertainty(losses_std),
                _has_uncertainty(similarities_std),
                _has_uncertainty(step_norms_std),
            )
        ),
    )


def plot_weight_update_similarity(
    job_dir: Path,
    outfile: Path | None,
    layer: str,
    compute_flag: bool,
) -> None:
    log_paths = collect_files_with_ending(job_dir, "training_log.json")
    if not log_paths:
        raise FileNotFoundError(f"No training logs found in {job_dir}")

    log_paths = sorted(log_paths, key=lambda path: tuple(_width_key(_label_from_log_path(path))))
    series_list = [
        _collect_plot_series(job_dir, log_path, layer, compute_flag) for log_path in log_paths
    ]

    task_names = {series.task_name for series in series_list}
    if len(task_names) != 1:
        raise ValueError("All runs in the provided job directory must correspond to the same task.")

    scheme_groups = {
        "standard": [series for series in series_list if series.scheme == "standard"],
        "muP": [series for series in series_list if series.scheme == "muP"],
    }
    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}

    first_series = series_list[0]
    dataset_info = first_series.dataset_info
    task_name = first_series.task_name
    x_label = _build_x_axis(
        first_series.losses_mean,
        first_series.dataset_info,
        first_series.network_info,
        compute_flag,
    )[1]
    pareto_points = _load_compute_optimal_points(job_dir)
    is_average_job = job_dir.name.startswith("avg")

    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    fig, ((cos_log, loss_log), (ax_step_norms, ax_step_norms_ratio)) = plt.subplots(
        2, 2, figsize=(15, 10)
    )

    if pareto_points is not None:
        c_opt_compute = pareto_points["opt_compute"].copy()
        if not compute_flag:
            batch_size = float(dataset_info.get("batch_size"))
            c_opt_compute = c_opt_compute / (batch_size * pareto_points["parameters"])
        loss_log.scatter(c_opt_compute, pareto_points["min_loss"], color="red", label="Compute Optimal Points")

    previous_series: PlotSeries | None = None
    for series in series_list:
        color = color_map[series.scheme][color_indices[series.scheme]]
        color_indices[series.scheme] += 1

        _plot_with_uncertainty(
            loss_log,
            series.x_axis[: len(series.losses_mean)],
            series.losses_mean,
            series.losses_std,
            color=color,
            shade=series.has_uncertainty,
            min_lower=np.finfo(float).tiny,
        )
        line = _plot_with_uncertainty(
            cos_log,
            series.x_axis[: len(series.similarities_mean)],
            series.similarities_mean,
            series.similarities_std,
            color=color,
            label=series.label,
            shade=series.has_uncertainty,
        )
        _plot_with_uncertainty(
            ax_step_norms,
            series.x_axis[: len(series.step_norms_mean)],
            series.step_norms_mean,
            series.step_norms_std,
            color=color,
            shade=series.has_uncertainty,
        )
        legend_entries[series.scheme].append((line, series.label))

        if previous_series is not None:
            regression_length = min(
                len(previous_series.step_norms_mean), len(series.step_norms_mean)
            )
            if regression_length > 0:
                slope, intercept = _fit_linear_regression(
                    previous_series.step_norms_mean[:regression_length],
                    series.step_norms_mean[:regression_length],
                )
                print("Slope:", slope)
                print("Intercetp:", intercept)

            ratio_x_axis, ratio_mean, ratio_std = _ratio_from_series(series, previous_series)
            _plot_with_uncertainty(
                ax_step_norms_ratio,
                ratio_x_axis,
                ratio_mean,
                ratio_std,
                color=color,
                shade=series.has_uncertainty or previous_series.has_uncertainty,
            )

            # _plot_with_uncertainty(
            # loss_log,
            # series.x_axis[: len(series.losses_mean)],
            # series.losses_mean - previous_series.losses_mean,
            # series.losses_std,
            # color=color,
            # shade=series.has_uncertainty,
            # min_lower=np.finfo(float).tiny,
            # )

        previous_series = series

    loss_log.set_xscale("log")
    loss_log.set_yscale("log")
    loss_log.set_xlabel(x_label, fontsize=16)
    loss_log.set_ylabel("Test Loss [log]", fontsize=16)
    loss_log.grid(True, which="both", alpha=0.3)
    loss_log.tick_params(axis="both", labelsize=13)

    cos_log.set_xlabel(x_label, fontsize=16)
    cos_log.set_ylabel(
        r"$\cos(\Delta \vec{W}_{t_{i+1}}, \Delta \vec{W}_{t_{i}})$",
        fontsize=16,
    )
    cos_log.grid(True, alpha=0.3)
    cos_log.set_xscale("log", base=10)
    cos_log.tick_params(axis="both", labelsize=13)

    ax_step_norms.set_xlabel(x_label, fontsize=16)
    ax_step_norms.set_ylabel(r"$\| \Delta \vec{W}_{t_{i}}^{\," + layer[-1] + r"} \|^2 = \| \vec{W}_{t_{i+1}}^{\," + layer[-1] + r"} - \vec{W}_{t_{i}}^{\," + layer[-1] + r"}\|^2$", fontsize=16)
    ax_step_norms.grid(True, alpha=0.3)
    ax_step_norms.set_xscale("log", base=10)
    # ax_step_norms.set_yscale("log")
    ax_step_norms.tick_params(axis="both", labelsize=13)

    ax_step_norms_ratio.set_xlabel(x_label, fontsize=16)
    ax_step_norms_ratio.set_ylabel(r"$R(t)$", fontsize=16)
    ax_step_norms_ratio.grid(True, alpha=0.3)
    ax_step_norms_ratio.set_xscale("log", base=10)
    ax_step_norms_ratio.tick_params(axis="both", labelsize=13)
    # ax_step_norms_ratio.set_yscale("log")
    ax_step_norms_ratio.set_ylim(-5,5)

    legend_titles = {
        "standard": "Standard parametrization",
        "muP": "muP parametrization",
    }
    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries, legend_titles
    )
    if combined_handles:
        legend = ax_step_norms.legend(
            combined_handles,
            combined_labels,
            loc="upper right",
            frameon=True,
            borderaxespad=0.0,
            handlelength=1.5,
            handletextpad=0.6,
            fontsize=13,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")

    info_lines = [
        f"lr = {dataset_info.get('lr')}",
        f"epochs = {dataset_info.get('epochs')}",
        f"batch size = {dataset_info.get('batch_size')}",
    ]
    if is_average_job:
        averaging_counts = sorted({series.num_averaged_runs for series in series_list})
        if len(averaging_counts) == 1:
            info_lines.append(f"runs averaged = {averaging_counts[0]}")
        else:
            info_lines.append(
                f"runs averaged = {averaging_counts[0]}-{averaging_counts[-1]}"
            )
        info_lines.append("uncertainty = +/- " + r"$\sigma$")

    loss_log.text(
        0.1,
        0.1,
        "\n".join(info_lines),
        transform=loss_log.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"),
    )

    if layer == "all_weights":
        layer_label = "All Weights of The Network"
    else:
        layer_label = f"The Weights in Layer {layer}"

    title = (
        r"$\vec{\theta}_{t}$ Describes "
        + f"{layer_label}"
        + f"\n {_format_save_loss_frequency(first_series.save_loss_frequency)} SGD Update Steps are Conducted Between Subsequent Data Points"
        + r" $t_{i+1}$ and $t_{i}$"
    )
    if is_average_job:
        title += "\n Pre-averaged job; shaded regions show +/- " + r"$\sigma$"
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    prefix = Path(f"figures_weight_update_similarity/{task_name}/{job_dir.name}")
    prefix.mkdir(parents=True, exist_ok=True)

    compute_appendix = "_c" if compute_flag else ""
    if outfile:
        file_path = prefix / (str(outfile) + compute_appendix)
    else:
        file_path = prefix / (f"weight_metrics_{layer}" + compute_appendix + ".png")

    if file_path.suffix == "":
        file_path = file_path.with_suffix(".png")

    fig.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weight-update metrics for a single raw job or pre-averaged avg job."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to a single job directory containing training_log.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional filename to save the plot under figures_weight_update_similarity/.",
        default=None,
    )
    parser.add_argument(
        "--output-group",
        type=str,
        default=None,
        help="Deprecated compatibility flag. Plot output is grouped by the job directory name.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="all_weights",
        help="Layer to plot, for example all_weights or Dense_0.",
    )
    parser.add_argument(
        "--compute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot compute on the x-axis instead of training steps.",
    )

    args = parser.parse_args()
    plot_weight_update_similarity(args.log_dir, args.output, args.layer, args.compute)


if __name__ == "__main__":
    main()
