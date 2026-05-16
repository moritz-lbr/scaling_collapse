"""Plot averaged update-norm analysis for Dense_0 and Dense_1."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_weight_update_similarity import (
    PlotSeries,
    _build_combined_legend,
    _build_x_axis,
    _format_save_loss_frequency,
    _has_uncertainty,
    _label_from_log_path,
    _prepare_color_map,
    _scheme_from_path,
    _width_key,
    collect_files_with_ending,
    load_training_log,
    load_yaml_as_dict,
    _zeros_or_array,
)


LAYERS = ("Dense_0", "Dense_1")
DEVIATION_EPSILON = 1e-12
SIGMA_THRESHOLDS = (2.0, 3.0, 5.0)
SIGMA_THRESHOLD_MARKERS = {2.0: "o", 3.0: "s", 5.0: "^"}
SIGMA_THRESHOLD_LINE_COLORS = {2.0: "#f6c56f", 3.0: "#ee8f21", 5.0: "#bf4f00"}


def _top_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return (
        r"$\| \Delta \vec{W}_{t_{i}}^{\,"
        + layer_index
        + r"} \|^2 = \| \vec{W}_{t_{i+1}}^{\,"
        + layer_index
        + r"} - \vec{W}_{t_{i}}^{\,"
        + layer_index
        + r"}\|^2$"
    )


def _middle_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return (
        r"$\frac{\| \Delta \vec{W}_{t_{i}}^{\,"
        + layer_index
        + r"} \|^2 (n)}{c^{" + layer_index + r"}(n)}$"
    )


def _deviation_ylabel(layer: str) -> str:
    layer_index = layer[-1]
    return (
        r"$\frac{\left|\log\left(\frac{\tilde{O}_N^{\,"
        + layer_index
        + r"}(t_i)+\epsilon}{\tilde{O}_{N_{\infty}}^{\,"
        + layer_index
        + r"}(t_i)+\epsilon}\right)\right|}{\sqrt{\sigma_N^2+\sigma_{N_{\infty}}^2}},\ \epsilon=10^{-12}$"
    )


def _load_avg_final_metrics_block(log_path: Path) -> Dict[str, Any]:
    log_data = load_training_log(log_path)
    final_metrics = log_data.get("avg_final_metrics")
    if not isinstance(final_metrics, dict):
        raise ValueError(f"No avg_final_metrics found in {log_path}")
    return final_metrics


def _load_avg_training_metrics_block(weight_metrics_path: Path) -> Dict[str, Any]:
    weight_metrics = load_training_log(weight_metrics_path)
    training_metrics = weight_metrics.get("avg_training_metrics")
    if not isinstance(training_metrics, dict):
        raise ValueError(f"No avg_training_metrics found in {weight_metrics_path}")
    return training_metrics


def _find_avg_weight_metrics_path(run_dir: Path, layer: str) -> Path:
    candidate = run_dir / "avg_final_weight_metrics" / f"{layer}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing averaged weight metrics file for {layer} in {run_dir}")
    return candidate


def _dense0_width_ratio(network_info: Dict[str, Any]) -> float:
    nodes_per_layer = network_info.get("nodes_per_layer", {})
    base_widths = network_info.get("base_layer_width") or network_info.get(
        "base_layer_widths"
    )
    if not isinstance(nodes_per_layer, dict) or "Dense_0" not in nodes_per_layer:
        raise ValueError("Missing network.nodes_per_layer.Dense_0 in simulation config.")

    if isinstance(base_widths, dict):
        base_width = base_widths.get("Dense_0")
    else:
        base_width = base_widths
    if base_width is None:
        raise ValueError("Missing Dense_0 base layer width in simulation config.")

    ratio = float(nodes_per_layer["Dense_0"]) / float(base_width)
    if ratio <= 0:
        raise ValueError("Dense_0 width/base-width ratio must be positive.")
    return ratio


def _dense0_width(series: PlotSeries) -> float:
    nodes_per_layer = series.network_info.get("nodes_per_layer", {})
    if not isinstance(nodes_per_layer, dict) or "Dense_0" not in nodes_per_layer:
        raise ValueError(f"Missing Dense_0 width for run {series.label}.")
    return float(nodes_per_layer["Dense_0"])


def _largest_width_series(series_list: List[PlotSeries]) -> PlotSeries:
    if not series_list:
        raise ValueError("Cannot select the largest run from an empty series list.")
    return max(
        series_list,
        key=lambda series: (_dense0_width(series), tuple(_width_key(series.label))),
    )


def _scaled_norms(series: PlotSeries, layer: str) -> Tuple[np.ndarray, np.ndarray]:
    ratio = _dense0_width_ratio(series.network_info)
    scale = 1.0 / ratio if layer == "Dense_0" else ratio
    return series.step_norms_mean * scale, series.step_norms_std * scale


def _slice_series(
    series: PlotSeries,
    start_idx: int | None,
    stop_idx: int | None,
) -> PlotSeries:
    if start_idx is None and stop_idx is None:
        return series

    time_slice = slice(start_idx, stop_idx)
    sliced = replace(
        series,
        x_axis=series.x_axis[time_slice],
        losses_mean=series.losses_mean[time_slice],
        losses_std=series.losses_std[time_slice],
        similarities_mean=series.similarities_mean[time_slice],
        similarities_std=series.similarities_std[time_slice],
        step_norms_mean=series.step_norms_mean[time_slice],
        step_norms_std=series.step_norms_std[time_slice],
    )
    if len(sliced.step_norms_mean) == 0:
        raise ValueError(
            f"Slice {start_idx}:{stop_idx} leaves no update-norm values for {series.label}."
        )
    return sliced


def _plot_with_positive_uncertainty(
    ax: Any,
    x_axis: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: Any,
    label: str | None = None,
    shade: bool,
) -> Any:
    line, = ax.plot(x_axis, mean, color=color, label=label)
    if shade:
        lower = mean - std
        upper = mean + std
        valid = (
            np.isfinite(x_axis)
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (lower > 0.0)
            & (upper > 0.0)
        )
        if np.any(valid):
            ax.fill_between(
                x_axis,
                lower,
                upper,
                where=valid,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
    return line


def _collect_avg_plot_series(
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

    final_metrics = _load_avg_final_metrics_block(log_path)
    history = final_metrics.get("history", {})
    if not isinstance(history, dict):
        raise ValueError(f"Missing avg_final_metrics.history in {log_path}")

    train_loss = history.get("train_loss")
    if not train_loss:
        raise ValueError(f"No train_loss history found in {log_path}")

    losses_mean = np.asarray(train_loss, dtype=float)
    losses_std = _zeros_or_array(history.get("train_loss_std"), losses_mean)

    weight_metrics_path = _find_avg_weight_metrics_path(log_path.parent, layer)
    training_metrics = _load_avg_training_metrics_block(weight_metrics_path)

    step_norms = training_metrics.get("step_norms")
    step_norms_std = training_metrics.get("step_norms_std")
    similarities = training_metrics.get("similarities")
    similarities_std = training_metrics.get("similarities_std")
    if not step_norms:
        raise ValueError(f"No averaged step_norms history found in {weight_metrics_path}")
    if step_norms_std is None:
        raise ValueError(f"No averaged step_norms_std history found in {weight_metrics_path}")
    if not similarities:
        raise ValueError(f"No averaged similarities history found in {weight_metrics_path}")
    if similarities_std is None:
        raise ValueError(f"No averaged similarities_std history found in {weight_metrics_path}")

    step_norms_mean = np.asarray(step_norms, dtype=float)
    step_norms_std_array = np.asarray(step_norms_std, dtype=float)
    similarities_mean = np.asarray(similarities, dtype=float)
    similarities_std_array = np.asarray(similarities_std, dtype=float)

    x_axis, _, save_loss_frequency = _build_x_axis(
        losses_mean,
        dataset_info,
        network_info,
        compute_flag,
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
        similarities_std=similarities_std_array,
        step_norms_mean=step_norms_mean,
        step_norms_std=step_norms_std_array,
        task_name=task_name,
        dataset_info=dataset_info,
        network_info=network_info,
        save_loss_frequency=save_loss_frequency,
        num_averaged_runs=num_averaged_runs,
        has_uncertainty=any(
            (
                _has_uncertainty(losses_std),
                _has_uncertainty(similarities_std_array),
                _has_uncertainty(step_norms_std_array),
            )
        ),
    )


def _deviation_from_avg_reference(
    series: PlotSeries,
    reference: PlotSeries,
    layer: str,
) -> Tuple[np.ndarray, np.ndarray]:
    min_len = min(
        len(series.step_norms_mean),
        len(reference.step_norms_mean),
        len(series.x_axis),
    )
    x_axis = series.x_axis[:min_len]
    series_norms, series_std = _scaled_norms(series, layer)
    reference_norms, reference_std = _scaled_norms(reference, layer)
    series_norms = series_norms[:min_len]
    series_std = series_std[:min_len]
    reference_norms = reference_norms[:min_len]
    reference_std = reference_std[:min_len]

    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.abs(
            np.log(series_norms + DEVIATION_EPSILON)
            - np.log(reference_norms + DEVIATION_EPSILON)
        )
        sigma_series = np.abs(
            np.divide(
                series_std,
                series_norms,
                out=np.full(min_len, np.nan, dtype=float),
                where=series_norms != 0.0,
            )
        )
        sigma_reference = np.abs(
            np.divide(
                reference_std,
                reference_norms,
                out=np.full(min_len, np.nan, dtype=float),
                where=reference_norms != 0.0,
            )
        )
        combined_sigma = np.sqrt(sigma_series**2 + sigma_reference**2)
        normalized_deviation = np.divide(
            deviation,
            combined_sigma,
            out=np.full(min_len, np.nan, dtype=float),
            where=combined_sigma > 0.0,
        )
    normalized_deviation[~np.isfinite(normalized_deviation)] = np.nan
    return x_axis, normalized_deviation


def _first_threshold_crossing(
    x_axis: np.ndarray,
    normalized_deviation: np.ndarray,
    threshold: float,
) -> float | None:
    valid = (
        np.isfinite(x_axis)
        & np.isfinite(normalized_deviation)
        & (x_axis > 0.0)
        & (normalized_deviation > threshold)
    )
    crossing_indices = np.flatnonzero(valid)
    if crossing_indices.size == 0:
        return None
    return float(x_axis[crossing_indices[0]])


def _breakdown_times(
    x_axis: np.ndarray,
    normalized_deviation: np.ndarray,
) -> Dict[str, float | None]:
    return {
        f"{threshold:g}_sigma": _first_threshold_crossing(
            x_axis,
            normalized_deviation,
            threshold,
        )
        for threshold in SIGMA_THRESHOLDS
    }


def _breakdown_record(
    series: PlotSeries,
    color: Any,
    x_axis: np.ndarray,
    normalized_deviation: np.ndarray,
) -> Dict[str, Any]:
    color_array = np.asarray(color, dtype=float).ravel()
    return {
        "label": series.label,
        "scheme": series.scheme,
        "network_size": int(_dense0_width(series)),
        "color": [float(value) for value in color_array],
        "break_down_times": _breakdown_times(x_axis, normalized_deviation),
    }


def _fit_breakdown_scaling(
    records: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, float | int] | None:
    key = f"{threshold:g}_sigma"
    xs = []
    ys = []
    for record in records:
        break_down_time = record["break_down_times"].get(key)
        network_size = record.get("network_size")
        if break_down_time is None or network_size is None:
            continue
        if break_down_time <= 0.0 or network_size <= 0.0:
            continue
        xs.append(float(network_size))
        ys.append(float(break_down_time))

    if len(xs) < 2:
        return None

    log_x = np.log2(np.asarray(xs, dtype=float))
    log_y = np.log2(np.asarray(ys, dtype=float))
    scaling_exponent, offset = np.polyfit(log_x, log_y, deg=1)
    return {
        "offset": float(offset),
        "scaling_exponent": float(scaling_exponent),
        "num_points": len(xs),
        "network_size_min": float(min(xs)),
        "network_size_max": float(max(xs)),
    }


def _compute_breakdown_fits(breakdown_data: Dict[str, Any]) -> Dict[str, Any]:
    fits: Dict[str, Any] = {}
    for layer in LAYERS:
        records = breakdown_data["layers"].get(layer, [])
        fits[layer] = {
            f"{threshold:g}_sigma": _fit_breakdown_scaling(records, threshold)
            for threshold in SIGMA_THRESHOLDS
        }
    return fits


def _write_breakdown_times(
    job_dir: Path,
    breakdown_data: Dict[str, Any],
) -> Path:
    file_path = job_dir / "break_down_times.json"
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(breakdown_data, file, indent=2)
    return file_path


def _plot_breakdown_times(
    breakdown_data: Dict[str, Any],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fits = breakdown_data.get("fits") or _compute_breakdown_fits(breakdown_data)

    for ax, layer in zip(axes, LAYERS):
        records = breakdown_data["layers"].get(layer, [])
        network_sizes = sorted({record["network_size"] for record in records})
        fit_lines = [r"$\log_2 t = \alpha \log_2 n + \beta$"]

        for threshold in SIGMA_THRESHOLDS:
            key = f"{threshold:g}_sigma"
            marker = SIGMA_THRESHOLD_MARKERS[threshold]
            for record in records:
                break_down_time = record["break_down_times"].get(key)
                if break_down_time is None or break_down_time <= 0.0:
                    continue
                ax.scatter(
                    record["network_size"],
                    break_down_time,
                    marker=marker,
                    s=55,
                    color=record["color"],
                    edgecolors="black",
                    linewidths=0.35,
                )

            fit = fits.get(layer, {}).get(key)
            if fit is None:
                fit_lines.append(rf"$>{threshold:g}\sigma$: fit unavailable")
                continue

            x_fit = 2 ** np.linspace(
                np.log2(fit["network_size_min"]),
                np.log2(fit["network_size_max"]),
                num=200,
            )
            y_fit = (2 ** fit["offset"]) * (x_fit ** fit["scaling_exponent"])
            ax.plot(
                x_fit,
                y_fit,
                color=SIGMA_THRESHOLD_LINE_COLORS[threshold],
                linewidth=1.7,
                alpha=0.95,
            )
            fit_lines.append(
                rf"$>{threshold:g}\sigma$: "
                rf"$\alpha={fit['scaling_exponent']:.3g}$, "
                rf"$\beta={fit['offset']:.3g}$"
            )

        ax.set_title(layer, fontsize=15)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Network width " + r"$(N)$", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        if network_sizes:
            ax.set_xticks(network_sizes)
            ax.set_xticklabels([str(size) for size in network_sizes], rotation=45, ha="right")
        ax.text(
            0.03,
            0.97,
            "\n".join(fit_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.35"),
        )

    axes[0].set_ylabel("Breakdown time " + r"$t^{*}(N)$", fontsize=13)
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="-",
            marker=SIGMA_THRESHOLD_MARKERS[threshold],
            color=SIGMA_THRESHOLD_LINE_COLORS[threshold],
            markerfacecolor="white",
            markeredgecolor="black",
            label=rf"$>{threshold:g}\sigma$",
        )
        for threshold in SIGMA_THRESHOLDS
    ]
    axes[1].legend(handles=handles, loc="best", frameon=True, fontsize=11)
    fig.suptitle("Breakdown Times by Network Size", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    file_path = output_dir / "break_down_times.png"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)
    return file_path


def _plot_layer_series(
    ax_top: Any,
    ax_middle: Any,
    series: PlotSeries,
    layer: str,
    color: Any,
    *,
    label: str | None = None,
) -> Any:
    x_axis = series.x_axis[: len(series.step_norms_mean)]
    line = _plot_with_positive_uncertainty(
        ax_top,
        x_axis,
        series.step_norms_mean,
        series.step_norms_std,
        color=color,
        label=label,
        shade=_has_uncertainty(series.step_norms_std),
    )

    scaled_mean, scaled_std = _scaled_norms(series, layer)
    _plot_with_positive_uncertainty(
        ax_middle,
        x_axis,
        scaled_mean,
        scaled_std,
        color=color,
        shade=_has_uncertainty(scaled_std),
    )
    return line


def _plot_deviation_from_largest_run(
    ax: Any,
    series: PlotSeries,
    reference: PlotSeries,
    layer: str,
    color: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    x_axis, deviation = _deviation_from_avg_reference(series, reference, layer)
    ax.plot(x_axis, deviation, color=color)
    return x_axis, deviation


def plot_update_norm_analysis(
    job_dir: Path,
    compute_flag: bool,
    start_idx: int | None = None,
    stop_idx: int | None = None,
) -> None:
    log_paths = collect_files_with_ending(job_dir, "training_log.json")
    if not log_paths:
        raise FileNotFoundError(f"No training logs found in {job_dir}")

    log_paths = sorted(log_paths, key=lambda path: tuple(_width_key(_label_from_log_path(path))))
    series_pairs = [
        tuple(_collect_avg_plot_series(job_dir, log_path, layer, compute_flag) for layer in LAYERS)
        for log_path in log_paths
    ]
    series_pairs = [
        tuple(_slice_series(series, start_idx, stop_idx) for series in pair)
        for pair in series_pairs
    ]

    task_names = {series.task_name for pair in series_pairs for series in pair}
    if len(task_names) != 1:
        raise ValueError("All runs in the provided job directory must correspond to the same task.")

    first_series = series_pairs[0][0]
    dataset_info = first_series.dataset_info
    task_name = first_series.task_name
    x_label = _build_x_axis(
        first_series.losses_mean,
        first_series.dataset_info,
        first_series.network_info,
        compute_flag,
    )[1]

    scheme_groups = {
        "standard": [pair[0] for pair in series_pairs if pair[0].scheme == "standard"],
        "muP": [pair[0] for pair in series_pairs if pair[0].scheme == "muP"],
    }
    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}
    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    largest_series_by_layer = {
        "Dense_0": _largest_width_series([pair[0] for pair in series_pairs]),
        "Dense_1": _largest_width_series([pair[1] for pair in series_pairs]),
    }
    breakdown_data: Dict[str, Any] = {
        "thresholds": [float(threshold) for threshold in SIGMA_THRESHOLDS],
        "x_axis": "compute" if compute_flag else "training_steps",
        "settings": {
            "start_idx": start_idx,
            "stop_idx": stop_idx,
        },
        "layers": {layer: [] for layer in LAYERS},
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 14), sharex=True)
    axes_by_layer = {
        "Dense_0": (axes[0, 0], axes[1, 0], axes[2, 0]),
        "Dense_1": (axes[0, 1], axes[1, 1], axes[2, 1]),
    }

    for dense0_series, dense1_series in series_pairs:
        color = color_map[dense0_series.scheme][color_indices[dense0_series.scheme]]
        color_indices[dense0_series.scheme] += 1

        line = _plot_layer_series(
            *axes_by_layer["Dense_0"][:2],
            dense0_series,
            "Dense_0",
            color,
            label=dense0_series.label,
        )
        _plot_layer_series(*axes_by_layer["Dense_1"][:2], dense1_series, "Dense_1", color)
        dense0_deviation = _plot_deviation_from_largest_run(
            axes_by_layer["Dense_0"][2],
            dense0_series,
            largest_series_by_layer["Dense_0"],
            "Dense_0",
            color,
        )
        dense1_deviation = _plot_deviation_from_largest_run(
            axes_by_layer["Dense_1"][2],
            dense1_series,
            largest_series_by_layer["Dense_1"],
            "Dense_1",
            color,
        )
        breakdown_data["layers"]["Dense_0"].append(
            _breakdown_record(dense0_series, color, *dense0_deviation)
        )
        breakdown_data["layers"]["Dense_1"].append(
            _breakdown_record(dense1_series, color, *dense1_deviation)
        )
        legend_entries[dense0_series.scheme].append((line, dense0_series.label))

    for layer in LAYERS:
        ax_top, ax_middle, ax_deviation = axes_by_layer[layer]
        ax_top.set_title(layer, fontsize=20)
        ax_top.set_ylabel(_top_ylabel(layer), fontsize=18)
        ax_middle.set_ylabel(_middle_ylabel(layer), fontsize=20)
        ax_deviation.set_ylabel(_deviation_ylabel(layer), fontsize=16)
        ax_deviation.set_xlabel(x_label, fontsize=18)
        ax_deviation.text(
            0.97,
            0.95,
            rf"$n_{{\infty}}$ proxy: {largest_series_by_layer[layer].label}",
            transform=ax_deviation.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )
        for ax in (ax_top, ax_middle):
            ax.set_xscale("log", base=10)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(axis="both", labelsize=18)
        ax_deviation.set_xscale("log", base=10)
        ax_deviation.grid(True, which="both", alpha=0.3)
        ax_deviation.tick_params(axis="both", labelsize=18)

    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries,
        {"standard": "Standard parametrization", "muP": "muP parametrization"},
    )
    if combined_handles:
        legend = axes[1, 1].legend(
            combined_handles,
            combined_labels,
            loc="lower left",
            frameon=True,
            borderaxespad=0.0,
            handlelength=1.5,
            handletextpad=0.6,
            fontsize=13,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")

    averaged_counts = sorted({pair[0].num_averaged_runs for pair in series_pairs})
    info_lines = [
        f"lr = {dataset_info.get('lr')}",
        f"epochs = {dataset_info.get('epochs')}",
        f"batch size = {dataset_info.get('batch_size')}",
        (
            f"runs averaged = {averaged_counts[0]}"
            if len(averaged_counts) == 1
            else f"runs averaged = {averaged_counts[0]}-{averaged_counts[-1]}"
        ),
        "uncertainty = +/- " + r"$\sigma$",
    ]
    if start_idx is not None or stop_idx is not None:
        info_lines.append(f"time slice = {start_idx}:{stop_idx}")
    axes[0, 1].text(
        0.05,
        0.05,
        "\n".join(info_lines),
        transform=axes[0, 1].transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"),
    )

    title = (
        r"Averaged Weight Update Norms for Dense_0 and Dense_1"
        + f"\n{_format_save_loss_frequency(first_series.save_loss_frequency)} SGD Update Steps are Conducted Between Subsequent Data Points"
        + r" $t_{i+1}$ and $t_i$"
    )
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    output_dir = Path(f"figures_update_norm_analysis/{task_name}/{job_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / ("update_norm_analysis_c.png" if compute_flag else "update_norm_analysis.png")
    breakdown_data["fits"] = _compute_breakdown_fits(breakdown_data)
    breakdown_file_path = _write_breakdown_times(job_dir, breakdown_data)
    with breakdown_file_path.open("r", encoding="utf-8") as file:
        saved_breakdown_data = json.load(file)
    breakdown_plot_path = _plot_breakdown_times(saved_breakdown_data, output_dir)
    fig.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")
    print(f"Saved breakdown plot to {breakdown_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot averaged Dense_0 and Dense_1 update-norm analysis for a job."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to an averaged job directory containing training_log.json files.",
    )
    parser.add_argument(
        "--compute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot compute on the x-axis instead of training steps.",
    )
    parser.add_argument(
        "--start-idx",
        "--start_idx",
        dest="start_idx",
        type=int,
        default=None,
        help="Optional start index for slicing all evaluated time series.",
    )
    parser.add_argument(
        "--stop-idx",
        "--stop_idx",
        dest="stop_idx",
        type=int,
        default=None,
        help="Optional stop index for slicing all evaluated time series.",
    )
    args = parser.parse_args()
    plot_update_norm_analysis(args.log_dir, args.compute, args.start_idx, args.stop_idx)


if __name__ == "__main__":
    main()
