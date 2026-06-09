"""Plot and fit self-averaging statistics across widths."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


MPL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training_analysis.mb_cov_utils import collect_files_with_ending, load_yaml_as_dict
from visualizations.plot_weight_update_similarity import (
    _build_combined_legend as _build_update_norm_legend,
    _prepare_color_map as _prepare_update_norm_color_map,
    _scheme_from_path as _scheme_from_update_norm_path,
)
from visualizations.self_averaging_metrics import (
    DEFAULT_WINDOW_STRIDE,
    DEFAULT_WINDOW_WIDTH,
    LAYERS,
    OBSERVABLE_TEST_LOSS,
    OBSERVABLE_UPDATE_NORM,
    OBSERVABLES,
    OUTPUT_FILENAMES,
    TEST_LOSS_COMPONENT,
    compute_self_averaging_metrics,
    dense0_width_ratio_from_config,
    load_step_norms,
    load_test_loss,
    rescale_update_norms,
    self_averaging_stats,
)


QUANTITIES = ("R", "variance", "squared_first_moment")
STAT_QUANTITIES = ("R", "variance", "squared_first_moment", "mean", "sample_count")
QUANTITY_LABELS = {
    "R": r"$R = \mathrm{Var}(X_j) / \mathbb{E}(X_j)^2$",
    "variance": r"$\mathrm{Var}(X_j)$",
    "squared_first_moment": r"$\mathbb{E}(X_j)^2$",
}
QUANTITY_LOG_LABELS = {
    "R": r"$\log_2(R)$",
    "variance": r"$\log_2(\mathrm{Var}(X_j))$",
    "squared_first_moment": r"$\log_2(\mathbb{E}(X_j)^2)$",
}
QUANTITY_FIT_MODELS = {
    "R": "log2(Var(X_j) / E(X_j)^2) = alpha(t) * log2(N) + intercept_layer",
    "variance": "log2(Var(X_j)) = gamma(t) * log2(N) + intercept_layer",
    "squared_first_moment": (
        "log2(E(X_j)^2) = omega(t) * log2(N) + intercept_layer"
    ),
}
QUANTITY_EXPONENT_SYMBOLS = {
    "R": r"\alpha",
    "variance": r"\gamma",
    "squared_first_moment": r"\omega",
}
QUANTITY_EXPONENT_NAMES = {
    "R": "alpha(t)",
    "variance": "gamma(t)",
    "squared_first_moment": "omega(t)",
}
DEFAULT_YLIM_QUANTILE_LOW = 0.01
DEFAULT_YLIM_QUANTILE_HIGH = 0.99
DEFAULT_RESIDUAL_RANDOM_SEED = 0


@dataclass(frozen=True)
class RunSelfAveraging:
    run_dir: Path
    label: str
    scheme: str
    task_name: str
    hidden_width: int
    steps_by_component: Dict[str, np.ndarray]
    stats_by_component: Dict[str, Dict[str, np.ndarray]]


@dataclass(frozen=True)
class ScalarObservableRecord:
    job_dir: Path
    run_dir: Path
    label: str
    scheme: str
    group_key: str
    task_name: str
    hidden_width: int
    steps_by_component: Dict[str, np.ndarray]
    values_by_component: Dict[str, np.ndarray]


def _components_for_observable(observable: str) -> Tuple[str, ...]:
    if observable == OBSERVABLE_UPDATE_NORM:
        return LAYERS
    if observable == OBSERVABLE_TEST_LOSS:
        return (TEST_LOSS_COMPONENT,)
    raise ValueError(f"Unsupported observable {observable!r}.")


def _label_from_run_name(run_name: str) -> str:
    return run_name.rsplit("-", maxsplit=1)[-1]


def _scheme_from_run_dir(run_dir: Path) -> str:
    return _scheme_from_update_norm_path(run_dir / "training_log.json")


def _width_key(label: str) -> Sequence[int]:
    try:
        direct = [int(part) for part in label.split("x")]
    except ValueError:
        direct = []
    if direct:
        return direct
    fallback = [int(match) for match in re.findall(r"\d+", label)]
    return fallback if fallback else [10**9]


def _sorted_training_logs(job_dir: Path) -> List[Path]:
    log_paths = collect_files_with_ending(job_dir, "training_log.json")
    return sorted(log_paths, key=lambda path: tuple(_width_key(_label_from_run_name(path.parent.name))))


def _job_sort_key(job_dir: Path) -> Tuple[int, str]:
    match = re.search(r"job-(\d+)", job_dir.name)
    if match:
        return int(match.group(1)), job_dir.name
    return 10**18, job_dir.name


def _combined_job_name(job_dirs: Sequence[Path]) -> str:
    names = [job_dir.name for job_dir in sorted(job_dirs, key=_job_sort_key)]
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return "_".join(names)
    return f"{names[0]}_to_{names[-1]}_{len(names)}jobs"


def _combined_fit_root(job_dirs: Sequence[Path], combined_name: str) -> Path:
    if len(job_dirs) == 1:
        return job_dirs[0]
    common_parent = Path(os.path.commonpath([str(job_dir.parent.resolve()) for job_dir in job_dirs]))
    return common_parent / combined_name


def _simulation_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "simulation_config.yaml"
    if not path.exists():
        return {}
    return load_yaml_as_dict(path)


def _task_name_from_config(simulation_config: Dict[str, Any], fallback: str) -> str:
    task = (
        simulation_config.get("training", {})
        .get("training_data", {})
        .get("task")
    )
    if task:
        return Path(str(task)).name
    return fallback


def _hidden_width_from_config(simulation_config: Dict[str, Any]) -> int | None:
    network_info = simulation_config.get("network", {})
    for key in ("nodes_per_layer", "base_layer_width", "base_layer_widths"):
        widths = network_info.get(key)
        if isinstance(widths, dict) and "Dense_0" in widths:
            return int(widths["Dense_0"])
    return None


def _save_loss_frequency_from_config(simulation_config: Dict[str, Any]) -> int:
    value = simulation_config.get("training", {}).get("save_loss_frequency", 1)
    if isinstance(value, str):
        if value.strip().lower() == "epoch":
            return 1
        return int(float(value))
    return int(value)


def _training_steps_for_update_norms(num_values: int, save_loss_frequency: int) -> np.ndarray:
    return np.arange(1, num_values + 1, dtype=np.int64) * int(save_loss_frequency)


def _training_steps_for_logged_losses(num_values: int, save_loss_frequency: int) -> np.ndarray:
    return np.arange(num_values, dtype=np.int64) * int(save_loss_frequency)


def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _validate_ylim_quantiles(low: float, high: float) -> None:
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(
            f"Y-limit quantiles must satisfy 0 <= low < high <= 1, got {low}, {high}."
        )


def _quantile_limits(
    values: Sequence[Any],
    low: float,
    high: float,
    *,
    positive_only: bool = False,
) -> Tuple[float, float] | None:
    _validate_ylim_quantiles(low, high)
    values_array = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values_array[np.isfinite(values_array)]
    if positive_only:
        finite_values = finite_values[finite_values > 0.0]
    if finite_values.size == 0:
        return None

    lower = float(np.quantile(finite_values, low))
    upper = float(np.quantile(finite_values, high))
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    if lower == upper:
        pad = max(abs(lower) * 0.1, 1e-12)
        if positive_only:
            return max(lower - pad, np.nextafter(0.0, 1.0)), upper + pad
        return lower - pad, upper + pad
    if positive_only and lower > 0.0 and upper > 0.0:
        return max(0.5 * lower, np.nextafter(0.0, 1.0)), 1.5 * upper

    span = upper - lower
    return lower - 0.1 * span, upper + 0.1 * span


def _apply_quantile_ylim(
    ax: Any,
    values: Sequence[Any],
    low: float,
    high: float,
    *,
    positive_only: bool = False,
) -> None:
    limits = _quantile_limits(values, low, high, positive_only=positive_only)
    if limits is not None and limits[0] < limits[1]:
        ax.set_ylim(*limits)


def _read_run_metrics(run_dir: Path, metrics_name: str) -> Dict[str, Any]:
    metrics_path = run_dir / "weight_metrics" / metrics_name
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing self-averaging metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _metrics_match_settings(
    metrics: Dict[str, Any],
    *,
    observable: str,
    window_width: int,
    window_stride: int,
) -> bool:
    settings_match = (
        metrics.get("observable") == observable
        and metrics.get("estimator") == "local_time_window"
        and int(metrics.get("window_width", -1)) == int(window_width)
        and int(metrics.get("window_stride", -1)) == int(window_stride)
    )
    if not settings_match:
        return False
    if observable == OBSERVABLE_UPDATE_NORM:
        rescaling = metrics.get("rescaling", {})
        if not isinstance(rescaling, dict) or rescaling.get("mode") != "dense0_width_ratio":
            return False

    try:
        block = _metric_block(metrics)
    except ValueError:
        return False
    for component in _components_for_observable(observable):
        component_data = block.get(component)
        if isinstance(component_data, dict) and "squared_first_moment" not in component_data:
            return False
    return True


def _metric_block(metrics: Dict[str, Any]) -> Dict[str, Any]:
    block = metrics.get("series")
    if isinstance(block, dict):
        return block
    block = metrics.get("layers")
    if isinstance(block, dict):
        return block
    raise ValueError("Self-averaging metrics file has neither a series nor layers block.")


def _collect_run_series(
    log_path: Path,
    *,
    observable: str,
    metrics_name: str,
    compute_missing: bool,
    force_metrics: bool,
    window_width: int,
    window_stride: int,
) -> RunSelfAveraging:
    run_dir = log_path.parent
    metrics_path = run_dir / "weight_metrics" / metrics_name
    should_compute = force_metrics or (compute_missing and not metrics_path.exists())
    if should_compute:
        compute_self_averaging_metrics(
            run_dir,
            observable=observable,
            output_name=metrics_name,
            window_width=window_width,
            window_stride=window_stride,
        )

    metrics = _read_run_metrics(run_dir, metrics_name)
    if not _metrics_match_settings(
        metrics,
        observable=observable,
        window_width=window_width,
        window_stride=window_stride,
    ):
        if compute_missing or force_metrics:
            compute_self_averaging_metrics(
                run_dir,
                observable=observable,
                output_name=metrics_name,
                window_width=window_width,
                window_stride=window_stride,
            )
            metrics = _read_run_metrics(run_dir, metrics_name)
        else:
            raise ValueError(
                f"{metrics_path} was not computed for observable={observable} with "
                f"window_width={window_width} and window_stride={window_stride}. "
                "Recompute it or pass --force-metrics."
            )

    simulation_config = _simulation_config(run_dir)
    hidden_width = metrics.get("hidden_width")
    if hidden_width is None:
        hidden_width = _hidden_width_from_config(simulation_config)
    if hidden_width is None:
        hidden_width = _width_key(_label_from_run_name(run_dir.name))[0]

    steps_by_component: Dict[str, np.ndarray] = {}
    stats_by_component: Dict[str, Dict[str, np.ndarray]] = {}
    block = _metric_block(metrics)
    for component in _components_for_observable(observable):
        component_data = block.get(component)
        if not isinstance(component_data, dict):
            continue
        steps_by_component[component] = np.asarray(
            component_data.get("training_steps", []),
            dtype=np.int64,
        )
        stats_by_component[component] = {
            quantity: _as_float_array(component_data.get(quantity, []))
            for quantity in STAT_QUANTITIES
        }
        for quantity in STAT_QUANTITIES:
            if steps_by_component[component].shape[0] != stats_by_component[component][quantity].shape[0]:
                raise ValueError(
                    f"Step/{quantity} length mismatch in {metrics_path} for {component}: "
                    f"{steps_by_component[component].shape[0]} vs "
                    f"{stats_by_component[component][quantity].shape[0]}."
                )

    return RunSelfAveraging(
        run_dir=run_dir,
        label=_label_from_run_name(run_dir.name),
        scheme=_scheme_from_run_dir(run_dir),
        task_name=_task_name_from_config(simulation_config, log_path.parent.parent.name),
        hidden_width=int(hidden_width),
        steps_by_component=steps_by_component,
        stats_by_component=stats_by_component,
    )


def _collect_single_job_series(
    job_dir: Path,
    *,
    observable: str,
    metrics_name: str,
    compute_missing: bool,
    force_metrics: bool,
    window_width: int,
    window_stride: int,
) -> List[RunSelfAveraging]:
    log_paths = _sorted_training_logs(job_dir)
    if not log_paths:
        raise FileNotFoundError(f"No training_log.json files found under {job_dir}")
    return [
        _collect_run_series(
            log_path,
            observable=observable,
            metrics_name=metrics_name,
            compute_missing=compute_missing,
            force_metrics=force_metrics,
            window_width=window_width,
            window_stride=window_stride,
        )
        for log_path in log_paths
    ]


def _raw_component_values(
    run_dir: Path,
    observable: str,
    save_loss_frequency: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if observable == OBSERVABLE_UPDATE_NORM:
        dense0_width_ratio = dense0_width_ratio_from_config(
            run_dir / "simulation_config.yaml"
        )
        values_by_component: Dict[str, np.ndarray] = {}
        steps_by_component: Dict[str, np.ndarray] = {}
        for layer in LAYERS:
            values, _ = load_step_norms(run_dir, layer)
            values = rescale_update_norms(values, layer, dense0_width_ratio)
            values_by_component[layer] = values
            steps_by_component[layer] = _training_steps_for_update_norms(
                int(values.shape[0]),
                save_loss_frequency,
            )
        return values_by_component, steps_by_component

    if observable == OBSERVABLE_TEST_LOSS:
        values, _ = load_test_loss(run_dir)
        return {
            TEST_LOSS_COMPONENT: values
        }, {
            TEST_LOSS_COMPONENT: _training_steps_for_logged_losses(
                int(values.shape[0]),
                save_loss_frequency,
            )
        }

    raise ValueError(f"Unsupported observable {observable!r}.")


def _collect_observable_record(job_dir: Path, log_path: Path, observable: str) -> ScalarObservableRecord:
    run_dir = log_path.parent
    simulation_config = _simulation_config(run_dir)
    hidden_width = _hidden_width_from_config(simulation_config)
    if hidden_width is None:
        hidden_width = _width_key(_label_from_run_name(run_dir.name))[0]
    save_loss_frequency = _save_loss_frequency_from_config(simulation_config)
    values_by_component, steps_by_component = _raw_component_values(
        run_dir,
        observable,
        save_loss_frequency,
    )

    return ScalarObservableRecord(
        job_dir=job_dir,
        run_dir=run_dir,
        label=_label_from_run_name(run_dir.name),
        scheme=_scheme_from_run_dir(run_dir),
        group_key=run_dir.name,
        task_name=_task_name_from_config(simulation_config, job_dir.name),
        hidden_width=int(hidden_width),
        steps_by_component=steps_by_component,
        values_by_component=values_by_component,
    )


def _collect_observable_records_by_group(
    job_dirs: Sequence[Path],
    observable: str,
) -> Dict[str, List[ScalarObservableRecord]]:
    records_by_group: Dict[str, List[ScalarObservableRecord]] = {}
    for job_dir in job_dirs:
        log_paths = _sorted_training_logs(job_dir)
        if not log_paths:
            raise FileNotFoundError(f"No training_log.json files found under {job_dir}")
        for log_path in log_paths:
            record = _collect_observable_record(job_dir, log_path, observable)
            records_by_group.setdefault(record.group_key, []).append(record)
    return records_by_group


def _collect_cross_job_series(job_dirs: Sequence[Path], observable: str) -> List[RunSelfAveraging]:
    records_by_group = _collect_observable_records_by_group(job_dirs, observable)

    series: List[RunSelfAveraging] = []
    for group_key, records in records_by_group.items():
        reference = records[0]
        hidden_widths = {record.hidden_width for record in records}
        if len(hidden_widths) != 1:
            raise ValueError(f"Matched group {group_key} has inconsistent hidden widths: {hidden_widths}")

        steps_by_component: Dict[str, np.ndarray] = {}
        stats_by_component: Dict[str, Dict[str, np.ndarray]] = {}
        for component in _components_for_observable(observable):
            values_by_step: Dict[int, List[float]] = {}
            for record in records:
                steps = record.steps_by_component[component]
                values = record.values_by_component[component]
                for step, value in zip(steps, values):
                    if math.isfinite(float(value)):
                        values_by_step.setdefault(int(step), []).append(float(value))

            valid_steps: List[int] = []
            stats = {quantity: [] for quantity in STAT_QUANTITIES}
            for step in sorted(values_by_step):
                sample_count = int(len(values_by_step[step]))
                ratio, variance, squared_first_moment, mean = self_averaging_stats(
                    values_by_step[step]
                )
                if (
                    ratio is None
                    or variance is None
                    or squared_first_moment is None
                    or mean is None
                ):
                    continue
                valid_steps.append(int(step))
                stats["R"].append(float(ratio))
                stats["variance"].append(float(variance))
                stats["squared_first_moment"].append(float(squared_first_moment))
                stats["mean"].append(float(mean))
                stats["sample_count"].append(sample_count)

            steps_by_component[component] = np.asarray(valid_steps, dtype=np.int64)
            stats_by_component[component] = {
                quantity: _as_float_array(stats[quantity]) for quantity in STAT_QUANTITIES
            }

        series.append(
            RunSelfAveraging(
                run_dir=reference.run_dir,
                label=reference.label,
                scheme=reference.scheme,
                task_name=reference.task_name,
                hidden_width=reference.hidden_width,
                steps_by_component=steps_by_component,
                stats_by_component=stats_by_component,
            )
        )

    return sorted(series, key=lambda run: tuple(_width_key(run.label)))


def _all_training_steps(series: Sequence[RunSelfAveraging], components: Sequence[str]) -> List[int]:
    steps: set[int] = set()
    for run in series:
        for component in components:
            steps.update(int(step) for step in run.steps_by_component.get(component, []))
    return sorted(steps)


def _value_at_step(
    run: RunSelfAveraging,
    component: str,
    quantity: str,
    step: int,
) -> float | None:
    steps = run.steps_by_component.get(component)
    stats = run.stats_by_component.get(component, {})
    values = stats.get(quantity)
    if steps is None or values is None:
        return None
    matches = np.flatnonzero(steps == int(step))
    if matches.size == 0:
        return None
    value = float(values[int(matches[0])])
    return value if math.isfinite(value) else None


def _log_variance_std(sample_count: float | None, *, log_base: float = 2.0) -> float | None:
    if sample_count is None:
        return None
    count = float(sample_count)
    if not math.isfinite(count) or count <= 1.0:
        return None
    return math.sqrt(2.0 / (count - 1.0)) / math.log(float(log_base))


def _log_variance_stds(sample_counts: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [
            np.nan if (std := _log_variance_std(sample_count)) is None else std
            for sample_count in sample_counts
        ],
        dtype=float,
    )


def _largest_width_run(series: Sequence[RunSelfAveraging]) -> RunSelfAveraging:
    if not series:
        raise ValueError("Cannot select a largest-width run from an empty series.")
    return max(series, key=lambda run: (run.hidden_width, tuple(_width_key(run.label))))


def _apply_classical_r_definition(
    series: Sequence[RunSelfAveraging],
    components: Sequence[str],
) -> None:
    for run in series:
        for component in components:
            stats = run.stats_by_component.get(component, {})
            variances = stats.get("variance")
            squared_first_moments = stats.get("squared_first_moment")
            if variances is None or squared_first_moments is None:
                continue

            classical_r: List[float] = []
            for variance, squared_first_moment in zip(variances, squared_first_moments):
                variance_float = float(variance)
                squared_first_moment_float = float(squared_first_moment)
                if (
                    squared_first_moment_float <= 0.0
                    or not math.isfinite(variance_float)
                    or not math.isfinite(squared_first_moment_float)
                ):
                    classical_r.append(float("nan"))
                    continue
                classical_r.append(variance_float / squared_first_moment_float)
            stats["R"] = np.asarray(classical_r, dtype=float)


def _write_single_job_classical_r_definition(
    series: Sequence[RunSelfAveraging],
    components: Sequence[str],
    metrics_name: str,
) -> None:
    for run in series:
        metrics_path = run.run_dir / "weight_metrics" / metrics_name
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        blocks: List[Dict[str, Any]] = []
        for block_name in ("series", "layers"):
            block = metrics.get(block_name)
            if isinstance(block, dict):
                blocks.append(block)
        if not blocks:
            blocks.append(_metric_block(metrics))

        for block in blocks:
            for component in components:
                component_data = block.get(component)
                stats = run.stats_by_component.get(component, {})
                if not isinstance(component_data, dict):
                    continue
                r_values = stats.get("R")
                if r_values is not None:
                    component_data["R"] = [
                        _none_if_not_finite(float(value))
                        for value in np.asarray(r_values, dtype=float)
                    ]
        metrics.pop("R_denominator", None)
        metrics["definition"] = (
            "For a single run, X_j are scalar observable values in a local time window "
            "around the reported time step. R = Var_j(X_j) / E_j[X_j]^2."
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _points_for_step(
    series: Sequence[RunSelfAveraging],
    component: str,
    quantity: str,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
    widths: List[float] = []
    values: List[float] = []
    labels: List[str] = []
    run_keys: List[str] = []
    sample_counts: List[float] = []
    for run in series:
        value = _value_at_step(run, component, quantity, step)
        if value is None or value <= 0.0 or run.hidden_width <= 0:
            continue
        widths.append(float(run.hidden_width))
        values.append(float(value))
        labels.append(run.label)
        run_keys.append(_run_key(run))
        sample_count = _value_at_step(run, component, "sample_count", step)
        sample_counts.append(float("nan") if sample_count is None else float(sample_count))
    return (
        np.asarray(widths, dtype=float),
        np.asarray(values, dtype=float),
        labels,
        run_keys,
        np.asarray(sample_counts, dtype=float),
    )


def _run_key(run: RunSelfAveraging) -> str:
    return str(run.run_dir.resolve())


def _run_color_map(
    series: Sequence[RunSelfAveraging],
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[Any, str]]]]:
    runs_by_scheme: Dict[str, List[RunSelfAveraging]] = {"standard": [], "muP": []}
    for run in series:
        runs_by_scheme.setdefault(run.scheme, []).append(run)

    cmap_by_scheme = {
        "standard": plt.cm.autumn,
        "muP": plt.cm.winter,
    }
    colors_by_run: Dict[str, Any] = {}
    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    for scheme in ("standard", "muP"):
        ordered_runs = sorted(
            runs_by_scheme.get(scheme, []),
            key=lambda run: tuple(_width_key(run.label)),
        )
        colors = _prepare_update_norm_color_map(
            len(ordered_runs),
            cmap_by_scheme[scheme],
        )
        for run, color in zip(ordered_runs, colors):
            colors_by_run[_run_key(run)] = color
            handle = plt.Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.35,
            )
            legend_entries[scheme].append((handle, run.label))

    return colors_by_run, legend_entries


def _add_update_norm_legend(
    target: Any,
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    *,
    loc: str,
    fontsize: int,
    bbox_to_anchor: Tuple[float, float] | None = None,
) -> Any | None:
    handles, labels, header_indices = _build_update_norm_legend(
        legend_entries,
        {"standard": "Standard parametrization", "muP": "muP parametrization"},
    )
    if not handles:
        return None
    kwargs: Dict[str, Any] = {
        "loc": loc,
        "frameon": True,
        "borderaxespad": 0.0,
        "handlelength": 1.5,
        "handletextpad": 0.6,
        "fontsize": fontsize,
    }
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = target.legend(handles, labels, **kwargs)
    legend._legend_box.align = "left"  # type: ignore[attr-defined]
    for index in header_indices:
        legend.get_texts()[index].set_fontweight("bold")
    return legend


def _fit_log2_power_law(
    widths: np.ndarray,
    values: np.ndarray,
    *,
    y_errors: np.ndarray | None = None,
) -> Dict[str, float | int | None]:
    valid = np.isfinite(widths) & np.isfinite(values) & (widths > 0.0) & (values > 0.0)
    if y_errors is not None:
        y_errors = np.asarray(y_errors, dtype=float).reshape(-1)
        if y_errors.shape[0] != valid.shape[0]:
            raise ValueError(
                f"y_errors length mismatch: {y_errors.shape[0]} vs {valid.shape[0]}"
            )
    widths = widths[valid]
    values = values[valid]
    errors = y_errors[valid] if y_errors is not None else None
    if widths.size < 2 or np.unique(widths).size < 2:
        return {
            "alpha_layer": None,
            "intercept_layer": None,
            "r_squared": None,
            "chi_squared": None,
            "degrees_of_freedom": None,
            "chi_squared_per_dof": None,
            "num_fit_points": int(widths.size),
        }

    x = np.log2(widths)
    y = np.log2(values)
    fit_x = x
    fit_y = y
    fit_errors = None
    if errors is not None:
        error_valid = np.isfinite(errors) & (errors > 0.0)
        if np.count_nonzero(error_valid) >= 2 and np.unique(x[error_valid]).size >= 2:
            fit_x = x[error_valid]
            fit_y = y[error_valid]
            fit_errors = errors[error_valid]

    if fit_errors is None:
        alpha, intercept = np.polyfit(fit_x, fit_y, deg=1)
    else:
        alpha, intercept = np.polyfit(fit_x, fit_y, deg=1, w=1.0 / fit_errors)
    predicted = float(alpha) * fit_x + float(intercept)
    ss_res = float(np.sum((fit_y - predicted) ** 2))
    ss_tot = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else None
    chi_squared = None
    degrees_of_freedom = None
    chi_squared_per_dof = None
    if fit_errors is not None:
        residuals = fit_y - predicted
        chi_squared = float(np.sum((residuals / fit_errors) ** 2))
        degrees_of_freedom = int(fit_y.size - 2)
        if degrees_of_freedom > 0:
            chi_squared_per_dof = chi_squared / float(degrees_of_freedom)
    return {
        "alpha_layer": float(alpha),
        "intercept_layer": float(intercept),
        "r_squared": r_squared,
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "chi_squared_per_dof": chi_squared_per_dof,
        "num_fit_points": int(fit_y.size),
    }


def _plot_step(
    series: Sequence[RunSelfAveraging],
    components: Sequence[str],
    quantity: str,
    step: int,
    output_dir: Path,
    fit_for_step: Dict[str, Dict[str, Any]],
    *,
    dpi: int,
    min_points: int,
    file_prefix: str,
    colors_by_run: Dict[str, Any],
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> None:
    exponent_symbol = _quantity_exponent_symbol(quantity)
    fig, axes = plt.subplots(
        1,
        len(components),
        figsize=(5.8 * len(components), 4.8),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for ax, component in zip(axes.ravel(), components):
        widths, values, labels, run_keys, sample_counts = _points_for_step(
            series,
            component,
            quantity,
            step,
        )
        y_values_for_limits: List[float] = []
        if widths.size:
            x = np.log2(widths)
            y = np.log2(values)
            y_values_for_limits.extend(float(value) for value in y if math.isfinite(float(value)))
            colors = [colors_by_run.get(run_key, "black") for run_key in run_keys]
            if quantity == "variance":
                y_errors = _log_variance_stds(sample_counts)
                for x_value, y_value, y_error, color in zip(x, y, y_errors, colors):
                    ax.errorbar(
                        x_value,
                        y_value,
                        yerr=None if not math.isfinite(float(y_error)) else float(y_error),
                        fmt="o",
                        markersize=5.2,
                        color=color,
                        ecolor=color,
                        elinewidth=0.9,
                        capsize=2.5,
                        alpha=0.9,
                        markeredgecolor="black",
                        markeredgewidth=0.35,
                    )
            else:
                ax.scatter(
                    x,
                    y,
                    s=34,
                    color=colors,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.35,
                )
            for x_value, y_value, label in zip(x, y, labels):
                ax.annotate(
                    label,
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                )
        fit = fit_for_step[component]
        alpha = fit.get("alpha_layer")
        intercept = fit.get("intercept_layer")
        if alpha is not None and intercept is not None and widths.size >= min_points:
            x_fit = np.linspace(
                float(np.min(np.log2(widths))),
                float(np.max(np.log2(widths))),
                100,
            )
            y_fit = float(alpha) * x_fit + float(intercept)
            ax.plot(x_fit, y_fit, color="tab:red", linewidth=1.4)
            y_values_for_limits.extend(
                float(value) for value in y_fit if math.isfinite(float(value))
            )
            text = rf"${exponent_symbol}={float(alpha):.3g}$"
            chi_per_dof = fit.get("chi_squared_per_dof")
            degrees_of_freedom = fit.get("degrees_of_freedom")
            if quantity == "variance" and chi_per_dof is not None:
                text += "\n" + (
                    rf"$\chi^2/\nu={float(chi_per_dof):.3g}$"
                    rf" $(\nu={int(degrees_of_freedom)})$"
                    if degrees_of_freedom is not None
                    else rf"$\chi^2/\nu={float(chi_per_dof):.3g}$"
                )
            ax.text(
                0.04,
                0.96,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
        ax.set_title(component)
        ax.set_xlabel(r"$\log_2(N)$")
        ax.set_ylabel(QUANTITY_LOG_LABELS[quantity])
        ax.grid(True, alpha=0.3)
        _apply_quantile_ylim(
            ax,
            y_values_for_limits,
            ylim_quantile_low,
            ylim_quantile_high,
        )

    fig.suptitle(f"{QUANTITY_LABELS[quantity]} by width at training step {step}")
    _add_update_norm_legend(
        fig,
        legend_entries,
        loc="center right",
        fontsize=8,
        bbox_to_anchor=(0.995, 0.5),
    )
    fig.tight_layout(rect=[0, 0.02, 0.86, 0.93])
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{file_prefix}_step-{step}.png"
    fig.savefig(file_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _none_if_not_finite(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def _output_names(observable: str, quantity: str) -> Tuple[str, str, str, str, str]:
    if observable == OBSERVABLE_UPDATE_NORM:
        if quantity == "R":
            return (
                "self_av_time_step",
                "self_av_fits.png",
                "self_av_layer_fit.json",
                "update_norm",
                "self_av_time_step",
            )
        if quantity == "variance":
            return (
                "self_av_variance_time_step",
                "self_av_variance_fits.png",
                "self_av_layer_variance_fit.json",
                "update_norm_variance",
                "self_av_variance_time_step",
            )
        if quantity == "squared_first_moment":
            return (
                "self_av_squared_first_moment_time_step",
                "self_av_squared_first_moment_fits.png",
                "self_av_layer_squared_first_moment_fit.json",
                "update_norm_squared_first_moment",
                "self_av_squared_first_moment_time_step",
            )
        raise ValueError(f"Unsupported quantity {quantity!r} for update norms.")

    if quantity == "R":
        return (
            "test_loss_self_av_time_step",
            "test_loss_self_av_fits.png",
            "self_av_test_loss_fit.json",
            "test_loss",
            "test_loss_self_av_time_step",
        )
    if quantity == "variance":
        return (
            "test_loss_self_av_variance_time_step",
            "test_loss_self_av_variance_fits.png",
            "self_av_test_loss_variance_fit.json",
            "test_loss_variance",
            "test_loss_self_av_variance_time_step",
        )
    if quantity == "squared_first_moment":
        return (
            "test_loss_self_av_squared_first_moment_time_step",
            "test_loss_self_av_squared_first_moment_fits.png",
            "self_av_test_loss_squared_first_moment_fit.json",
            "test_loss_squared_first_moment",
            "test_loss_self_av_squared_first_moment_time_step",
        )
    raise ValueError(f"Unsupported quantity {quantity!r} for test loss.")


def _write_fit_file(
    fit_root: Path,
    fit_filename: str,
    fit_data: Dict[str, Any],
) -> Path:
    output_dir = fit_root / "self_av_fits"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / fit_filename
    output_path.write_text(json.dumps(fit_data, indent=2), encoding="utf-8")
    return output_path


def _quantity_exponent_symbol(quantity: str) -> str:
    return QUANTITY_EXPONENT_SYMBOLS.get(quantity, "alpha")


def _plot_alpha_time_series(
    fit_data: Dict[str, Any],
    components: Sequence[str],
    output_dir: Path,
    output_filename: str,
    *,
    dpi: int,
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> Path:
    exponent_symbol = fit_data.get(
        "exponent_symbol",
        _quantity_exponent_symbol(str(fit_data.get("quantity", ""))),
    )
    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(8.8, max(4.0, 3.6 * len(components))),
        sharex=True,
        squeeze=False,
    )
    for ax, component in zip(axes.ravel(), components):
        component_data = fit_data["components"][component]
        steps = np.asarray(component_data["training_steps"], dtype=float)
        alphas = _as_float_array(component_data["alpha_layer"])
        valid = np.isfinite(steps) & (steps > 0.0) & np.isfinite(alphas)
        positive_steps = steps[np.isfinite(steps) & (steps > 0.0)]
        ax.plot(steps[valid], alphas[valid], color="black", linewidth=1.5)
        ax.scatter(steps[valid], alphas[valid], color="black", s=12)
        ax.set_xscale("log", base=10)
        if positive_steps.size:
            x_min = float(np.min(positive_steps))
            x_max = float(np.max(positive_steps))
            if x_min == x_max:
                ax.set_xlim(0.9 * x_min, 1.1 * x_max)
            else:
                ax.set_xlim(x_min, x_max)
        ax.set_ylabel(rf"${exponent_symbol}(t)$")
        ax.set_title(component)
        ax.grid(True, which="both", alpha=0.3)
        _apply_quantile_ylim(
            ax,
            alphas[valid],
            ylim_quantile_low,
            ylim_quantile_high,
        )

    axes.ravel()[-1].set_xlabel("Training step")
    fig.suptitle(
        f"{fit_data['quantity_label']} exponent {fit_data.get('exponent_name', '')} over training"
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def _exponents_by_step(component_fit: Dict[str, List[Any]]) -> Dict[int, float]:
    steps = np.asarray(component_fit.get("training_steps", []), dtype=float)
    exponents = _as_float_array(component_fit.get("alpha_layer", []))
    values_by_step: Dict[int, float] = {}
    for step, exponent in zip(steps, exponents):
        if math.isfinite(float(step)) and math.isfinite(float(exponent)):
            values_by_step[int(step)] = float(exponent)
    return values_by_step


def _diagnostics_output_filename(observable: str) -> str:
    if observable == OBSERVABLE_TEST_LOSS:
        return "test_loss_exponent_diagnostics.png"
    return "exponent_diagnostics.png"


def _plot_exponent_diagnostics(
    fits_by_quantity: Dict[str, Dict[str, Dict[str, List[Any]]]],
    components: Sequence[str],
    output_dir: Path,
    observable: str,
    *,
    dpi: int,
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> Path:
    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(8.8, max(4.0, 3.6 * len(components))),
        sharex=True,
        squeeze=False,
    )
    for ax, component in zip(axes.ravel(), components):
        alpha_by_step = _exponents_by_step(fits_by_quantity["R"][component])
        gamma_by_step = _exponents_by_step(fits_by_quantity["variance"][component])
        omega_by_step = _exponents_by_step(
            fits_by_quantity["squared_first_moment"][component]
        )
        common_steps = sorted(
            set(alpha_by_step)
            & set(gamma_by_step)
            & set(omega_by_step)
        )
        steps = np.asarray(common_steps, dtype=float)
        diagnostics = np.asarray(
            [
                alpha_by_step[step] - (gamma_by_step[step] - omega_by_step[step])
                for step in common_steps
            ],
            dtype=float,
        )
        valid = np.isfinite(steps) & (steps > 0.0) & np.isfinite(diagnostics)
        positive_steps = steps[np.isfinite(steps) & (steps > 0.0)]
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.plot(steps[valid], diagnostics[valid], color="black", linewidth=1.5)
        ax.scatter(steps[valid], diagnostics[valid], color="black", s=12)
        ax.set_xscale("log", base=10)
        if positive_steps.size:
            x_min = float(np.min(positive_steps))
            x_max = float(np.max(positive_steps))
            if x_min == x_max:
                ax.set_xlim(0.9 * x_min, 1.1 * x_max)
            else:
                ax.set_xlim(x_min, x_max)
        ax.set_ylabel(r"$\alpha(t) - [\gamma(t)-\omega(t)]$")
        ax.set_title(component)
        ax.grid(True, which="both", alpha=0.3)
        _apply_quantile_ylim(
            ax,
            diagnostics[valid],
            ylim_quantile_low,
            ylim_quantile_high,
        )

    axes.ravel()[-1].set_xlabel("Training step")
    fig.suptitle("Exponent Diagnostics")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _diagnostics_output_filename(observable)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def _stat_by_step(
    run: RunSelfAveraging,
    component: str,
    quantity: str,
) -> Dict[int, float]:
    steps = run.steps_by_component.get(component)
    values = run.stats_by_component.get(component, {}).get(quantity)
    if steps is None or values is None:
        return {}
    result: Dict[int, float] = {}
    for step, value in zip(steps, values):
        value_float = float(value)
        if math.isfinite(value_float):
            result[int(step)] = value_float
    return result


def _raw_values_by_step(
    record: ScalarObservableRecord,
    component: str,
) -> Dict[int, float]:
    steps = record.steps_by_component.get(component)
    values = record.values_by_component.get(component)
    if steps is None or values is None:
        return {}
    result: Dict[int, float] = {}
    for step, value in zip(steps, values):
        value_float = float(value)
        if math.isfinite(value_float):
            result[int(step)] = value_float
    return result


def _selected_residual_records(
    records_by_group: Dict[str, List[ScalarObservableRecord]],
    ordered_groups: Sequence[str],
    *,
    num_draws: int,
    seed: int,
) -> List[Dict[str, ScalarObservableRecord]]:
    if seed < 0:
        raise ValueError(f"Residual random seed must be non-negative, got {seed}.")
    rng = np.random.default_rng(int(seed))
    selections: List[Dict[str, ScalarObservableRecord]] = []
    for _ in range(num_draws):
        row_selection: Dict[str, ScalarObservableRecord] = {}
        for group_key in ordered_groups:
            records = records_by_group.get(group_key, [])
            if not records:
                continue
            row_selection[group_key] = records[int(rng.integers(0, len(records)))]
        selections.append(row_selection)
    return selections


def _residual_output_filename(observable: str, component: str) -> str:
    if observable == OBSERVABLE_TEST_LOSS:
        return "test_loss_residual_diagnostics.png"
    return f"residual_diagnostics_{component}.png"


def _plot_residual_diagnostics(
    series: Sequence[RunSelfAveraging],
    records_by_group: Dict[str, List[ScalarObservableRecord]],
    components: Sequence[str],
    output_dir: Path,
    observable: str,
    colors_by_run: Dict[str, Any],
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    *,
    random_seed: int,
    dpi: int,
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> List[Path]:
    ordered_series = sorted(series, key=lambda run: tuple(_width_key(run.label)))
    ordered_groups = [run.run_dir.name for run in ordered_series]
    series_by_group = {run.run_dir.name: run for run in ordered_series}
    selections = _selected_residual_records(
        records_by_group,
        ordered_groups,
        num_draws=3,
        seed=random_seed,
    )
    output_paths: List[Path] = []

    for component in components:
        fig, axes = plt.subplots(
            3,
            2,
            figsize=(14.2, 10.2),
            sharex=True,
            squeeze=False,
        )
        for row_index, selection in enumerate(selections):
            residual_ax = axes[row_index, 0]
            standardized_ax = axes[row_index, 1]
            residual_values_for_limits: List[float] = []
            standardized_values_for_limits: List[float] = []

            for group_key in ordered_groups:
                aggregate_run = series_by_group.get(group_key)
                record = selection.get(group_key)
                if aggregate_run is None or record is None:
                    continue

                mean_by_step = _stat_by_step(aggregate_run, component, "mean")
                r_by_step = _stat_by_step(aggregate_run, component, "R")
                raw_by_step = _raw_values_by_step(record, component)
                common_steps = sorted(set(mean_by_step) & set(raw_by_step))

                x_values: List[float] = []
                residuals: List[float] = []
                standardized_residuals: List[float] = []
                standardized_x_values: List[float] = []
                for step in common_steps:
                    if step <= 0:
                        continue
                    mean = mean_by_step[step]
                    raw_value = raw_by_step[step]
                    if mean == 0.0 or not math.isfinite(mean):
                        continue
                    residual = (raw_value - mean) / mean
                    if not math.isfinite(residual):
                        continue
                    x_values.append(float(step))
                    residuals.append(float(residual))
                    r_value = r_by_step.get(step)
                    if r_value is not None and math.isfinite(r_value) and r_value > 0.0:
                        standardized = residual / math.sqrt(r_value)
                        if math.isfinite(standardized):
                            standardized_x_values.append(float(step))
                            standardized_residuals.append(float(standardized))

                if not x_values:
                    continue
                color = colors_by_run.get(_run_key(aggregate_run), "black")
                residual_ax.plot(
                    x_values,
                    residuals,
                    color=color,
                    linewidth=1.1,
                    marker="o",
                    markersize=2.0,
                    alpha=0.9,
                )
                residual_values_for_limits.extend(residuals)
                if standardized_x_values:
                    standardized_ax.plot(
                        standardized_x_values,
                        standardized_residuals,
                        color=color,
                        linewidth=1.1,
                        marker="o",
                        markersize=2.0,
                        alpha=0.9,
                    )
                    standardized_values_for_limits.extend(standardized_residuals)

            for ax in (residual_ax, standardized_ax):
                ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9, alpha=0.8)
                ax.set_xscale("log", base=10)
                ax.grid(True, which="both", alpha=0.3)
            residual_ax.set_ylabel(f"draw {row_index + 1}\n" + r"$Z_N(t)$")
            standardized_ax.set_ylabel(r"$\tilde{Z}_N(t)$")
            if row_index == 0:
                residual_ax.set_title(
                    r"$Z_N(t) = [O_N(t)-\mathbb{E}O_N(t)]/\mathbb{E}O_N(t)$"
                )
                standardized_ax.set_title(
                    r"$\tilde{Z}_N(t)=Z_N(t)/\sqrt{R_N(t)}$"
                )
            _apply_quantile_ylim(
                residual_ax,
                residual_values_for_limits,
                ylim_quantile_low,
                ylim_quantile_high,
            )
            _apply_quantile_ylim(
                standardized_ax,
                standardized_values_for_limits,
                ylim_quantile_low,
                ylim_quantile_high,
            )

        axes[-1, 0].set_xlabel("Training step")
        axes[-1, 1].set_xlabel("Training step")
        fig.suptitle(f"Relative Residual Diagnostics: {component}")
        _add_update_norm_legend(
            axes[0, 1],
            legend_entries,
            loc="upper left",
            fontsize=8,
            bbox_to_anchor=(1.02, 1.0),
        )
        fig.tight_layout(rect=[0, 0.02, 0.88, 0.95])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _residual_output_filename(observable, component)
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def _plot_rescaled_update_norm_means(
    series: Sequence[RunSelfAveraging],
    output_dir: Path,
    colors_by_run: Dict[str, Any],
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    *,
    dpi: int,
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> Path:
    largest_run = _largest_width_run(series)
    fig, axes = plt.subplots(
        2,
        len(LAYERS),
        figsize=(13.5, 8.2),
        sharex="col",
        squeeze=False,
    )

    for column, layer in enumerate(LAYERS):
        top_ax = axes[0, column]
        ratio_ax = axes[1, column]
        top_y_values: List[float] = []
        ratio_y_values: List[float] = [1.0]
        reference_steps = largest_run.steps_by_component.get(layer)
        reference_stats = largest_run.stats_by_component.get(layer, {})
        reference_means = reference_stats.get("mean")
        reference_by_step: Dict[int, float] = {}
        if reference_steps is not None and reference_means is not None:
            for step, value in zip(reference_steps, reference_means):
                value_float = float(value)
                if math.isfinite(value_float) and value_float > 0.0:
                    reference_by_step[int(step)] = value_float

        for run in sorted(series, key=lambda item: tuple(_width_key(item.label))):
            steps = run.steps_by_component.get(layer)
            stats = run.stats_by_component.get(layer, {})
            means = stats.get("mean")
            variances = stats.get("variance")
            if steps is None or means is None or variances is None:
                continue
            if not (len(steps) == len(means) == len(variances)):
                continue

            steps_float = np.asarray(steps, dtype=float)
            means_float = np.asarray(means, dtype=float)
            variances_float = np.asarray(variances, dtype=float)
            valid = (
                np.isfinite(steps_float)
                & (steps_float > 0.0)
                & np.isfinite(means_float)
                & (means_float > 0.0)
            )
            if not np.any(valid):
                continue

            color = colors_by_run.get(_run_key(run), "black")
            top_ax.plot(
                steps_float[valid],
                means_float[valid],
                color=color,
                linewidth=1.25,
                marker="o",
                markersize=2.6,
                label=run.label,
            )
            top_y_values.extend(
                float(value) for value in means_float[valid] if math.isfinite(float(value))
            )

            spread = np.sqrt(np.clip(variances_float, 0.0, None))
            lower = means_float - spread
            upper = means_float + spread
            band_valid = (
                valid
                & np.isfinite(lower)
                & np.isfinite(upper)
                & (upper > 0.0)
            )
            if np.any(band_valid):
                positive_lower = np.clip(lower, np.nextafter(0.0, 1.0), None)
                top_ax.fill_between(
                    steps_float,
                    positive_lower,
                    upper,
                    where=band_valid,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )

            ratio_steps: List[float] = []
            ratio_values: List[float] = []
            ratio_spreads: List[float] = []
            for step, mean, variance in zip(steps_float, means_float, variances_float):
                reference_mean = reference_by_step.get(int(step))
                if reference_mean is None or reference_mean <= 0.0:
                    continue
                if not (math.isfinite(float(step)) and float(step) > 0.0):
                    continue
                if not (math.isfinite(float(mean)) and float(mean) > 0.0):
                    continue
                ratio_steps.append(float(step))
                ratio_values.append(float(mean) / reference_mean)
                variance_float = float(variance)
                ratio_spreads.append(
                    math.sqrt(max(variance_float, 0.0)) / reference_mean
                    if math.isfinite(variance_float)
                    else float("nan")
                )

            if ratio_values:
                ratio_steps_array = np.asarray(ratio_steps, dtype=float)
                ratio_values_array = np.asarray(ratio_values, dtype=float)
                ratio_spreads_array = np.asarray(ratio_spreads, dtype=float)
                ratio_ax.plot(
                    ratio_steps_array,
                    ratio_values_array,
                    color=color,
                    linewidth=1.25,
                    marker="o",
                    markersize=2.6,
                )
                ratio_y_values.extend(
                    float(value)
                    for value in ratio_values_array
                    if math.isfinite(float(value))
                )
                ratio_lower = ratio_values_array - ratio_spreads_array
                ratio_upper = ratio_values_array + ratio_spreads_array
                ratio_band_valid = (
                    np.isfinite(ratio_steps_array)
                    & np.isfinite(ratio_lower)
                    & np.isfinite(ratio_upper)
                    & (ratio_upper > 0.0)
                )
                if np.any(ratio_band_valid):
                    positive_ratio_lower = np.clip(
                        ratio_lower,
                        np.nextafter(0.0, 1.0),
                        None,
                    )
                    ratio_ax.fill_between(
                        ratio_steps_array,
                        positive_ratio_lower,
                        ratio_upper,
                        where=ratio_band_valid,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                    )

        top_ax.set_title(layer)
        top_ax.set_xscale("log", base=10)
        top_ax.set_yscale("log")
        top_ax.set_ylabel("Mean rescaled update norm")
        top_ax.grid(True, which="both", alpha=0.3)
        _apply_quantile_ylim(
            top_ax,
            top_y_values,
            ylim_quantile_low,
            ylim_quantile_high,
            positive_only=True,
        )

        ratio_ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
        ratio_ax.set_title(f"{layer} / {largest_run.label}")
        ratio_ax.set_xscale("log", base=10)
        ratio_ax.set_yscale("log")
        ratio_ax.set_xlabel("Training step")
        ratio_ax.set_ylabel("Ratio to largest-width mean")
        ratio_ax.grid(True, which="both", alpha=0.3)
        _apply_quantile_ylim(
            ratio_ax,
            ratio_y_values,
            ylim_quantile_low,
            ylim_quantile_high,
            positive_only=True,
        )

    fig.suptitle(
        "Rescaled update-norm means and ratios to the largest-width run; "
        "shaded bands show ±sqrt(variance)"
    )
    _add_update_norm_legend(
        axes[1, -1],
        legend_entries,
        loc="lower left",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rescaled_update_norm_means.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def _plot_rescaled_loss_means(
    series: Sequence[RunSelfAveraging],
    output_dir: Path,
    colors_by_run: Dict[str, Any],
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    *,
    dpi: int,
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> Path:
    component = TEST_LOSS_COMPONENT
    largest_run = _largest_width_run(series)
    reference_steps = largest_run.steps_by_component.get(component)
    reference_stats = largest_run.stats_by_component.get(component, {})
    reference_means = reference_stats.get("mean")
    reference_by_step: Dict[int, float] = {}
    if reference_steps is not None and reference_means is not None:
        for step, value in zip(reference_steps, reference_means):
            value_float = float(value)
            if math.isfinite(value_float) and value_float > 0.0:
                reference_by_step[int(step)] = value_float

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.2),
        sharex=True,
        squeeze=False,
    )
    mean_ax = axes[0, 0]
    ratio_ax = axes[1, 0]
    mean_y_values: List[float] = []
    ratio_y_values: List[float] = [1.0]

    for run in sorted(series, key=lambda item: tuple(_width_key(item.label))):
        steps = run.steps_by_component.get(component)
        stats = run.stats_by_component.get(component, {})
        means = stats.get("mean")
        variances = stats.get("variance")
        if steps is None or means is None or variances is None:
            continue
        if not (len(steps) == len(means) == len(variances)):
            continue

        steps_float = np.asarray(steps, dtype=float)
        means_float = np.asarray(means, dtype=float)
        variances_float = np.asarray(variances, dtype=float)
        valid = (
            np.isfinite(steps_float)
            & (steps_float > 0.0)
            & np.isfinite(means_float)
            & (means_float > 0.0)
        )
        if not np.any(valid):
            continue

        color = colors_by_run.get(_run_key(run), "black")
        mean_ax.plot(
            steps_float[valid],
            means_float[valid],
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=2.6,
            label=run.label,
        )
        mean_y_values.extend(
            float(value) for value in means_float[valid] if math.isfinite(float(value))
        )

        spread = np.sqrt(np.clip(variances_float, 0.0, None))
        lower = means_float - spread
        upper = means_float + spread
        band_valid = (
            valid
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (upper > 0.0)
        )
        if np.any(band_valid):
            positive_lower = np.clip(lower, np.nextafter(0.0, 1.0), None)
            mean_ax.fill_between(
                steps_float,
                positive_lower,
                upper,
                where=band_valid,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        ratio_steps: List[float] = []
        ratio_values: List[float] = []
        ratio_spreads: List[float] = []
        for step, mean, variance in zip(steps_float, means_float, variances_float):
            reference_mean = reference_by_step.get(int(step))
            if reference_mean is None or reference_mean <= 0.0:
                continue
            if not (math.isfinite(float(step)) and float(step) > 0.0):
                continue
            if not (math.isfinite(float(mean)) and float(mean) > 0.0):
                continue
            ratio_steps.append(float(step))
            ratio_values.append(float(mean) / reference_mean)
            variance_float = float(variance)
            ratio_spreads.append(
                math.sqrt(max(variance_float, 0.0)) / reference_mean
                if math.isfinite(variance_float)
                else float("nan")
            )

        if ratio_values:
            ratio_steps_array = np.asarray(ratio_steps, dtype=float)
            ratio_values_array = np.asarray(ratio_values, dtype=float)
            ratio_spreads_array = np.asarray(ratio_spreads, dtype=float)
            ratio_ax.plot(
                ratio_steps_array,
                ratio_values_array,
                color=color,
                linewidth=1.25,
                marker="o",
                markersize=2.6,
            )
            ratio_y_values.extend(
                float(value)
                for value in ratio_values_array
                if math.isfinite(float(value))
            )
            ratio_lower = ratio_values_array - ratio_spreads_array
            ratio_upper = ratio_values_array + ratio_spreads_array
            ratio_band_valid = (
                np.isfinite(ratio_steps_array)
                & np.isfinite(ratio_lower)
                & np.isfinite(ratio_upper)
                & (ratio_upper > 0.0)
            )
            if np.any(ratio_band_valid):
                positive_ratio_lower = np.clip(
                    ratio_lower,
                    np.nextafter(0.0, 1.0),
                    None,
                )
                ratio_ax.fill_between(
                    ratio_steps_array,
                    positive_ratio_lower,
                    ratio_upper,
                    where=ratio_band_valid,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )

    mean_ax.set_title("Test loss")
    mean_ax.set_xscale("log", base=10)
    mean_ax.set_yscale("log")
    mean_ax.set_ylabel("Mean test loss")
    mean_ax.grid(True, which="both", alpha=0.3)
    _apply_quantile_ylim(
        mean_ax,
        mean_y_values,
        ylim_quantile_low,
        ylim_quantile_high,
        positive_only=True,
    )

    ratio_ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
    ratio_ax.set_title(f"Test loss / {largest_run.label}")
    ratio_ax.set_xscale("log", base=10)
    ratio_ax.set_yscale("log")
    ratio_ax.set_xlabel("Training step")
    ratio_ax.set_ylabel("Ratio to largest-width loss")
    ratio_ax.grid(True, which="both", alpha=0.3)
    _apply_quantile_ylim(
        ratio_ax,
        ratio_y_values,
        ylim_quantile_low,
        ylim_quantile_high,
        positive_only=True,
    )

    fig.suptitle(
        "Test-loss means and ratios to the largest-width run; "
        "shaded bands show ±sqrt(variance)"
    )
    _add_update_norm_legend(
        ratio_ax,
        legend_entries,
        loc="lower left",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rescaled_loss_means.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def _resolve_task_name(series: Sequence[RunSelfAveraging], override: str | None) -> str:
    if override is not None:
        return override
    task_names = {run.task_name for run in series}
    if len(task_names) == 1:
        return next(iter(task_names))
    return "mixed_tasks"


def _build_fit_for_quantity(
    series: Sequence[RunSelfAveraging],
    components: Sequence[str],
    quantity: str,
    steps: Sequence[int],
    *,
    step_plot_dir: Path,
    file_prefix: str,
    skip_step_plots: bool,
    dpi: int,
    min_points: int,
    colors_by_run: Dict[str, Any],
    legend_entries: Dict[str, List[Tuple[Any, str]]],
    ylim_quantile_low: float,
    ylim_quantile_high: float,
) -> Dict[str, Any]:
    component_fits: Dict[str, Dict[str, List[Any]]] = {
        component: {
            "training_steps": [],
            "alpha_layer": [],
            "intercept_layer": [],
            "r_squared": [],
            "chi_squared": [],
            "degrees_of_freedom": [],
            "chi_squared_per_dof": [],
            "num_fit_points": [],
            "num_points": [],
            "widths": [],
            quantity: [],
            "log_variance_std": [],
            "run_labels": [],
        }
        for component in components
    }

    for step in steps:
        fit_for_step: Dict[str, Dict[str, Any]] = {}
        for component in components:
            widths, values, labels, _, sample_counts = _points_for_step(
                series,
                component,
                quantity,
                int(step),
            )
            y_errors = _log_variance_stds(sample_counts) if quantity == "variance" else None
            fit = _fit_log2_power_law(widths, values, y_errors=y_errors)
            fit_for_step[component] = fit
            component_fit = component_fits[component]
            component_fit["training_steps"].append(int(step))
            component_fit["alpha_layer"].append(_none_if_not_finite(fit["alpha_layer"]))
            component_fit["intercept_layer"].append(_none_if_not_finite(fit["intercept_layer"]))
            component_fit["r_squared"].append(_none_if_not_finite(fit["r_squared"]))
            component_fit["chi_squared"].append(_none_if_not_finite(fit["chi_squared"]))
            component_fit["degrees_of_freedom"].append(fit["degrees_of_freedom"])
            component_fit["chi_squared_per_dof"].append(
                _none_if_not_finite(fit["chi_squared_per_dof"])
            )
            component_fit["num_fit_points"].append(int(fit["num_fit_points"]))
            component_fit["num_points"].append(int(widths.size))
            component_fit["widths"].append([int(value) for value in widths])
            component_fit[quantity].append([float(value) for value in values])
            component_fit["log_variance_std"].append(
                []
                if y_errors is None
                else [
                    _none_if_not_finite(float(value))
                    for value in np.asarray(y_errors, dtype=float)
                ]
            )
            component_fit["run_labels"].append(labels)

        if not skip_step_plots:
            _plot_step(
                series,
                components,
                quantity,
                int(step),
                step_plot_dir,
                fit_for_step,
                dpi=dpi,
                min_points=min_points,
                file_prefix=file_prefix,
                colors_by_run=colors_by_run,
                legend_entries=legend_entries,
                ylim_quantile_low=ylim_quantile_low,
                ylim_quantile_high=ylim_quantile_high,
            )

    return component_fits


def run_self_averaging_plots_for_observable(
    job_dirs: Sequence[Path],
    *,
    observable: str,
    metrics_name: str | None = None,
    figures_root: Path | None = None,
    task_name: str | None = None,
    compute_missing: bool = False,
    force_metrics: bool = False,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    skip_step_plots: bool = False,
    dpi: int = 170,
    min_points: int = 2,
    ylim_quantile_low: float = DEFAULT_YLIM_QUANTILE_LOW,
    ylim_quantile_high: float = DEFAULT_YLIM_QUANTILE_HIGH,
    residual_random_seed: int = DEFAULT_RESIDUAL_RANDOM_SEED,
) -> List[Tuple[Path, Path]]:
    if not job_dirs:
        raise ValueError("At least one job directory is required.")
    if observable not in OBSERVABLES:
        raise ValueError(f"Unsupported observable {observable!r}.")
    _validate_ylim_quantiles(ylim_quantile_low, ylim_quantile_high)
    resolved_job_dirs = [job_dir.resolve() for job_dir in job_dirs]
    combined_name = _combined_job_name(resolved_job_dirs)
    metrics_name = metrics_name or OUTPUT_FILENAMES[observable]
    components = _components_for_observable(observable)

    if len(resolved_job_dirs) == 1:
        estimator = "local_time_window"
        series = _collect_single_job_series(
            resolved_job_dirs[0],
            observable=observable,
            metrics_name=metrics_name,
            compute_missing=compute_missing,
            force_metrics=force_metrics,
            window_width=window_width,
            window_stride=window_stride,
        )
    else:
        estimator = "independent_jobs_same_time"
        series = _collect_cross_job_series(resolved_job_dirs, observable)

    _apply_classical_r_definition(series, components)
    if len(resolved_job_dirs) == 1:
        _write_single_job_classical_r_definition(
            series,
            components,
            metrics_name,
        )

    if figures_root is None:
        figures_root = REPO_ROOT / "figures_self_av"
    resolved_task_name = _resolve_task_name(series, task_name)
    output_dir = figures_root / resolved_task_name / combined_name
    steps = _all_training_steps(series, components)
    if not steps:
        raise ValueError(f"No self-averaging training steps found for {combined_name}")

    colors_by_run, legend_entries = _run_color_map(series)
    records_by_group = _collect_observable_records_by_group(resolved_job_dirs, observable)
    if observable == OBSERVABLE_UPDATE_NORM:
        mean_plot_path = _plot_rescaled_update_norm_means(
            series,
            output_dir,
            colors_by_run,
            legend_entries,
            dpi=dpi,
            ylim_quantile_low=ylim_quantile_low,
            ylim_quantile_high=ylim_quantile_high,
        )
        print(f"Saved rescaled update-norm mean plot to {mean_plot_path}")
    if observable == OBSERVABLE_TEST_LOSS:
        loss_plot_path = _plot_rescaled_loss_means(
            series,
            output_dir,
            colors_by_run,
            legend_entries,
            dpi=dpi,
            ylim_quantile_low=ylim_quantile_low,
            ylim_quantile_high=ylim_quantile_high,
        )
        print(f"Saved rescaled loss mean plot to {loss_plot_path}")
    residual_paths = _plot_residual_diagnostics(
        series,
        records_by_group,
        components,
        output_dir,
        observable,
        colors_by_run,
        legend_entries,
        random_seed=residual_random_seed,
        dpi=dpi,
        ylim_quantile_low=ylim_quantile_low,
        ylim_quantile_high=ylim_quantile_high,
    )
    for residual_path in residual_paths:
        print(f"Saved residual diagnostics plot to {residual_path}")

    fit_root = _combined_fit_root(resolved_job_dirs, combined_name)
    outputs: List[Tuple[Path, Path]] = []
    fits_by_quantity: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    for quantity in QUANTITIES:
        step_dir_name, alpha_filename, fit_filename, analysis_key, file_prefix = _output_names(
            observable,
            quantity,
        )
        component_fits = _build_fit_for_quantity(
            series,
            components,
            quantity,
            steps,
            step_plot_dir=output_dir / step_dir_name,
            file_prefix=file_prefix,
            skip_step_plots=skip_step_plots,
            dpi=dpi,
            min_points=min_points,
            colors_by_run=colors_by_run,
            legend_entries=legend_entries,
            ylim_quantile_low=ylim_quantile_low,
            ylim_quantile_high=ylim_quantile_high,
        )
        fits_by_quantity[quantity] = component_fits
        fit_data = {
            "created_by": "visualizations/plot_self_averaging.py",
            "job_dirs": [str(job_dir) for job_dir in resolved_job_dirs],
            "analysis_name": combined_name,
            "analysis_key": analysis_key,
            "fit_model": QUANTITY_FIT_MODELS[quantity],
            "observable": observable,
            "quantity": quantity,
            "quantity_label": QUANTITY_LABELS[quantity],
            "exponent_name": QUANTITY_EXPONENT_NAMES[quantity],
            "exponent_symbol": QUANTITY_EXPONENT_SYMBOLS[quantity],
            "estimator": estimator,
            "metrics_file": metrics_name,
            "window_width": int(window_width) if estimator == "local_time_window" else None,
            "window_stride": int(window_stride) if estimator == "local_time_window" else None,
            "components": component_fits,
        }
        if quantity == "variance":
            fit_data["log_variance_uncertainty"] = {
                "log_base": 2.0,
                "formula": "sigma_log2_variance = sqrt(2 / (M - 1)) / ln(2)",
                "M_source": (
                    "sample_count from local time windows for single-job analyses; "
                    "number of independent same-time job values for multi-job analyses"
                ),
                "assumption": (
                    "Approximate chi-square uncertainty for independent Gaussian samples."
                ),
            }
        if observable == OBSERVABLE_UPDATE_NORM:
            fit_data["layers"] = component_fits
        fit_path = _write_fit_file(fit_root, fit_filename, fit_data)
        alpha_plot_path = _plot_alpha_time_series(
            fit_data,
            components,
            output_dir,
            alpha_filename,
            dpi=dpi,
            ylim_quantile_low=ylim_quantile_low,
            ylim_quantile_high=ylim_quantile_high,
        )
        outputs.append((fit_path, alpha_plot_path))
    diagnostics_path = _plot_exponent_diagnostics(
        fits_by_quantity,
        components,
        output_dir,
        observable,
        dpi=dpi,
        ylim_quantile_low=ylim_quantile_low,
        ylim_quantile_high=ylim_quantile_high,
    )
    print(f"Saved exponent diagnostics plot to {diagnostics_path}")
    return outputs


def _parse_job_dirs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[Path]:
    job_dirs: List[Path] = []
    job_dirs.extend(args.job_dir_flags or [])
    job_dirs.extend(args.job_dirs or [])
    if not job_dirs:
        parser.error("Provide at least one job directory.")
    return job_dirs


def _observables_from_arg(value: str) -> Tuple[str, ...]:
    if value == "all":
        return OBSERVABLES
    return (value,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot log2 self-averaging statistics vs log2(N) and fit exponents."
    )
    parser.add_argument(
        "job_dirs",
        nargs="*",
        type=Path,
        help="Job directories containing run folders with training_log.json files.",
    )
    parser.add_argument(
        "--job-dir",
        dest="job_dir_flags",
        action="append",
        type=Path,
        default=None,
        help="Job directory. May be supplied multiple times.",
    )
    parser.add_argument(
        "--observable",
        choices=OBSERVABLES + ("all",),
        default=OBSERVABLE_UPDATE_NORM,
        help="Observable to analyze.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default=None,
        help="Metrics filename under each run's weight_metrics/. Defaults depend on --observable.",
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=None,
        help="Output root for figures. Defaults to figures_self_av in the repo root.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Override the task-name directory under figures_self_av/.",
    )
    parser.add_argument(
        "--compute-missing",
        action="store_true",
        help="For a single job, compute missing run-level self-av files before plotting.",
    )
    parser.add_argument(
        "--force-metrics",
        action="store_true",
        help="For a single job, recompute run-level self-av files before plotting.",
    )
    parser.add_argument(
        "--window-width",
        "--window_width",
        dest="window_width",
        type=int,
        default=DEFAULT_WINDOW_WIDTH,
        help="Single-job local moment window width in saved observable samples.",
    )
    parser.add_argument(
        "--window-stride",
        "--window_stride",
        dest="window_stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE,
        help="Single-job stride between evaluated local moment windows.",
    )
    parser.add_argument(
        "--skip-step-plots",
        action="store_true",
        help="Only write fit parameters and alpha_layer time-series plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--ylim-quantile-low",
        type=float,
        default=DEFAULT_YLIM_QUANTILE_LOW,
        help="Lower quantile used for automatic y-limit clipping.",
    )
    parser.add_argument(
        "--ylim-quantile-high",
        type=float,
        default=DEFAULT_YLIM_QUANTILE_HIGH,
        help="Upper quantile used for automatic y-limit clipping.",
    )
    parser.add_argument(
        "--residual-random-seed",
        type=int,
        default=DEFAULT_RESIDUAL_RANDOM_SEED,
        help="Random seed for selecting the three per-width runs in residual diagnostics.",
    )
    args = parser.parse_args()

    if args.window_width < 2:
        parser.error("--window-width must be at least 2.")
    if args.window_stride < 1:
        parser.error("--window-stride must be at least 1.")
    if args.residual_random_seed < 0:
        parser.error("--residual-random-seed must be non-negative.")
    try:
        _validate_ylim_quantiles(args.ylim_quantile_low, args.ylim_quantile_high)
    except ValueError as exc:
        parser.error(str(exc))
    if args.metrics_name is not None and args.observable == "all":
        parser.error("--metrics-name can only be used with a single --observable.")

    all_outputs: List[Tuple[Path, Path]] = []
    for observable in _observables_from_arg(args.observable):
        all_outputs.extend(
            run_self_averaging_plots_for_observable(
                _parse_job_dirs(args, parser),
                observable=observable,
                metrics_name=args.metrics_name,
                figures_root=args.figures_root,
                task_name=args.task_name,
                compute_missing=args.compute_missing,
                force_metrics=args.force_metrics,
                window_width=args.window_width,
                window_stride=args.window_stride,
                skip_step_plots=args.skip_step_plots,
                dpi=args.dpi,
                ylim_quantile_low=args.ylim_quantile_low,
                ylim_quantile_high=args.ylim_quantile_high,
                residual_random_seed=args.residual_random_seed,
            )
        )

    for fit_path, alpha_plot_path in all_outputs:
        print(f"Saved fit parameters to {fit_path}")
        print(f"Saved alpha_layer plot to {alpha_plot_path}")


if __name__ == "__main__":
    main()
