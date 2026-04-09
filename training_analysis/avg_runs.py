"""Average multiple training jobs into a single avg job directory."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _find_weight_metrics_path(run_dir: Path, layer: str) -> Path:
    for directory_name in ("weight_metrics", "avg_final_weight_metrics"):
        candidate = run_dir / directory_name / f"{layer}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing weight metrics file for {layer} in {run_dir}")


def collect_loss_metrics(log_path: Path) -> Tuple[np.ndarray, np.ndarray, float, float]:
    final_metrics = _load_final_metrics_block(log_path)
    history = final_metrics.get("history", {})
    if not isinstance(history, dict):
        raise ValueError(f"No metrics history found in {log_path}")

    train_losses = history.get("train_loss")
    test_losses = history.get("test_loss")
    if not train_losses:
        raise ValueError(f"No train loss history found in {log_path}")
    if not test_losses:
        raise ValueError(f"No test loss history found in {log_path}")

    train_losses_array = np.asarray(train_losses, dtype=float)
    test_losses_array = np.asarray(test_losses, dtype=float)
    final_train_loss = final_metrics.get("final_train_loss")
    final_test_loss = final_metrics.get("final_test_loss")

    if final_train_loss is None:
        final_train_loss = float(train_losses_array[-1])
    if final_test_loss is None:
        final_test_loss = float(test_losses_array[-1])

    return (
        train_losses_array,
        test_losses_array,
        float(final_train_loss),
        float(final_test_loss),
    )


def collect_weight_metrics(weight_metrics_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    training_metrics = _load_training_metrics_block(weight_metrics_path)
    step_norms = training_metrics.get("step_norms")
    similarities = training_metrics.get("similarities")
    if not step_norms:
        raise ValueError(f"No step norms history found in {weight_metrics_path}")
    if not similarities:
        raise ValueError(f"No similarities history found in {weight_metrics_path}")

    return np.asarray(step_norms, dtype=float), np.asarray(similarities, dtype=float)


@dataclass
class RunSeries:
    job_dir: Path
    run_dir: Path
    job_key: str
    scheme: str
    label: str
    x_axis: np.ndarray
    train_losses: np.ndarray
    test_losses: np.ndarray
    final_train_loss: float
    final_test_loss: float
    step_norms: np.ndarray
    similarities: np.ndarray
    task_name: str
    simulation_info: Dict[str, Any]
    dataset_info: Dict[str, Any]
    network_info: Dict[str, Any]


@dataclass
class AggregatedSeries:
    scheme: str
    label: str
    x_axis: np.ndarray
    train_losses_mean: np.ndarray
    train_losses_std: np.ndarray
    test_losses_mean: np.ndarray
    test_losses_std: np.ndarray
    final_train_loss_mean: float
    final_train_loss_std: float
    final_test_loss_mean: float
    final_test_loss_std: float
    step_norms_mean: np.ndarray
    step_norms_std: np.ndarray
    similarities_mean: np.ndarray
    similarities_std: np.ndarray
    records: List[RunSeries]


def _collect_run_series(
    job_dir: Path,
    log_path: Path,
    layer: str,
    compute_flag: bool,
) -> RunSeries:
    simulation_config_path = log_path.parent / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml next to {log_path}")

    simulation_info = load_yaml_as_dict(simulation_config_path)
    dataset_info = simulation_info.get("training")
    network_info = simulation_info.get("network")
    if not isinstance(dataset_info, dict) or not isinstance(network_info, dict):
        raise ValueError(f"Incomplete simulation config in {simulation_config_path}")

    train_losses, test_losses, final_train_loss, final_test_loss = collect_loss_metrics(log_path)
    step_norms, similarities = collect_weight_metrics(
        _find_weight_metrics_path(log_path.parent, layer)
    )
    x_axis, _, _ = _build_x_axis(test_losses, dataset_info, network_info, compute_flag)

    task_path = dataset_info.get("training_data", {}).get("task")
    task_name = Path(task_path).name if task_path else job_dir.name

    return RunSeries(
        job_dir=job_dir,
        run_dir=log_path.parent,
        job_key=str(job_dir.resolve()),
        scheme=_scheme_from_path(log_path),
        label=_label_from_log_path(log_path),
        x_axis=x_axis,
        train_losses=train_losses,
        test_losses=test_losses,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        step_norms=step_norms,
        similarities=similarities,
        task_name=task_name,
        simulation_info=simulation_info,
        dataset_info=dataset_info,
        network_info=network_info,
    )


def _aggregate_scalars(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    valid = ~np.isnan(array)
    if not np.any(valid):
        return float("nan"), float("nan")

    valid_values = array[valid]
    mean = float(np.mean(valid_values))
    if valid_values.shape[0] == 1:
        std = 0.0
    else:
        std = float(np.std(valid_values))
    return mean, std


def _aggregate_arrays(arrays: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    min_len = min(len(array) for array in arrays)
    stacked = np.stack([np.asarray(array[:min_len], dtype=float) for array in arrays], axis=0)
    valid = ~np.isnan(stacked)
    counts = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, stacked, 0.0), axis=0)
    mean = np.divide(
        sums,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )

    if stacked.shape[0] == 1:
        std = np.where(counts > 0, 0.0, np.nan)
    else:
        squared_error = np.sum(np.where(valid, (stacked - mean) ** 2, 0.0), axis=0)
        variance = np.divide(
            squared_error,
            counts,
            out=np.full(counts.shape, np.nan, dtype=float),
            where=counts > 0,
        )
        std = np.sqrt(variance)
    return mean, std


def _common_x_axis(x_axes: Sequence[np.ndarray], label: str) -> np.ndarray:
    min_len = min(len(axis) for axis in x_axes)
    base = np.asarray(x_axes[0][:min_len], dtype=float)
    for axis in x_axes[1:]:
        candidate = np.asarray(axis[:min_len], dtype=float)
        if not np.allclose(base, candidate):
            raise ValueError(
                f"Runs for {label} do not share the same x-axis and cannot be averaged."
            )
    return base


def _aggregate_group(
    key: Tuple[str, str],
    records: Sequence[RunSeries],
    num_requested_jobs: int,
) -> AggregatedSeries:
    scheme, label = key
    if len(records) < num_requested_jobs:
        print(
            f"Warning: averaging {scheme}/{label} over {len(records)} of "
            f"{num_requested_jobs} provided job directories."
        )

    x_axis = _common_x_axis([record.x_axis for record in records], f"{scheme}/{label}")
    train_losses_mean, train_losses_std = _aggregate_arrays(
        [record.train_losses for record in records]
    )
    test_losses_mean, test_losses_std = _aggregate_arrays(
        [record.test_losses for record in records]
    )
    final_train_loss_mean, final_train_loss_std = _aggregate_scalars(
        [record.final_train_loss for record in records]
    )
    final_test_loss_mean, final_test_loss_std = _aggregate_scalars(
        [record.final_test_loss for record in records]
    )
    step_norms_mean, step_norms_std = _aggregate_arrays(
        [record.step_norms for record in records]
    )
    similarities_mean, similarities_std = _aggregate_arrays(
        [record.similarities for record in records]
    )

    return AggregatedSeries(
        scheme=scheme,
        label=label,
        x_axis=x_axis,
        train_losses_mean=train_losses_mean,
        train_losses_std=train_losses_std,
        test_losses_mean=test_losses_mean,
        test_losses_std=test_losses_std,
        final_train_loss_mean=final_train_loss_mean,
        final_train_loss_std=final_train_loss_std,
        final_test_loss_mean=final_test_loss_mean,
        final_test_loss_std=final_test_loss_std,
        step_norms_mean=step_norms_mean,
        step_norms_std=step_norms_std,
        similarities_mean=similarities_mean,
        similarities_std=similarities_std,
        records=list(records),
    )


def _job_sort_key(job_dir: Path) -> Tuple[int, str]:
    match = re.search(r"job-(\d+)", job_dir.name)
    if match:
        return int(match.group(1)), job_dir.name
    return 10**18, job_dir.name


def _average_job_directory_name(job_dirs: Sequence[Path]) -> str:
    ordered_job_dirs = sorted(job_dirs, key=_job_sort_key)
    first_name = ordered_job_dirs[0].name
    last_name = ordered_job_dirs[-1].name
    if first_name == last_name:
        return f"avg_{first_name}"
    return f"avg_{first_name}_{last_name}"


def _average_job_directory(job_dirs: Sequence[Path]) -> Path:
    common_parent = Path(
        os.path.commonpath([str(job_dir.resolve().parent) for job_dir in job_dirs])
    )
    return common_parent / _average_job_directory_name(job_dirs)


def _json_ready_array(values: np.ndarray) -> List[float]:
    return np.asarray(values, dtype=float).tolist()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def _collect_weight_metric_files(run_dir: Path) -> Dict[str, Path]:
    for directory_name in ("weight_metrics", "avg_final_weight_metrics"):
        metrics_dir = run_dir / directory_name
        if metrics_dir.exists():
            return {
                metric_path.name: metric_path
                for metric_path in sorted(metrics_dir.glob("*.json"))
            }
    return {}


def _build_avg_training_log(series: AggregatedSeries) -> Dict[str, Any]:
    return {
        "avg_final_metrics": {
            "final_train_loss": float(series.final_train_loss_mean),
            "final_train_loss_std": float(series.final_train_loss_std),
            "final_test_loss": float(series.final_test_loss_mean),
            "final_test_loss_std": float(series.final_test_loss_std),
            "history": {
                "train_loss": _json_ready_array(series.train_losses_mean),
                "train_loss_std": _json_ready_array(series.train_losses_std),
                "test_loss": _json_ready_array(series.test_losses_mean),
                "test_loss_std": _json_ready_array(series.test_losses_std),
            },
        }
    }


def _build_avg_simulation_config(
    series: AggregatedSeries, num_requested_jobs: int
) -> Dict[str, Any]:
    simulation_info = copy.deepcopy(series.records[0].simulation_info)
    simulation_info["averaging"] = {
        "num_files_averaged": len(series.records),
        "num_requested_jobs": num_requested_jobs,
        "source_job_dirs": [
            str(record.job_dir.resolve())
            for record in sorted(series.records, key=lambda item: item.job_key)
        ],
        "source_run_dirs": [
            str(record.run_dir.resolve())
            for record in sorted(series.records, key=lambda item: item.job_key)
        ],
    }
    return simulation_info


def _aggregate_weight_metric_payloads(metric_paths: Sequence[Path]) -> Dict[str, Any]:
    metric_blocks = [_load_training_metrics_block(metric_path) for metric_path in metric_paths]

    step_norms_mean, step_norms_std = _aggregate_arrays(
        [np.asarray(metrics["step_norms"], dtype=float) for metrics in metric_blocks]
    )
    similarities_mean, similarities_std = _aggregate_arrays(
        [np.asarray(metrics["similarities"], dtype=float) for metrics in metric_blocks]
    )
    cum_path_length_mean, cum_path_length_std = _aggregate_scalars(
        [float(metrics["cum_path_length"]) for metrics in metric_blocks]
    )
    normalized_distance_mean, normalized_distance_std = _aggregate_scalars(
        [float(metrics["normalized_distance"]) for metrics in metric_blocks]
    )
    relative_distance_mean, relative_distance_std = _aggregate_scalars(
        [float(metrics["relative_distance"]) for metrics in metric_blocks]
    )
    overlap_mean, overlap_std = _aggregate_scalars(
        [float(metrics["overlap"]) for metrics in metric_blocks]
    )

    return {
        "avg_training_metrics": {
            "step_norms": _json_ready_array(step_norms_mean),
            "step_norms_std": _json_ready_array(step_norms_std),
            "similarities": _json_ready_array(similarities_mean),
            "similarities_std": _json_ready_array(similarities_std),
            "cum_path_length": float(cum_path_length_mean),
            "cum_path_length_std": float(cum_path_length_std),
            "normalized_distance": float(normalized_distance_mean),
            "normalized_distance_std": float(normalized_distance_std),
            "relative_distance": float(relative_distance_mean),
            "relative_distance_std": float(relative_distance_std),
            "overlap": float(overlap_mean),
            "overlap_std": float(overlap_std),
        }
    }


def _export_averaged_runs(
    job_dirs: Sequence[Path],
    aggregated_series: Sequence[AggregatedSeries],
    output_dir: Path | None = None,
) -> Path:
    average_job_dir = output_dir or _average_job_directory(job_dirs)
    average_job_dir.mkdir(parents=True, exist_ok=True)

    for series in aggregated_series:
        averaged_run_dir = average_job_dir / series.records[0].run_dir.name
        _write_yaml(
            averaged_run_dir / "simulation_config.yaml",
            _build_avg_simulation_config(series, len(job_dirs)),
        )
        _write_json(averaged_run_dir / "training_log.json", _build_avg_training_log(series))

        averaged_weight_metrics_dir = averaged_run_dir / "avg_final_weight_metrics"
        averaged_weight_metrics_dir.mkdir(parents=True, exist_ok=True)
        metric_paths_by_name: Dict[str, List[Path]] = defaultdict(list)
        for record in series.records:
            for metric_name, metric_path in _collect_weight_metric_files(record.run_dir).items():
                metric_paths_by_name[metric_name].append(metric_path)

        for metric_name in sorted(metric_paths_by_name):
            metric_paths = metric_paths_by_name[metric_name]
            if len(metric_paths) < len(series.records):
                print(
                    f"Warning: averaging weight metrics for {series.scheme}/{series.label}/{metric_name} "
                    f"over {len(metric_paths)} of {len(series.records)} matched runs."
                )
            _write_json(
                averaged_weight_metrics_dir / metric_name,
                _aggregate_weight_metric_payloads(metric_paths),
            )

    return average_job_dir


def avg_runs(job_dirs: Sequence[Path], layer: str, compute_flag: bool, output_dir: Path | None = None) -> Path:
    if not job_dirs:
        raise ValueError("Please provide at least one job directory.")

    all_log_paths: Dict[Path, List[Path]] = {}
    for job_dir in job_dirs:
        log_paths = collect_files_with_ending(job_dir, "training_log.json")
        if not log_paths:
            raise FileNotFoundError(f"No training logs found in {job_dir}")
        all_log_paths[job_dir] = sorted(
            log_paths,
            key=lambda path: tuple(_width_key(_label_from_log_path(path))),
        )

    ordered_keys: List[Tuple[str, str]] = []
    seen_keys: set[Tuple[str, str]] = set()
    series_records: List[RunSeries] = []
    for job_dir in job_dirs:
        for log_path in all_log_paths[job_dir]:
            record = _collect_run_series(job_dir, log_path, layer, compute_flag)
            series_records.append(record)
            key = (record.scheme, record.label)
            if key not in seen_keys:
                seen_keys.add(key)
                ordered_keys.append(key)

    task_names = {record.task_name for record in series_records}
    if len(task_names) != 1:
        raise ValueError(
            "All provided job directories must correspond to the same task in order to average them."
        )

    grouped_records: Dict[Tuple[str, str], List[RunSeries]] = defaultdict(list)
    for record in series_records:
        grouped_records[(record.scheme, record.label)].append(record)

    aggregated_series = [
        _aggregate_group(key, grouped_records[key], len(job_dirs)) for key in ordered_keys
    ]
    return _export_averaged_runs(job_dirs, aggregated_series, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average multiple jobs into a new avg job directory."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to the job directories that should be averaged.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="all_weights",
        help="Layer used to validate matching x-axes while averaging.",
    )
    parser.add_argument(
        "--compute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret the x-axis as compute when validating runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for the averaged job.",
    )

    args = parser.parse_args()
    average_job_dir = avg_runs(args.log_dir, args.layer, args.compute, args.output_dir)
    print(f"Saved averaged runs to {average_job_dir}")


if __name__ == "__main__":
    main()
