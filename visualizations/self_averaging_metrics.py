"""Compute self-averaging ratios from saved scalar training observables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training_analysis.mb_cov_utils import LocalZarrArray, load_yaml_as_dict


LAYERS = ("Dense_0", "Dense_1")
TEST_LOSS_COMPONENT = "test_loss"
OBSERVABLE_UPDATE_NORM = "update_norm"
OBSERVABLE_TEST_LOSS = "test_loss"
OBSERVABLES = (OBSERVABLE_UPDATE_NORM, OBSERVABLE_TEST_LOSS)
OUTPUT_FILENAMES = {
    OBSERVABLE_UPDATE_NORM: "self_av_layer.json",
    OBSERVABLE_TEST_LOSS: "self_av_test_loss.json",
}
OUTPUT_FILENAME = OUTPUT_FILENAMES[OBSERVABLE_UPDATE_NORM]
DEFAULT_WINDOW_WIDTH = 10
DEFAULT_WINDOW_STRIDE = 10


def _safe_json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_save_loss_frequency(simulation_config_path: Path) -> int:
    simulation_info = load_yaml_as_dict(simulation_config_path)
    value = simulation_info.get("training", {}).get("save_loss_frequency", 1)
    if isinstance(value, str):
        if value.strip().lower() == "epoch":
            return 1
        return int(float(value))
    return int(value)


def _hidden_width_from_config(simulation_config_path: Path) -> int | None:
    if not simulation_config_path.exists():
        return None
    simulation_info = load_yaml_as_dict(simulation_config_path)
    network_info = simulation_info.get("network", {})
    for key in ("nodes_per_layer", "base_layer_width", "base_layer_widths"):
        widths = network_info.get(key)
        if isinstance(widths, dict) and "Dense_0" in widths:
            return int(widths["Dense_0"])
    return None


def dense0_width_ratio_from_config(simulation_config_path: Path) -> float:
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml: {simulation_config_path}")
    simulation_info = load_yaml_as_dict(simulation_config_path)
    network_info = simulation_info.get("network", {})
    nodes_per_layer = network_info.get("nodes_per_layer", {})
    base_widths = network_info.get("base_layer_width") or network_info.get(
        "base_layer_widths"
    )
    if not isinstance(nodes_per_layer, dict) or "Dense_0" not in nodes_per_layer:
        raise ValueError(
            f"Missing network.nodes_per_layer.Dense_0 in {simulation_config_path}"
        )

    if isinstance(base_widths, dict):
        base_width = base_widths.get("Dense_0")
    else:
        base_width = base_widths
    if base_width is None:
        raise ValueError(f"Missing Dense_0 base layer width in {simulation_config_path}")

    ratio = float(nodes_per_layer["Dense_0"]) / float(base_width)
    if ratio <= 0.0:
        raise ValueError(
            f"Dense_0 width/base-width ratio must be positive in {simulation_config_path}"
        )
    return ratio


def update_norm_rescale_factor(layer: str, dense0_width_ratio: float) -> float:
    if layer == "Dense_0":
        return 1.0 / float(dense0_width_ratio)
    if layer == "Dense_1":
        return float(dense0_width_ratio)
    raise ValueError(f"Unsupported layer {layer!r}.")


def rescale_update_norms(
    values: np.ndarray,
    layer: str,
    dense0_width_ratio: float,
) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * update_norm_rescale_factor(
        layer,
        dense0_width_ratio,
    )


def _load_training_metrics_block(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    metrics = data.get("training_metrics")
    if metrics is None:
        metrics = data.get("avg_training_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"No training metrics found in {path}")
    return metrics


def _load_step_norms_from_json(run_dir: Path, layer: str) -> Tuple[np.ndarray, str] | None:
    for directory_name in ("weight_metrics", "avg_final_weight_metrics"):
        path = run_dir / directory_name / f"{layer}.json"
        if not path.exists():
            continue
        metrics = _load_training_metrics_block(path)
        step_norms = metrics.get("step_norms")
        if step_norms is None:
            continue
        return np.asarray(step_norms, dtype=np.float64), str(path)
    return None


def _zarr_step_norm_candidates(metrics_path: Path, layer: str) -> List[Path]:
    return [
        metrics_path / layer / "step_norms",
        metrics_path / layer / "step_norm",
        metrics_path / layer / "training_metrics" / "step_norms",
        metrics_path / layer / "training_metrics" / "step_norm",
        metrics_path / "training_metrics" / layer / "step_norms",
        metrics_path / "training_metrics" / layer / "step_norm",
        metrics_path / "step_norms" / layer,
        metrics_path / "step_norm" / layer,
        metrics_path / layer / "step_norms_mean",
        metrics_path / "step_norms_mean" / layer,
    ]


def _load_zarr_vector(array_path: Path) -> np.ndarray:
    store = LocalZarrArray.open(array_path)
    values = np.asarray(store.read_all(), dtype=np.float64)
    return values.reshape(-1)


def _load_step_norms_from_zarr(run_dir: Path, layer: str) -> Tuple[np.ndarray, str] | None:
    metrics_path = run_dir / "weight_metrics.zarr"
    if not metrics_path.exists():
        return None

    for array_path in _zarr_step_norm_candidates(metrics_path, layer):
        if (array_path / "zarr.json").exists():
            return _load_zarr_vector(array_path), str(array_path)

    for metadata_path in metrics_path.rglob("zarr.json"):
        array_path = metadata_path.parent
        parts = set(array_path.relative_to(metrics_path).parts)
        if layer in parts and (
            "step_norms" in parts or "step_norm" in parts or "step_norms_mean" in parts
        ):
            return _load_zarr_vector(array_path), str(array_path)

    return None


def load_step_norms(run_dir: Path, layer: str) -> Tuple[np.ndarray, str]:
    """Load the saved squared update norms for one layer."""

    run_dir = run_dir.resolve()
    from_zarr = _load_step_norms_from_zarr(run_dir, layer)
    if from_zarr is not None:
        return from_zarr
    from_json = _load_step_norms_from_json(run_dir, layer)
    if from_json is not None:
        return from_json
    raise FileNotFoundError(
        f"No saved step_norms found for {layer} in {run_dir}. "
        "Expected weight_metrics.zarr or weight_metrics/<layer>.json."
    )


def load_test_loss(run_dir: Path) -> Tuple[np.ndarray, str]:
    log_path = run_dir.resolve() / "training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training_log.json in {run_dir}")
    log_data = _read_json(log_path)
    history = log_data.get("final_metrics", {}).get("history", {})
    test_loss = history.get("test_loss")
    if test_loss is None:
        raise ValueError(f"No final_metrics.history.test_loss found in {log_path}")
    return np.asarray(test_loss, dtype=np.float64), str(log_path)


def _validate_window(window_width: int, window_stride: int) -> None:
    if window_width < 2:
        raise ValueError(f"window_width must be at least 2, got {window_width}.")
    if window_stride < 1:
        raise ValueError(f"window_stride must be at least 1, got {window_stride}.")


def _centered_window_slices(
    num_values: int,
    window_width: int,
    window_stride: int,
) -> List[Tuple[int, int, int]]:
    _validate_window(window_width, window_stride)
    if window_width > num_values:
        raise ValueError(
            f"Cannot build a centered window of width {window_width} "
            f"from {num_values} observable values."
        )

    center_offset = window_width // 2
    output_len = num_values - window_width + 1
    centers = np.arange(
        center_offset,
        center_offset + output_len,
        window_stride,
        dtype=np.int64,
    )
    return [
        (int(center), int(center - center_offset), int(center - center_offset + window_width))
        for center in centers
    ]


def self_averaging_stats(values: Sequence[float]) -> Tuple[float | None, float | None, float | None, float | None]:
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_values = values_array[np.isfinite(values_array)]
    if finite_values.size < 2:
        return None, None, None, None
    variance = float(np.var(finite_values))
    mean = float(np.mean(finite_values))
    squared_first_moment = float(mean**2)
    ratio = variance / squared_first_moment if squared_first_moment > 0.0 else None
    return ratio, variance, squared_first_moment, mean


def _training_steps_for_update_norms(num_values: int, save_loss_frequency: int) -> np.ndarray:
    return np.arange(1, num_values + 1, dtype=np.int64) * int(save_loss_frequency)


def _training_steps_for_logged_losses(num_values: int, save_loss_frequency: int) -> np.ndarray:
    return np.arange(num_values, dtype=np.int64) * int(save_loss_frequency)


def compute_series_self_averaging_from_local_windows(
    values: np.ndarray,
    training_steps: np.ndarray,
    *,
    window_width: int,
    window_stride: int,
) -> Dict[str, Any]:
    if values.shape[0] != training_steps.shape[0]:
        raise ValueError(
            f"Observable and training_steps length mismatch: "
            f"{values.shape[0]} vs {training_steps.shape[0]}."
        )

    metrics: Dict[str, Any] = {
        "training_steps": [],
        "observable_indices": [],
        "window_start_indices": [],
        "window_stop_indices": [],
        "R": [],
        "variance": [],
        "squared_first_moment": [],
        "mean": [],
        "sample_count": [],
    }

    for center, start, stop in _centered_window_slices(
        int(values.shape[0]),
        window_width,
        window_stride,
    ):
        window_values = values[start:stop]
        finite_count = int(np.isfinite(window_values).sum())
        ratio, variance, squared_first_moment, mean = self_averaging_stats(window_values)
        metrics["training_steps"].append(int(training_steps[center]))
        metrics["observable_indices"].append(int(center))
        metrics["window_start_indices"].append(int(start))
        metrics["window_stop_indices"].append(int(stop))
        metrics["R"].append(None if ratio is None else _safe_json_float(ratio))
        metrics["variance"].append(None if variance is None else _safe_json_float(variance))
        metrics["squared_first_moment"].append(
            None if squared_first_moment is None else _safe_json_float(squared_first_moment)
        )
        metrics["mean"].append(None if mean is None else _safe_json_float(mean))
        metrics["sample_count"].append(finite_count)

    return metrics


def _series_for_observable(
    run_dir: Path,
    observable: str,
    save_loss_frequency: int,
    simulation_config_path: Path,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, str], Dict[str, Any]]:
    if observable == OBSERVABLE_UPDATE_NORM:
        dense0_width_ratio = dense0_width_ratio_from_config(simulation_config_path)
        series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        sources: Dict[str, str] = {}
        rescaling: Dict[str, Any] = {
            "mode": "dense0_width_ratio",
            "dense0_width_ratio": float(dense0_width_ratio),
            "layers": {},
        }
        for layer in LAYERS:
            values, source = load_step_norms(run_dir, layer)
            values = rescale_update_norms(values, layer, dense0_width_ratio)
            series[layer] = (
                values,
                _training_steps_for_update_norms(int(values.shape[0]), save_loss_frequency),
            )
            sources[layer] = source
            scale = update_norm_rescale_factor(layer, dense0_width_ratio)
            rescaling["layers"][layer] = {
                "scale_factor": float(scale),
                "rule": (
                    "step_norm / (Dense_0 width / Dense_0 base width)"
                    if layer == "Dense_0"
                    else "step_norm * (Dense_0 width / Dense_0 base width)"
                ),
            }
        return series, sources, rescaling

    if observable == OBSERVABLE_TEST_LOSS:
        values, source = load_test_loss(run_dir)
        return {
            TEST_LOSS_COMPONENT: (
                values,
                _training_steps_for_logged_losses(int(values.shape[0]), save_loss_frequency),
            )
        }, {TEST_LOSS_COMPONENT: source}, {"mode": "none"}

    raise ValueError(f"Unsupported observable {observable!r}.")


def compute_self_averaging_metrics(
    run_dir: Path,
    *,
    observable: str = OBSERVABLE_UPDATE_NORM,
    output_name: str | None = None,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
) -> Path:
    run_dir = run_dir.resolve()
    if observable not in OBSERVABLES:
        raise ValueError(f"Unsupported observable {observable!r}.")
    simulation_config_path = run_dir / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Missing simulation_config.yaml in {run_dir}")
    _validate_window(window_width, window_stride)

    save_loss_frequency = _load_save_loss_frequency(simulation_config_path)
    hidden_width = _hidden_width_from_config(simulation_config_path)
    raw_series, sources, rescaling = _series_for_observable(
        run_dir,
        observable,
        save_loss_frequency,
        simulation_config_path,
    )

    series_metrics: Dict[str, Any] = {}
    for component, (values, training_steps) in raw_series.items():
        series_metrics[component] = compute_series_self_averaging_from_local_windows(
            values,
            training_steps,
            window_width=window_width,
            window_stride=window_stride,
        )

    output_dir = run_dir / "weight_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (output_name or OUTPUT_FILENAMES[observable])
    data = {
        "created_by": "visualizations/self_averaging_metrics.py",
        "definition": (
            "For a single run, X_j are scalar observable values in a local time window "
            "around the reported time step. R = Var_j(X_j) / E_j[X_j]^2."
        ),
        "observable": observable,
        "estimator": "local_time_window",
        "run_dir": str(run_dir),
        "save_loss_frequency": int(save_loss_frequency),
        "hidden_width": None if hidden_width is None else int(hidden_width),
        "window_width": int(window_width),
        "window_stride": int(window_stride),
        "source_values": sources,
        "rescaling": rescaling,
        "series": series_metrics,
    }
    if observable == OBSERVABLE_UPDATE_NORM:
        data["layers"] = series_metrics
        data["source_step_norms"] = sources
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute self-averaging ratios from saved scalar observables."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing training_log.json and/or weight_metrics.",
    )
    parser.add_argument(
        "--observable",
        choices=OBSERVABLES,
        default=OBSERVABLE_UPDATE_NORM,
        help="Observable to analyze.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Filename written under weight_metrics/. Defaults depend on --observable.",
    )
    parser.add_argument(
        "--window-width",
        "--window_width",
        dest="window_width",
        type=int,
        default=DEFAULT_WINDOW_WIDTH,
        help="Number of consecutive observable values in each centered local window.",
    )
    parser.add_argument(
        "--window-stride",
        "--window_stride",
        dest="window_stride",
        type=int,
        default=DEFAULT_WINDOW_STRIDE,
        help="Stride, in saved observable indices, between evaluated window centers.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Accepted for compatibility; this script does not show a progress bar.",
    )
    args = parser.parse_args()

    try:
        _validate_window(args.window_width, args.window_stride)
    except ValueError as exc:
        parser.error(str(exc))

    output_path = compute_self_averaging_metrics(
        args.run_dir,
        observable=args.observable,
        output_name=args.output_name,
        window_width=args.window_width,
        window_stride=args.window_stride,
    )
    print(f"Saved self-averaging metrics to {output_path}")


if __name__ == "__main__":
    main()
