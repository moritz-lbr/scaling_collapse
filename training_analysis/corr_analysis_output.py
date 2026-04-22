from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from tqdm import tqdm
import zarr  # type: ignore


def load_training_log(log_path: Path) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_log_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "training_log.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"No training_log.json found in {path}")
        return candidate
    return path


def load_term_history(
    zarr_path: str | Path,
    sample_index: int,
    *,
    dtype=np.float32,
) -> np.ndarray:
    root = zarr.open_group(Path(zarr_path), mode="r")

    if "logits" in root.array_keys():
        terms = root["logits"]
        term_path = "logits"
    elif "Dense_0" in root.group_keys() and "logits" in root["Dense_0"].array_keys():
        terms = root["Dense_0"]["logits"]
        term_path = "Dense_0/terms"
    else:
        raise ValueError(
            "Dataset 'terms' not found at the Zarr root and legacy dataset 'Dense_0/logits' "
            f"is also missing. Available root groups: {list(root.group_keys())}, "
            f"root arrays: {list(root.array_keys())}"
        )

    if len(terms.shape) != 3:
        raise ValueError(
            f"Expected dataset shape (T, N, width_0) for {term_path}, got {terms.shape}"
        )

    num_steps, num_samples, width = terms.shape
    if sample_index < 1 or sample_index > num_samples:
        raise ValueError(
            f"sample_index must be between 1 and {num_samples}, got {sample_index}."
        )

    selected = np.asarray(terms[:, sample_index - 1, :], dtype=dtype)
    if selected.shape[0] != num_steps:
        raise ValueError(
            f"Unexpected selected term history shape {selected.shape}; expected first dimension {num_steps}."
        )

    # return np.log(np.abs(selected)+1e-12)  # Add small constant to avoid log(0)
    return selected


def compute_running_covariances_and_spectral_radii(
    xj: np.ndarray,
    delta_t: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-step covariance matrices for xj with shape (T, D).
    Output shapes are (T - delta_t, D, D) for covariance matrices and
    (T - delta_t,) for spectral radii, where cov[t] uses rows
    xj[t:t + delta_t].
    """
    if xj.ndim != 2:
        raise ValueError(f"Expected xj with shape (T, D), got {xj.shape}")
    if delta_t < 1:
        raise ValueError(f"delta_t must be >= 1, got {delta_t}")

    num_steps, num_features = xj.shape
    if num_features < 1:
        raise ValueError(f"Expected at least one feature, got {num_features}")
    num_windows = num_steps - delta_t
    if num_windows < 0:
        raise ValueError(
            f"delta_t must be <= number of steps ({num_steps}), got {delta_t}"
        )

    cov = np.empty((num_windows, num_features, num_features), dtype=np.float32)
    spec_rad = np.empty((num_windows,), dtype=np.float32)

    for t in range(num_windows):
        cov_t = np.atleast_2d(
            np.cov(xj[t : t + delta_t].T, rowvar=True, bias=True)
        ).astype(np.float32, copy=False)
        cov[t] = cov_t
        spec_rad[t] = np.linalg.eigvalsh(cov_t)[-1] / num_features

    return cov, spec_rad


def compute_running_covariances(xj: np.ndarray, delta_t: int) -> np.ndarray:
    cov, _ = compute_running_covariances_and_spectral_radii(xj, delta_t)
    return cov


def analyze_training_run(
    log_dir: Path,
    delta_t: int,
    output_dir: Path | None,
    sample_index: int,
) -> None:
    log_path = resolve_log_path(log_dir)
    log_data = load_training_log(log_path)
    history = log_data.get("final_metrics", {}).get("history", {})
    losses = history.get("train_loss")
    if not losses:
        raise ValueError(f"No loss history found in {log_path}")

    pbar = tqdm(
        total=1,
        desc="Computing output metrics",
        unit="step",
        leave=False,
    )

    output_dir = Path(".") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = log_path.parent
    xj = load_term_history(f"{weights_path}/weights.zarr", sample_index)
    cov, spec_rad = compute_running_covariances_and_spectral_radii(xj, delta_t)

    pbar.update(1)
    pbar.close()

    np.save(output_dir / f"xj_logits_sample_{sample_index}.npy", xj)
    np.save(output_dir / f"cov_logits_sample_{sample_index}.npy", cov)
    np.save(output_dir / f"spec_rad_logits_sample_{sample_index}.npy", spec_rad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sliding-window covariance matrices for one stored output-term sample."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to a run directory or its training_log.json file.",
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        required=True,
        help="Time window for computing running covariances.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        required=True,
        help="Which stored sample to analyze, using 1-based indexing (1..5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional directory for the saved .npy files.",
    )

    args = parser.parse_args()
    analyze_training_run(args.log_dir, args.delta_t, args.output, args.sample_index)


if __name__ == "__main__":
    main()
