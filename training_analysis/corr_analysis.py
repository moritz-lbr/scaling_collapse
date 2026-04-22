from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import zarr # type: ignore
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root) + "/src")
import yaml
from tqdm import tqdm
import numpy as np 
from sklearn.feature_selection import mutual_info_regression


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

def _flatten_weights(entry: Any, layer: str) -> np.ndarray:
    arrays: List[np.ndarray] = []

    def traverse(node: Any) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                traverse(node[key])
            return
        
        # arr = np.asarray(node[layer]["kernel"], dtype=float)
        arr = np.asarray(node, dtype=float)
        arrays.append(arr.ravel())

    if layer == "all_weights":
        traverse(entry)
    else:
        arr = np.asarray(entry[layer]["kernel"], dtype=float)
        arrays.append(arr.ravel())

    if not arrays:
        return np.array([], dtype=float)
    return np.concatenate(arrays)

def print_zarr_contents(zarr_path: str | Path, max_lines=200):
    root = zarr.open_group(Path(zarr_path), mode="r")
    print("Zarr store:", Path(zarr_path).resolve())
    print("Root groups:", list(root.group_keys()))
    print("Root arrays :", list(root.array_keys()))
    # This prints a tree view (usually very informative)
    try:
        print(root.tree())
    except Exception as e:
        print("Could not print tree():", e)

def iter_layer_weight_deltas(
    zarr_path: str | Path,
    layer_group: str,
    *,
    include: tuple[str, ...] = ("kernel", "bias"),
    dtype=np.float32,
):
    """
    Yields (t, delta) for t=1..T-1 where:
      delta = flattened_weights(t) - flattened_weights(t-1)
    """

    if isinstance(include, str):
        include = (include,)
    else:
        include = tuple(include)

    root = zarr.open_group(Path(zarr_path), mode="r")

    if layer_group == "all_weights":
        groups = sorted(list(root.group_keys()))
    else:
        groups = [layer_group]

    dsets = []
    used = []

    for grp in groups:
        if grp not in root.group_keys():
            raise ValueError(
                f"Layer group '{grp}' not found. Available groups: {list(root.group_keys())}"
            )
        g = root[grp]
        available = list(g.array_keys())
        grp_used = [name for name in include if name in available]
        if not grp_used:
            continue

        for name in grp_used:
            dsets.append((grp, name, g[name]))
            used.append(f"{grp}/{name}")

    if not dsets:
        raise ValueError(
            f"No arrays from {include} found in selected groups: {groups}. "
            f"Available groups: {list(root.group_keys())}"
        )


    # infer T and sanity check
    T = dsets[0][2].shape[0]
    if T < 1:
        raise ValueError("No time steps stored (T=0).")

    sizes = []
    sizes = []
    for grp, name, ds in dsets:
        if ds.shape[0] != T:
            raise ValueError(f"Time dimension mismatch for {grp}/{name}: {ds.shape[0]} vs {T}")
        sizes.append(int(np.prod(ds.shape[1:], dtype=np.int64)))


    total = int(np.sum(sizes, dtype=np.int64))

    def read_flat(t: int) -> np.ndarray:
        out = np.empty((total,), dtype=dtype)
        offset = 0
        for (_, _, ds), sz in zip(dsets, sizes):
            arr_t = np.asarray(ds[t])
            flat = arr_t.reshape(-1)
            out[offset:offset + sz] = flat.astype(dtype, copy=False)
            offset += sz
        return out

    first_vec = read_flat(0)

    
    prev = first_vec
    for t in range(1, T):
        curr = read_flat(t)
        delta = curr - prev
        yield t, delta
        prev = curr
 

def layer_vectors_first_last_and_deltas(
    zarr_path: str | Path,
    layer_group: str,
    *,
    include: tuple[str, ...] = ("kernel", "bias"),
    dtype=np.float32,
):
    """
    Returns:
      delta_matrix: shape (T-1, row_dim), where each row is
                    sum_over_columns(W_t - W_{t-1}) for one time step.
    """

    if isinstance(include, str):
        include = (include,)
    else:
        include = tuple(include)

    root = zarr.open_group(Path(zarr_path), mode="r")

    if layer_group == "all_weights":
        groups = sorted(list(root.group_keys()))
    else:
        groups = [layer_group]

    dsets = []
    used = []

    for grp in groups:
        if grp not in root.group_keys():
            raise ValueError(
                f"Layer group '{grp}' not found. Available groups: {list(root.group_keys())}"
            )
        g = root[grp]
        available = list(g.array_keys())
        grp_used = [name for name in include if name in available]
        if not grp_used:
            continue

        for name in grp_used:
            dsets.append((grp, name, g[name]))
            used.append(f"{grp}/{name}")

    if not dsets:
        raise ValueError(
            f"No arrays from {include} found in selected groups: {groups}. "
            f"Available groups: {list(root.group_keys())}"
        )


    # infer T and sanity check
    T = dsets[0][2].shape[0]
    if T < 1:
        raise ValueError("No time steps stored (T=0).")

    for grp, name, ds in dsets:
        if ds.shape[0] != T:
            raise ValueError(f"Time dimension mismatch for {grp}/{name}: {ds.shape[0]} vs {T}")

    if T < 2:
        row_dim: int | None = None
        for grp, name, ds in dsets:
            if len(ds.shape) != 3:
                raise ValueError(
                    f"Expected dataset shape (T, rows, cols) for {grp}/{name}, got {ds.shape}"
                )
            if row_dim is None:
                row_dim = ds.shape[1]
            elif row_dim != ds.shape[1]:
                raise ValueError(
                    f"Row dimension mismatch across arrays: {row_dim} vs {ds.shape[1]}"
                )
        return np.empty((0, int(row_dim)), dtype=dtype)

    def delta_column_sum(curr_t: int, prev_t: int) -> np.ndarray:
        """Compute sum_over_columns(W_curr - W_prev), preserving row dimension."""
        row_dim: int | None = None
        collected: List[np.ndarray] = []

        for grp, name, ds in dsets:
            curr = np.asarray(ds[curr_t], dtype=dtype)
            prev = np.asarray(ds[prev_t], dtype=dtype)

            if curr.ndim != 2 or prev.ndim != 2:
                raise ValueError(
                    f"Expected 2D weight matrix for {grp}/{name}, got {curr.shape} and {prev.shape}"
                )

            if curr.shape != prev.shape:
                raise ValueError(
                    f"Shape mismatch between consecutive steps for {grp}/{name}: "
                    f"{curr.shape} vs {prev.shape}"
                )

            if row_dim is None:
                row_dim = curr.shape[0]
            elif row_dim != curr.shape[0]:
                raise ValueError(
                    f"Row dimension mismatch across arrays: {row_dim} vs {curr.shape[0]}"
                )

            delta = curr - prev
            delta_squared = delta**2
            if grp == "Dense_1":
                delta_squared = delta_squared.T

            # if grp == dsets[0][-1]:
            #     delta_sum = delta_squared.sum(axis=1)
            # else:
            #     delta_sum = delta_squared.sum(axis=0)
            delta_sum = delta_squared.sum(axis=0)
            log_deltas = np.log(delta_sum + 1e-12)  
            # Keep the output width dimension (32 for Dense_0 in this dataset).
            collected.append(log_deltas)

        if len(collected) == 1:
            return collected[0].astype(dtype, copy=False)

        return np.stack(collected, axis=0).sum(axis=0).astype(dtype, copy=False)

    first_delta = delta_column_sum(1, 0)
    delta_matrix = np.empty((T - 1, first_delta.shape[0]), dtype=dtype)
    delta_matrix[0] = first_delta
    for t in range(2, T):
        delta_matrix[t - 1] = delta_column_sum(t, t - 1)

    return delta_matrix

def compute_deltas(weight_history: List[Any], layer: str):
    flattened = [_flatten_weights(snapshot, layer) for snapshot in weight_history]
    # flattened = [np.asarray(snapshot[layer]["kernel"], dtype=float).ravel() for snapshot in weight_history]
    # flattened = [_flatten_weights(snapshot) for snapshot in weight_history]
    lengths = {vec.shape[0] for vec in flattened}
    if len(lengths) != 1:
        raise ValueError("Weight snapshots do not all have the same size.")

    deltas = [flattened[i] - flattened[i - 1] for i in range(1, len(flattened))]
    return np.array(deltas), flattened[0], flattened[-1]


def compute_running_covariances_and_spectral_radii(
    xj: np.ndarray,
    delta_t: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-step covariance matrices for xj with shape (T, D).
    Output shapes are (T - delta_t, D, D) for covariance matrices and
    (T - delta_t,) for spectral radii, where cov[t] uses rows
    xj[t:t+delta_t].
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
    cov_log = np.empty((num_windows, num_features, num_features), dtype=np.float32)
    spec_rad = np.empty((num_windows,), dtype=np.float32)

    xj = xj - np.mean(xj, axis=1, keepdims=True)

    for t in range(num_windows):
        # bias=True avoids NaNs at t=0 (single sample -> zeros matrix).
        cov_t = np.atleast_2d(
            np.cov(xj[t: t + delta_t].T, rowvar=True, bias=True)
        ).astype(np.float32, copy=False)
        cov[t] = cov_t

        # window_log = xj[t : t + delta_t]

        # cov_log_t = np.atleast_2d(
        # np.cov(window_log.T, rowvar=True, bias=True)
        # ).astype(np.float32, copy=False)
        # cov_log[t] = cov_log_t

        # mean_log = window_log.mean(axis=0).astype(np.float32, copy=False)

        # # E[g_i] for lognormal variables: exp(mu_i + 0.5 * var_i)
        # exp_g = np.exp(mean_log + 0.5 * np.diag(cov_log_t)).astype(np.float32, copy=False)

        # cov_t = np.outer(exp_g, exp_g) * (np.exp(cov_log_t) - 1.0)
        # cov[t] = cov_t

        spec_rad[t] = np.linalg.eigvalsh(cov_t)[-1] / num_features

    return cov, spec_rad


def compute_running_covariances(xj: np.ndarray, delta_t: int) -> np.ndarray:
    cov, _ = compute_running_covariances_and_spectral_radii(xj, delta_t)
    return cov


def analyze_training_run(log_dir: Path, delta_t: int, output_dir: Path, layers: str) -> None:
    log_data = load_training_log(log_dir)
    history = log_data.get("final_metrics").get("history")
    losses = history.get("train_loss")
    if not losses:
        raise ValueError(f"No loss history found in {log_dir}")    


    pbar = tqdm(
                total=len(layers),
                desc=f"Computing weight metrics",
                unit="step",
                leave=False
            )

    output_dir = Path(".") if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, layer in tqdm(enumerate(layers), total=len(layers)):
        weights_path = log_dir.parent
        xj = layer_vectors_first_last_and_deltas(
            f"{weights_path}/weights.zarr", layer, include=("kernel",)
        )
        cov, spec_rad = compute_running_covariances_and_spectral_radii(xj, delta_t)

        pbar.total = len(layers)
        pbar.n = layer_idx + 1
        pbar.set_postfix("", refresh=False)
        pbar.refresh()

        np.save(output_dir / f"xj_{layer}.npy", xj)
        np.save(output_dir / f"cov_{layer}.npy", cov)
        np.save(output_dir / f"spec_rad_{layer}.npy", spec_rad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cosine similarity between consecutive weight updates for all runs in a job."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Path to a job directory containing training_log.json files.",
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        required=True,
        help="Time window for computing running covariances.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional filename to save the plot under figures_weight_update_similarity/.",
        default=None,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=str,
        default=["all_weights"],
        help="Optional flag to plot the weight metrics only for the specified layer of the network. Each network layer can be adressed as: 'Dense_0' ... 'Dense_N'"
    )

    args = parser.parse_args()
    analyze_training_run(args.log_dir, args.delta_t, args.output, args.layers)


if __name__ == "__main__":
    main()

        
