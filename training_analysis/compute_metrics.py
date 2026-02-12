from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import zarr
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root) + "/src")
import yaml
from tqdm import tqdm
import numpy as np 


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
      first_vec: flattened weights at t=0
      last_vec:  flattened weights at t=T-1
      deltas():  generator yielding (t, delta) for t=1..T-1 where
                delta = vec(t) - vec(t-1)
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
    last_vec = read_flat(T - 1)

    return first_vec, last_vec

def compute_deltas(weight_history: List[Any], layer: str):
    flattened = [_flatten_weights(snapshot, layer) for snapshot in weight_history]
    # flattened = [np.asarray(snapshot[layer]["kernel"], dtype=float).ravel() for snapshot in weight_history]
    # flattened = [_flatten_weights(snapshot) for snapshot in weight_history]
    lengths = {vec.shape[0] for vec in flattened}
    if len(lengths) != 1:
        raise ValueError("Weight snapshots do not all have the same size.")

    deltas = [flattened[i] - flattened[i - 1] for i in range(1, len(flattened))]
    return deltas, flattened[0], flattened[-1]

def _cumulative_and_actual_path_length(deltas: List[Any], initial_weights: np.ndarray, final_weights: np.ndarray) -> Tuple[float, float, float]:
    step_norms: List = []
    for i in range(0, len(deltas)):
        step_norms.append(np.linalg.norm(deltas[i]))
    step_norms = np.array(step_norms)
    cum_path_length = np.sum(step_norms)
    delta_T = np.linalg.norm(final_weights - initial_weights)
    theta_0 = np.linalg.norm(initial_weights)
    normalized_distance = delta_T / theta_0
    relative_distance = delta_T / cum_path_length
    overlap = float(np.sqrt(np.dot(initial_weights, final_weights)) / theta_0)
    return step_norms, cum_path_length, normalized_distance, relative_distance, overlap

def _cosine_similarity(deltas: List[Any]) -> np.ndarray:
    sims: List[float] = []
    for i in range(1, len(deltas)):
        a, b = deltas[i], deltas[i - 1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(float(np.dot(a, b) / denom) if denom > 0 else np.nan)
    if len(sims) == 1:
        return sims[0]
    else:
        return np.array(sims)

def compute_metrics(history, weights_path, layer):
    if "weights" in history:
        weights = history.get("weights")
        deltas, initial_weights, final_weights = compute_deltas(weights, layer)
        similarities = _cosine_similarity(deltas)
        step_norms, cum_path_length, normalized_distance, relative_distance, overlap = _cumulative_and_actual_path_length(deltas, initial_weights, final_weights)
    else: 
        step_norms: List[float] = []
        similarities: List[float] = []
        dw_prev = None
        for t, dw in iter_layer_weight_deltas(f"{weights_path}/weights.zarr", layer, include=("kernel")):
            step_norms.append(float(np.linalg.norm(dw)))
            if dw_prev is None:
                dw_prev = dw
                pass
            else:
                cs = _cosine_similarity([dw_prev, dw])
                similarities.append(float(cs))
                dw_prev = dw
        
        step_norms = step_norms[1:]
        similarities = similarities[1:]
        cum_path_length = float(np.sum(step_norms))
        initial_weights, final_weights = layer_vectors_first_last_and_deltas(f"{weights_path}/weights.zarr", layer, include=("kernel"))
        delta_T = np.linalg.norm(final_weights - initial_weights)
        theta_0 = np.linalg.norm(initial_weights)
        normalized_distance = float(delta_T / theta_0)
        relative_distance = float(delta_T / cum_path_length)
        overlap = float(np.sqrt(np.dot(initial_weights, final_weights)) / theta_0)         

    return step_norms, cum_path_length, normalized_distance, relative_distance, overlap, similarities

def analyze_training_run(log_dir: Path, output_dir: Path, layers: str) -> None:
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

    for i,layer in tqdm(enumerate(layers)):
        step_norms, cum_path_length, normalized_distance, relative_distance, overlap, cs = compute_metrics(history, log_dir.parent, layer)

        training_metrics = {
            "step_norms": list(step_norms),
            "similarities": list(cs),
            "cum_path_length": float(cum_path_length),
            "normalized_distance": float(normalized_distance),
            "relative_distance": float(relative_distance),
            "overlap": float(overlap),
        }

        data = {"training_metrics": training_metrics}

        pbar.total = len(layers)
        pbar.n = i + 1
        pbar.set_postfix("", refresh=False)
        pbar.refresh()

        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"{layer}.json").open("w") as f:
            json.dump(data, f, indent=2)

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
    analyze_training_run(args.log_dir, args.output, args.layers)


if __name__ == "__main__":
    main()

        


