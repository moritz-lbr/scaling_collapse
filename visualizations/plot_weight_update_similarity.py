"""Plot cosine similarity between successive weight updates for all runs in a job."""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import zarr
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root) + "/src")
from tqdm import tqdm

from utils import load_yaml_as_dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches


def extract_after_char(s: str, start: str, stop: str) -> str:
    return s[s.rfind(start) + 1 : s.rfind(stop)]


def load_training_log(log_path: Path) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _scheme_from_path(log_path: Path) -> str:
    directory_name = log_path.parent.name
    return "standard" if "standard" in directory_name else "muP"


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

def _extract_weights(history_blob: Dict[str, Any]) -> List[Any]:
    history = history_blob.get("history", history_blob)
    weights = history.get("weights")
    return weights if isinstance(weights, list) else []

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
            step_norms.append(np.linalg.norm(dw))
            if dw_prev is None:
                dw_prev = dw
                pass
            else:
                cs = _cosine_similarity([dw_prev, dw])
                similarities.append(cs)
                dw_prev = dw
        
        cum_path_length = np.sum(step_norms)
        initial_weights, final_weights = layer_vectors_first_last_and_deltas(f"{weights_path}/weights.zarr", layer, include=("kernel"))
        delta_T = np.linalg.norm(final_weights - initial_weights)
        theta_0 = np.linalg.norm(initial_weights)
        normalized_distance = delta_T / theta_0
        relative_distance = delta_T / cum_path_length
        overlap = float(np.sqrt(np.dot(initial_weights, final_weights)) / theta_0)         

    return np.array(step_norms), cum_path_length, normalized_distance, relative_distance, overlap, np.array(similarities)


def plot_weight_update_similarity(job_dir: Path, outfile: Path | None, layer: str, compute_flag: bool) -> None:
    log_paths = collect_files_with_ending(job_dir, "training_log.json")
    if not log_paths:
        raise FileNotFoundError(f"No training logs found in {job_dir}")

    log_paths = sorted(
        log_paths,
        key=lambda p: tuple(_width_key(extract_after_char(str(p), "-", "/"))),
    )

    scheme_groups = {
        "standard": [log for log in log_paths if _scheme_from_path(log) == "standard"],
        "muP": [log for log in log_paths if _scheme_from_path(log) == "muP"],
    }

    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}

    node_list: List = []

    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}

    fig, ((cos_log, loss_log), (ax_step_norms, ax_path_lengths)) = plt.subplots(2,2, figsize=(15, 10))

    print("Number of Nodes |", "Cumulated Path Length |", "Normalized Distance |", "Relative Distance |", "Overlap")

    pbar = tqdm(
                total=len(log_paths),
                desc=f"Plotting",
                unit="step",
                leave=False
            )

    for i, log_path in enumerate(log_paths):
        simluation_config_path = collect_files_with_ending(log_path.parent, "simulation_config.yaml")[0]
        simulation_info = load_yaml_as_dict(simluation_config_path)
        
        dataset_info = simulation_info.get("training")
        network_info = simulation_info.get("network")
        
        num_nodes = network_info.get("nodes_per_layer").get("Dense_0")
        node_list.append(num_nodes)

        task_path = dataset_info.get("training_data").get("task")
        task_name = Path(task_path).name

        save_loss_frequency = dataset_info.get("save_loss_frequency")

        log_data = load_training_log(log_path)
        history = log_data.get("final_metrics").get("history")
        losses = history.get("train_loss")
        if not losses:
            raise ValueError(f"No loss history found in {log_path}")        

        step_norms, cum_path_length, normalized_distance, relative_distance, overlap, similarities = compute_metrics(history, log_path.parent, layer)

        pbar.total = len(log_paths)
        pbar.n = i + 1
        pbar.set_postfix("", refresh=False)
        pbar.refresh()

        print("\n")

        print(f"{num_nodes} |", f"{cum_path_length} |", f"{normalized_distance} |", f"{relative_distance} |", overlap)
        
        if similarities.size == 0:
            print(f"Skipping {log_path}: need at least three weight snapshots.")
            continue

        scheme = _scheme_from_path(log_path)
        color = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1

        label = extract_after_char(str(log_path), "-", "/")

        if save_loss_frequency == "epoch":
            save_loss_frequency = 1

        if compute_flag:
            parameters = network_info.get("total_params")
            batch_size = dataset_info.get("batch_size")
            x_axis = np.array(range(1, len(losses) + 1)) * save_loss_frequency * batch_size * parameters
            x_label = "Training Compute" 
        else:
            x_axis = np.array(range(1, len(losses) + 1)) * save_loss_frequency
            x_label = "Training Steps" 

        line, = loss_log.plot(x_axis, losses, color=color)
        cos_log.plot(x_axis[:-1], similarities, color=color, label=label)
        ax_step_norms.plot(x_axis, step_norms, color=color)
        ax_path_lengths.plot(num_nodes, cum_path_length, color=color, marker="o")
        # ax_path_lengths.plot(num_nodes, normalized_distance, color=color, marker="D")
        # ax_path_lengths.plot(num_nodes, overlap, color=color, marker="x")

        legend_entries[scheme].append((line, label))

    ax_path_lengths.plot(num_nodes, cum_path_length, label=r"Cumulative Path Length: $F(T) = \sum_{t_{i}=1}^{T} \| \vec{\theta}_{t_{i+1}} - \vec{\theta}_{t_{i}} \|$", color=color, marker="o")
    # ax_path_lengths.plot(num_nodes, normalized_distance, label=r"Normalized Distance: $\frac{\| \vec{\theta}_{T} - \vec{\theta}_{t_{0}} \|}{\| \vec{\theta}_{t_{0}} \|}$", color=color, marker="D")
    # ax_path_lengths.plot(num_nodes, overlap, color=color, label=r"Overlap: $\frac{\sqrt{\langle \vec{\theta}_{T}, \vec{\theta}_{t_{0}} \rangle}}{\| \vec{\theta}_{t_{0}} \|}$", marker="x")
    path_lengths_legend = ax_path_lengths.legend(fontsize=13)
    for h in path_lengths_legend.legend_handles:
        h.set_color("black")
        h.set_markerfacecolor("black")

    loss_log.set_xscale("log")
    loss_log.set_yscale("log")
    loss_log.set_xlabel(f"{x_label} [log]", fontsize=16)
    loss_log.set_ylabel(f"Training Loss [log]", fontsize=16)
    loss_log.grid(True, which="both", alpha=0.3)
    loss_log.tick_params(axis='both', labelsize=13)

    cos_log.set_xlabel(r"Training Steps $t_{i}$ [log]", fontsize=16)
    cos_log.set_ylabel(r"$\cos(\Delta \vec{\theta}_{t_{i+1}}, \Delta \vec{\theta}_{t_{i}})$", fontsize=16)
    cos_log.grid(True, alpha=0.3)
    cos_log.set_xscale("log", base=10)
    cos_log.tick_params(axis="both", labelsize=13)

    ax_step_norms.set_xlabel(r"Training Steps $t_{i}$ [log]", fontsize=16)
    ax_step_norms.set_ylabel(r"$\| \Delta \vec{\theta}_{t_{i}} \| = \| \vec{\theta}_{t_{i+1}} - \vec{\theta}_{t_{i}}\|$", fontsize=16)
    ax_step_norms.grid(True, alpha=0.3)
    ax_step_norms.set_xscale("log", base=10)
    ax_step_norms.tick_params(axis="both", labelsize=13)

    ax_path_lengths.set_xlabel(r"Network Size $[log_{2}]$", fontsize=16)
    ax_path_lengths.set_ylabel(r"$F(T)$", fontsize=16)
    ax_path_lengths.grid(True, alpha=0.3)
    ax_path_lengths.set_xticks(node_list)
    ax_path_lengths.set_xticklabels([str(value) for value in node_list])
    ax_path_lengths.set_xscale("log", base=2)
    ax_path_lengths.tick_params(axis="both", labelsize=13)

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

    loss_log.text(
        0.1, 0.1,                     # (x, y) in Axes coordinates
        f"lr = {dataset_info.get('lr')} \nepochs = {dataset_info.get('epochs')} \nbatch size = {dataset_info.get('batch_size')}",     # multiline text
        transform=loss_log.transAxes,        # anchor relative to axes
        ha='left', va='bottom',           # align text box to corner
        fontsize=15,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.4"
        )
    )

    fig.suptitle(r"$\vec{\theta}_{t}$ Describes the Networks Whole Weight Vector" + f"\n {save_loss_frequency} SGD Update Steps Between Subsequent Data Points" + r" $t_{i+1}$ and $t_{i}$", fontsize=18)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    prefix = Path(f"figures_weight_update_similarity/{task_name}/{job_dir.name}")
    prefix.mkdir(parents=True, exist_ok=True)
    if outfile:
        file_path = prefix / outfile
    else:
        file_path = prefix / f"weight_metrics_{layer}.png"
    if file_path.suffix == "":
        file_path = file_path.with_suffix(".png")
    fig.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")


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
        "--layer",
        type=str,
        default="all_weights",
        help="Optional flag to plot the weight metrics only for the specified layer of the network. Each network layer can be adressed as: 'Dense_0' ... 'Dense_N'"
    )
    parser.add_argument(
        "--compute",
        action="store_true",
        help="Optional flag to plot the amount of compute [FLOPS] on the x-axis corresponding to the weight metric or loss at each training step."
    )

    args = parser.parse_args()
    plot_weight_update_similarity(args.log_dir, args.output, args.layer, args.compute)


if __name__ == "__main__":
    main()
