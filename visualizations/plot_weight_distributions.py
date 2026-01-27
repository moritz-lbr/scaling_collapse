"""Plot histograms of layer weights from a training_log.json file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

_CACHE_DIR = Path(".cache/matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_training_log(log_path: Path) -> Dict[str, object]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_log_path(path: Path) -> Path:
    if path.is_file():
        return path

    matches = list(path.rglob("training_log.json"))
    if not matches:
        raise FileNotFoundError(f"No training_log.json found under {path}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple training_log.json files found under {path}; please specify one explicitly."
        )
    return matches[0]


def _extract_kernels(final_params: Dict[str, object]) -> Iterable[Tuple[str, np.ndarray]]:
    for layer_name, params in final_params.items():
        kernel = None
        if isinstance(params, dict):
            if "kernel" in params:
                kernel = params["kernel"]
            elif isinstance(params.get("params"), dict):
                kernel = params["params"].get("kernel")
        elif isinstance(params, list):
            kernel = params

        if kernel is None:
            continue

        arr = np.asarray(kernel, dtype=float)
        if arr.size == 0:
            continue

        # Move the output neuron axis first so each node corresponds to a downstream neuron.
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = np.moveaxis(arr, -1, 0)
            arr = arr.reshape(arr.shape[0], -1)
        yield layer_name, arr


def _hist_bins(values: np.ndarray, bin_override: int | None) -> np.ndarray:
    flat = values.ravel()
    if flat.size == 0:
        raise ValueError("Cannot build histogram bins for empty weights.")

    data_min, data_max = float(flat.min()), float(flat.max())
    if data_min == data_max:
        span = max(abs(data_min), 1.0)
        data_min -= 0.5 * span
        data_max += 0.5 * span

    bin_count = bin_override or min(80, max(20, int(np.sqrt(flat.size))))
    return np.linspace(data_min, data_max, bin_count)


def _plot_layer(
    layer_name: str,
    weights: np.ndarray,
    output_dir: Path,
    bin_override: int | None,
    max_nodes: int | None,
    run_id: str
) -> None:
    bins = _hist_bins(weights, bin_override)
    num_nodes = weights.shape[0]
    nodes_to_plot = min(num_nodes, max_nodes) if max_nodes else num_nodes
    alpha = max(0.05, min(0.4, 8.0 / max(nodes_to_plot, 1)))

    fig, ax = plt.subplots(figsize=(8, 6))
    # for node_idx in range(nodes_to_plot):
        # ax.hist(
        #     weights[node_idx],
        #     bins=bins,
        #     density=True,
        #     histtype="step",
        #     color="C0",
        #     alpha=alpha,
        #     linewidth=1.0,
        # )

    # node_average = weights.mean(axis=1)
    # input_weights = weights.mean(axis=0)
    # print(feature_average.size)
    # ax.hist(
    #     node_average,
    #     bins=bins,
    #     density=True,
    #     histtype="stepfilled",
    #     color="C1",
    #     alpha=0.9,
    #     linewidth=1.3,
    #     label="Node Average",
    # )

    combined = weights.ravel()
    ax.hist(
        combined,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color="C0",
        alpha=0.9,
        linewidth=1.3,
        label="Feature Average",
    )


    

    title_suffix = f"{nodes_to_plot} of {num_nodes} nodes" if nodes_to_plot != num_nodes else f"{num_nodes} nodes"
    ax.set_title(f"{layer_name} weight distribution ({title_suffix})")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False)

    outfile = output_dir / f"{run_id}_{layer_name.lower()}_weights.png"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_weight_distributions(
    log_path: Path,
    output_dir: Path,
    bins: int | None,
    max_nodes: int | None,
) -> None:
    resolved_path = _resolve_log_path(log_path)
    log_data = load_training_log(resolved_path)
    final_params = log_data.get("final_params")
    if not isinstance(final_params, dict):
        raise ValueError(f"No final_params dict found in {resolved_path}")

    kernels = list(_extract_kernels(final_params))
    if not kernels:
        raise ValueError(f"No kernels found in final_params from {resolved_path}")

    run = str(log_path.parent.name)
    job_id = str(log_path.parent.parent.name)
    run_id = job_id + "/" + run
    for layer_name, weights in kernels:
        _plot_layer(layer_name, weights, output_dir, bins, max_nodes, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot weight distributions per layer from a training_log.json. "
            "Each layer's kernel is split per node and overlaid as histogram steps."
        )
    )
    parser.add_argument(
        "--log-path",
        "--log-dir",
        dest="log_path",
        type=Path,
        required=True,
        help="Path to a training_log.json file or a directory containing exactly one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weight_distributions"),
        help="Directory to save the generated histogram images.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        help="Optional number of histogram bins. Defaults to a heuristic based on weight count.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Optionally limit how many nodes to overlay per layer (all by default).",
    )

    args = parser.parse_args()
    plot_weight_distributions(args.log_path, args.output_dir, args.bins, args.max_nodes)


if __name__ == "__main__":
    main()
