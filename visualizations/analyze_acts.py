"""Script to visualize average layer activations per epoch for different model widths."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence
import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root) + "/src")
from utils import load_yaml_as_dict

import matplotlib
matplotlib.use("Agg")

_CACHE_DIR = Path(".cache/matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


def load_training_log(log_path: Path) -> Dict[str, object]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def _extract_layer_activations(log_path: Path) -> np.ndarray:
    log_data = load_training_log(log_path)
    history = log_data.get("final_metrics", {}).get("history")
    if not history:
        history = log_data.get("history", {})
    activations = history.get("layer_activations")
    if activations is None:
        raise ValueError(f"No layer_activations history found in {log_path}")

    activations_array = np.asarray(activations, dtype=float)
    if activations_array.ndim != 2:
        raise ValueError(
            f"Expected layer_activations to be a 2D array, got shape {activations_array.shape}"
        )
    return activations_array


def _gather_activation_histories(log_dir: Path) -> Dict[str, np.ndarray]:
    log_files = sorted(collect_files_with_ending(log_dir, "training_log.json"))
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {log_dir}")

    width_to_history: Dict[str, np.ndarray] = {}
    for log_path in log_files:
        width_label = extract_after_char(str(log_path), "-", "/")
        activations = _extract_layer_activations(log_path)
        width_to_history[width_label] = activations

    return dict(sorted(width_to_history.items(), key=lambda item: tuple(_width_key(item[0]))))


def plot_layer_activations(log_dir: Path, outfile: Path | None) -> None:
    histories = _gather_activation_histories(log_dir)
    if not histories:
        raise ValueError(f"No activation histories available in {log_dir}")
    
    simluation_config_path = collect_files_with_ending(log_dir.parent, "simulation_config.yaml")[0]
    simulation_info = load_yaml_as_dict(simluation_config_path)
    dataset_info = simulation_info["training"]
    network_info = simulation_info["network"]
    save_loss_frequency = dataset_info.get("save_loss_frequency")

    epoch_counts = {width: history.shape[0] for width, history in histories.items()}
    min_epochs = min(epoch_counts.values())
    if min_epochs < 1:
        raise ValueError("Activation histories must contain at least one epoch.")

    num_layers_set = {history.shape[1] for history in histories.values()}
    if len(num_layers_set) != 1:
        raise ValueError(
            f"Inconsistent layer counts across widths: {sorted(num_layers_set)}"
        )
    num_layers = num_layers_set.pop()

    width_items = list(histories.items())
    parsed_widths: List[List[int]] = []
    for width_label, history in width_items:
        widths = [int(part) for part in re.findall(r"\d+", width_label)]
        if len(widths) < num_layers:
            raise ValueError(
                f"Width specification '{width_label}' has fewer entries than activation layers ({num_layers})."
            )
        parsed_widths.append(widths)

    hidden_count = num_layers
    if hidden_count <= 0:
        raise ValueError("Layer activation history does not include hidden layers to plot.")

    num_epochs = min(10, min_epochs)
    if min_epochs < 10:
        print(
            f"Warning: only {min_epochs} epochs available; plotting the first {num_epochs} epochs."
        )
    if min_epochs > 10:
        print(
            f"Note: {min_epochs} epochs available; plotting the first {num_epochs} epochs."
        )

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes_flat = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 1, hidden_count))
    unique_width_values = sorted(
        {widths[idx] for widths in parsed_widths for idx in range(hidden_count)}
    )

    legend_handles: List[object] = []
    legend_labels: List[str] = []
    for epoch_idx in range(len(axes_flat)):
        ax = axes_flat[epoch_idx]
        if epoch_idx >= num_epochs:
            ax.axis("off")
            continue

        for layer_idx in range(hidden_count):
            layer_widths = [widths[0] for widths in parsed_widths]
            layer_values = [history[epoch_idx, layer_idx] for _, history in width_items]
            (line,) = ax.semilogx(
                layer_widths,
                layer_values,
                base=2,
                marker="o",
                color=colors[layer_idx],
                label=f"Hidden Layer {layer_idx + 1}",
            )
            if epoch_idx == 0 and layer_idx < len(colors):
                legend_handles.append(line)
                legend_labels.append(f"Dense Layer {layer_idx}")

        ax.set_title(f"Optimization Step {(epoch_idx + 1)*save_loss_frequency}", fontsize=15)
        if epoch_idx >= 5:
            ax.set_xlabel(r"Hidden Layer Width $[log_2]$", fontsize=15)
        if epoch_idx % 5 == 0:
            ax.set_ylabel("Logit Norms", fontsize=15)
        ax.set_xticks(layer_widths)
        ax.set_xticklabels([str(value) for value in layer_widths])
    
        ax.tick_params(axis="y", which="both", labelleft=True)
        ax.grid(True, alpha=0.3)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), frameon=False, fontsize=15)

    fig.suptitle(f"Logit Norms for each Network Layer vs. Network Width in the Hidden Layers (Shown for Multiple Optimization Steps)", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    prefix = "figures_analyze_acts/"
    if outfile:
        file_path = prefix + outfile
    else:
        file_path = prefix + str(log_dir.name)


    Path(prefix).mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight")
    print(f"Saved plot to {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize layer activation averages across widths for each epoch. "
            "By default, reads logs from experiments/experiment1/logs/run_analyze_actications."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("experiments/experiment1/logs/run_analyze_actications"),
        help="Directory containing subfolders with training_log.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated plot image.",
    )

    args = parser.parse_args()
    plot_layer_activations(args.log_dir, args.output)


if __name__ == "__main__":
    main()
