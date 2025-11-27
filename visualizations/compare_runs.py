"""Utility script to plot training logs with a combined legend grouped by parametrization scheme."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root) + "/src")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils import extract_after_char, collect_files_with_ending, load_yaml_as_dict


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
        header = Line2D([], [], linestyle="", marker="", linewidth=0)
        handles.append(header)
        labels.append(legend_titles[scheme])
        header_indices.append(len(labels) - 1)
        for handle, label in sorted_linear:
            handles.append(handle)
            labels.append(label)

    return handles, labels, header_indices


def plot_run(logs: Path, outfile: Path, loss_type: str, compute_flag: bool) -> None:
    log_files = []
    for run in logs:
        files = collect_files_with_ending(run, "training_log.json")
        log_files += files
    if not log_files:
        raise FileNotFoundError(f"No training logs found in {logs}")

    scheme_groups = {
        "standard": [log for log in log_files if _scheme_from_path(log) == "standard"],
        "muP": [log for log in log_files if _scheme_from_path(log) == "muP"],
    }

    fig, (ax_loss, ax_loss_log) = plt.subplots(1, 2, figsize=(12, 5))

    color_map = {
        "standard": _prepare_color_map(len(scheme_groups["standard"]), plt.cm.autumn),
        "muP": _prepare_color_map(len(scheme_groups["muP"]), plt.cm.winter),
    }
    color_indices = {"standard": 0, "muP": 0}

    legend_entries: Dict[str, List[Tuple[Any, str]]] = {
        "standard": [],
        "muP": [],
    }

    for log_path in log_files:
        log_data = load_training_log(log_path)
        history = log_data.get("history") or log_data.get("final_metrics", {}).get("history", {})
        losses = history.get(f"{loss_type}_loss")
        if not losses:
            raise ValueError(f"No test loss history found in {log_path}")
        
        simluation_config_path = collect_files_with_ending(log_path.parent, "simulation_config.yaml")[0]
        simulation_info = load_yaml_as_dict(simluation_config_path)
        dataset_info = simulation_info["training"]
        network_info = simulation_info["network"]
        save_loss_frequency = dataset_info.get("save_loss_frequency")

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

        scheme = _scheme_from_path(log_path)
        color = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1

        label = extract_after_char(str(log_path), "-", "/")

        line_loss, = ax_loss.plot(x_axis, losses, color=color, label=label)
        ax_loss_log.plot(x_axis, losses, color=color, label=label)

        legend_entries[scheme].append((line_loss, label))

    ax_loss.set_xlabel(f"{x_label}", fontsize=20)
    ax_loss.set_ylabel(f"{loss_type.capitalize()} Loss", fontsize=20)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax_loss.xaxis.get_offset_text().set_fontsize(15)
    ax_loss.yaxis.get_offset_text().set_fontsize(15)
    ax_loss.tick_params(axis='both', labelsize=14)
    # ax_loss.set_title("Loss vs. Epoch", fontsize=20)



    ax_loss_log.set_xscale("log")
    ax_loss_log.set_yscale("log")
    ax_loss_log.set_xlabel(f"{x_label} [log]", fontsize=20)
    ax_loss_log.set_ylabel(f"{loss_type.capitalize()} Loss [log]", fontsize=20)
    ax_loss_log.grid(True, which="both", alpha=0.3)
    ax_loss_log.tick_params(axis='both', labelsize=14)
    # ax_loss_log.set_title("Loss vs. epoch [log-log]", fontsize=20)
    

    legend_titles = {
        "standard": "Standard parametrization",
        "muP": "muP parametrization",
    }

    combined_handles, combined_labels, header_indices = _build_combined_legend(
        legend_entries, legend_titles
    )

    fig.suptitle("Runs")
    fig.tight_layout(rect=[0, 0.15, 1, 0.97])

    if combined_handles:
        legend = ax_loss.legend(
            combined_handles,
            combined_labels,
            loc="upper right",
            frameon=True,
            borderaxespad=0.0,
            handlelength=1.5,
            handletextpad=0.6,
            fontsize=15
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")
        
    
    ax_loss.text(
        0.02, 0.02,                     # (x, y) in Axes coordinates
        f"lr = {dataset_info.get('lr')} \nepochs = {dataset_info.get('epochs')} \nbatch size = {dataset_info.get('batch_size')}",     # multiline text
        transform=ax_loss.transAxes,        # anchor relative to axes
        ha='left', va='bottom',           # align text box to corner
        fontsize=15,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.4"
        )
    )

    if outfile:
        outfile.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, bbox_inches="tight")
        print(f"Saved plot to {outfile}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize training logs and show parametrization sections in a single legend."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        nargs="+",
        help="Path to a directory containing one or more training_log.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated plot image.",
        default=None
    )

    parser.add_argument(
        "--loss",
        type=str,
        help="Optional flag to plot the test loss. Usage: Add --loss train when running this program",
        default="train"
    )

    parser.add_argument(
        "--compute",
        action="store_true",
        help="Optional flag to plot the test loss. Usage: Add --loss train when running this program"
    )


    args = parser.parse_args()
    plot_run(args.log_dir, args.output, args.loss, args.compute)


if __name__ == "__main__":
    main()
