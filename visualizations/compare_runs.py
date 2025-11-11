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

from utils import extract_after_char, collect_files_with_ending


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


def plot_run(logs: Path, outfile: Path | None) -> None:
    log_files = sorted(collect_files_with_ending(logs, "training_log.json"))
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
        test_losses = history.get("test_loss", [])
        if not test_losses:
            raise ValueError(f"No test loss history found in {log_path}")
        
        dataset_info = log_data.get("dataset_info")
        if isinstance(dataset_info.get("save_loss_frequency"), int):
            save_loss_frequency = dataset_info.get("save_loss_frequency") 
        else:
            save_loss_frequency = 1  # Default to 1 if not specified or invalid
        train_samples = np.array(range(1, len(test_losses) + 1)) * dataset_info.get("n_train")/save_loss_frequency
        scheme = _scheme_from_path(log_path)
        color = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1

        label = extract_after_char(str(log_path), "-", "/")

        line_loss, = ax_loss.plot(train_samples, test_losses, color=color, label=label)
        ax_loss_log.plot(train_samples, test_losses, color=color, label=label)

        legend_entries[scheme].append((line_loss, label))

    ax_loss.set_xlabel("Training Samples", fontsize=20)
    ax_loss.set_ylabel("Test Loss", fontsize=20)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax_loss.xaxis.get_offset_text().set_fontsize(15)
    ax_loss.yaxis.get_offset_text().set_fontsize(15)
    ax_loss.tick_params(axis='both', labelsize=14)
    # ax_loss.set_title("Loss vs. Epoch", fontsize=20)



    ax_loss_log.set_xscale("log")
    ax_loss_log.set_yscale("log")
    ax_loss_log.set_xlabel("Training Samples [log]", fontsize=20)
    ax_loss_log.set_ylabel("Test Loss [log]", fontsize=20)
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
        help="Path to a directory containing one or more training_log.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated plot image.",
    )

    args = parser.parse_args()
    plot_run(args.log_dir, args.output)


if __name__ == "__main__":
    main()
