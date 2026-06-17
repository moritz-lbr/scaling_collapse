#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from typing import Any, Sequence

_CACHE_DIR = Path(".cache/matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR.resolve()))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numcodecs.zstd import Zstd
import numpy as np
import yaml


@dataclass(frozen=True)
class ImageLayout:
    height: int
    width: int
    channels: int
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class DenseRunInfo:
    zarr_path: Path
    run_dir: Path
    job_dir: Path
    dense0_dir: Path
    dense1_dir: Path
    dense0_meta: dict[str, Any]
    dense1_meta: dict[str, Any]
    dense0_shape: tuple[int, ...]
    dense1_shape: tuple[int, ...]
    step_to_time_index: dict[int, int]
    input_dim: int
    output_dim: int
    total_params: float
    batch_size: float
    activation0: Any
    activation1: Any
    loss_type: str


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def training_step_stride(run_dir: Path) -> int:
    training = load_yaml(run_dir / "simulation_config.yaml").get("training", {})
    value = training.get("save_loss_frequency", 1)
    if isinstance(value, str):
        if value.strip().lower() == "epoch":
            batch_size = int(training.get("batch_size", 1))
            n_train = int(training.get("training_data", {}).get("n_train", batch_size))
            return 1 if batch_size >= n_train else n_train // batch_size
        return int(float(value))
    return int(value)


def task_name(run_dir: Path) -> str | None:
    task = (
        load_yaml(run_dir / "simulation_config.yaml")
        .get("training", {})
        .get("training_data", {})
        .get("task")
    )
    return Path(task).name if task else None


def task_path(run_dir: Path) -> Path | None:
    task = (
        load_yaml(run_dir / "simulation_config.yaml")
        .get("training", {})
        .get("training_data", {})
        .get("task")
    )
    if not task:
        return None
    path = Path(str(task)).expanduser()
    if path.exists():
        return path.resolve()
    candidates = [Path.cwd() / path, Path.cwd() / "datasets" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def square_layout(unit_count: int) -> ImageLayout | None:
    side = isqrt(unit_count)
    if side * side != unit_count:
        return None
    return ImageLayout(side, side, 1, ("gray",))


def rgb_layout(unit_count: int) -> ImageLayout | None:
    if unit_count % 3 != 0:
        return None
    side = isqrt(unit_count // 3)
    if side * side * 3 != unit_count:
        return None
    return ImageLayout(side, side, 3, ("R", "G", "B"))


def infer_image_layout(run_dir: Path, unit_count: int) -> ImageLayout | None:
    path = task_path(run_dir)
    overview = load_yaml(path / "dataset_overview.yaml") if path and path.is_dir() else {}
    task_text = " ".join(
        str(value).lower()
        for value in (
            path.name if path else "",
            path.as_posix() if path else "",
            overview.get("training", {}).get("variant", ""),
        )
    )

    if unit_count == 784 and ("mnist" in task_text or "emnist" in task_text):
        return ImageLayout(28, 28, 1, ("gray",))
    if unit_count == 3072 and (
        "svhn" in task_text or "cifar" in task_text or "32x32" in task_text
    ):
        return ImageLayout(32, 32, 3, ("R", "G", "B"))

    layout = square_layout(unit_count)
    if layout is not None:
        return layout
    return rgb_layout(unit_count)


def zarr_paths(input_dir: Path) -> list[Path]:
    if input_dir.name == "weights.zarr" and input_dir.is_dir():
        return [input_dir]
    return sorted(path for path in input_dir.rglob("weights.zarr") if path.is_dir())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def scheme_from_run(run: str) -> str:
    text = str(run).lower()
    if "standard" in text:
        return "standard"
    if "mup" in text or "mu_p" in text or "mu-p" in text:
        return "muP"
    return "muP"


def label_from_run(run: str) -> str:
    return run.rsplit("-", maxsplit=1)[-1]


def width_key(label: str) -> Sequence[int]:
    try:
        return [int(part) for part in label.split("x")]
    except ValueError:
        return [int(match) for match in re.findall(r"\d+", label)] or [10**9]


def combined_legend(legend_entries: dict[str, list[tuple[Any, str]]]) -> tuple[list[Any], list[str], list[int]]:
    handles: list[Any] = []
    labels: list[str] = []
    header_indices: list[int] = []
    titles = {"standard": "Standard parametrization", "muP": "muP parametrization"}
    for scheme in ("standard", "muP"):
        entries = sorted(legend_entries[scheme], key=lambda item: tuple(width_key(item[1])))
        if not entries:
            continue
        handles.append(plt.Line2D([], [], linestyle="", marker="", linewidth=0))
        labels.append(titles[scheme])
        header_indices.append(len(labels) - 1)
        for handle, label in entries:
            handles.append(handle)
            labels.append(label)
    return handles, labels, header_indices


def keep_log_step(step: int, min_step: int) -> bool:
    if step == 0:
        return True
    if step < min_step:
        return False
    decade = 10 ** (len(str(step)) - 1)
    return step % decade == 0


def selected_times(length: int, frequency: int, min_step: int) -> list[tuple[int, int]]:
    return [
        (index, int(step))
        for index, step in enumerate(np.arange(length, dtype=np.int64) * frequency)
        if keep_log_step(int(step), min_step)
    ]


def set_nonnegative_loglike_xscale(ax: plt.Axes, arrays: Sequence[Any]) -> None:
    finite_values = []
    for values in arrays:
        array = np.asarray(values, dtype=float).ravel()
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return
    values = np.concatenate(finite_values)
    if np.all(values > 0):
        ax.set_xscale("log")
        return
    positive = values[values > 0]
    if np.all(values >= 0) and positive.size:
        ax.set_xscale("symlog", linthresh=float(np.min(positive)), linscale=0.5)


def matrix_keys(store: Path) -> list[str]:
    attrs = load_json(store / "zarr.json").get("attributes", {})
    keys = list(attrs.get("param_keys", []))
    return [
        key
        for key in keys
        if len(load_json(store / key / "zarr.json").get("shape", [])) >= 3
    ]


def layer_name(key: str) -> str:
    parts = key.split("/")
    return parts[-2] if parts[-1] == "kernel" and len(parts) > 1 else key.replace("/", "_")


def weight_vectors_by_input_unit(weights: np.ndarray) -> np.ndarray:
    output_dim = weights.shape[-1]
    matrix = weights.reshape(-1, output_dim)
    return matrix.astype(np.float32, copy=False)


def cosine_from_input_vectors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1)
    gram = matrix @ matrix.T
    denom = norms[:, None] * norms[None, :]
    cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    overlap = cosine.sum(axis=1) - np.diag(cosine)
    return cosine, norms, overlap


def cosine_over_outputs(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cosine_from_input_vectors(weight_vectors_by_input_unit(weights))


def selected_reference_units(unit_count: int, count: int) -> np.ndarray:
    if unit_count < 1 or count < 1:
        return np.empty(0, dtype=np.int64)
    indices = np.floor(np.linspace(0, unit_count, num=count, endpoint=False)).astype(np.int64)
    indices = np.clip(indices, 0, unit_count - 1)
    return np.unique(indices)


def cosine_columns(matrix: np.ndarray, reference_units: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    refs = matrix[reference_units]
    dots = matrix @ refs.T
    denom = norms[:, None] * norms[reference_units][None, :]
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)


def normalized_input_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > 0,
    )


def offdiag_frobenius(cosine: np.ndarray) -> float:
    offdiag = cosine.copy()
    np.fill_diagonal(offdiag, 0.0)
    return float(np.linalg.norm(offdiag))


def random_superposition_frobenius_baseline(input_dim: int, output_dim: int) -> float:
    if input_dim < 1 or output_dim < 1:
        return 0.0
    return float(np.sqrt(input_dim * max(input_dim - 1, 0) / output_dim))


def effective_participation_ratio(input_dim: int, frob_norms: np.ndarray) -> np.ndarray:
    d = float(input_dim)
    squared = np.asarray(frob_norms, dtype=float) ** 2
    return np.divide(d * d, d + squared, out=np.zeros_like(squared), where=(d + squared) > 0)


def baseline_subtracted_frobenius(entry: dict[str, Any]) -> np.ndarray:
    return subtract_random_superposition_baseline(
        int(entry["input_dim"]),
        int(entry["output_dim"]),
        entry["frob_norms"],
    )


def subtract_random_superposition_baseline(
    input_dim: int,
    output_dim: int,
    frob_norms: Any,
) -> np.ndarray:
    baseline = random_superposition_frobenius_baseline(
        input_dim,
        output_dim,
    )
    return np.asarray(frob_norms, dtype=float) - baseline


def training_compute(entry: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(entry["steps"], dtype=float)
        * float(entry.get("batch_size", 1.0))
        * float(entry.get("total_params", 1.0))
    )


def superposition_reduced_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    normalized = normalized_input_vectors(matrix)
    input_dim, output_dim = normalized.shape
    if output_dim <= input_dim:
        return normalized.T @ normalized, float(np.count_nonzero(np.linalg.norm(matrix, axis=1) > 0))
    return normalized @ normalized.T, float(np.count_nonzero(np.linalg.norm(matrix, axis=1) > 0))


def superposition_summary(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    reduced, diagonal_frobenius_sq = superposition_reduced_matrix(matrix)
    frob_sq = float(np.sum(np.square(reduced, dtype=np.float64)))
    offdiag_sq = max(frob_sq - diagonal_frobenius_sq, 0.0)
    eigenvalues = np.linalg.eigvalsh(reduced)
    return float(np.sqrt(offdiag_sq)), np.sort(np.maximum(eigenvalues.astype(float), 0.0))[::-1]


def read_snapshot(array_dir: Path, meta: dict[str, Any], time_index: int) -> np.ndarray:
    shape = tuple(int(v) for v in meta["shape"])
    chunk_shape = tuple(int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"])
    if chunk_shape[1:] != shape[1:]:
        raise ValueError(f"Only full-spatial chunks are supported: {array_dir}")

    chunk_index = time_index // chunk_shape[0]
    inner_index = time_index % chunk_shape[0]
    sep = meta.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    chunk_key = [str(chunk_index)] + ["0"] * (len(shape) - 1)
    chunk_path = (
        array_dir / "c" / Path(*chunk_key)
        if sep == "/"
        else array_dir / "c" / sep.join(chunk_key)
    )
    raw = chunk_path.read_bytes()
    if any(codec.get("name") == "zstd" for codec in meta.get("codecs", [])):
        raw = Zstd().decode(raw)

    dtype = np.dtype(meta["data_type"])
    endian = next(
        (
            codec.get("configuration", {}).get("endian")
            for codec in meta.get("codecs", [])
            if codec.get("name") == "bytes"
        ),
        None,
    )
    if endian == "little":
        dtype = dtype.newbyteorder("<")
    elif endian == "big":
        dtype = dtype.newbyteorder(">")

    chunk = np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)
    return np.asarray(chunk[inner_index], dtype=np.float32)


def plot_frame(
    cosine: np.ndarray,
    norms: np.ndarray,
    overlap: np.ndarray,
    output_path: Path,
    *,
    title: str,
    overlap_span: float,
    dpi: int,
) -> None:
    x = np.arange(norms.size)
    span = max(float(overlap_span), 1e-12)
    bar_norm = Normalize(vmin=-span, vmax=span)
    bar_cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(8.0, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 0.055), height_ratios=(4.0, 1.15))
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    bax = fig.add_subplot(grid[1, 0])
    bcax = fig.add_subplot(grid[1, 1])

    image = ax.imshow(cosine, origin="upper", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, cax=cax, label=r"$\cos(W_{:,i}, W_{:,j})$")
    ax.set_title(title)
    ax.set_xlabel("Output unit j")
    ax.set_ylabel("Output unit i")

    bax.bar(x, norms, width=1.0, color=bar_cmap(bar_norm(overlap)), linewidth=0)
    bax.set_xlim(-0.5, max(norms.size - 0.5, 0.5))
    bax.set_xlabel("Output unit i")
    bax.set_ylabel(r"$||W_{:,i}||_2$")
    mappable = plt.cm.ScalarMappable(norm=bar_norm, cmap=bar_cmap)
    mappable.set_array([])
    fig.colorbar(mappable, cax=bcax, label=r"$\sum_{j \ne i}\cos(W_{:,i}, W_{:,j})$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def unit_image_coordinates(unit: int, layout: ImageLayout) -> tuple[int, int, int]:
    pixel_index = int(unit) // layout.channels
    channel = int(unit) % layout.channels
    row = pixel_index // layout.width
    col = pixel_index % layout.width
    return row, col, channel


def plot_input_space_frame(
    columns: np.ndarray,
    reference_units: np.ndarray,
    layout: ImageLayout,
    output_path: Path,
    *,
    title: str,
    channel_mode: str,
    dpi: int,
) -> None:
    expected_units = layout.height * layout.width * layout.channels
    if columns.shape[0] != expected_units:
        raise ValueError(
            f"Cannot reshape {columns.shape[0]} input units as "
            f"{layout.height}x{layout.width}x{layout.channels}."
        )

    if layout.channels == 1:
        mode = "split"
    else:
        mode = channel_mode
    row_count = layout.channels if mode == "split" else 1
    col_count = len(reference_units)
    cube = columns.reshape(layout.height, layout.width, layout.channels, col_count)

    fig_width = max(8.0, 2.05 * col_count + 0.8)
    fig_height = max(2.8, 2.0 * row_count + 0.8)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        row_count,
        col_count + 1,
        width_ratios=[1.0] * col_count + [0.06],
    )

    image = None
    for ref_index, ref_unit in enumerate(reference_units):
        ref_row, ref_col, ref_channel = unit_image_coordinates(int(ref_unit), layout)
        for row_index in range(row_count):
            channel = row_index if mode == "split" else ref_channel
            ax = fig.add_subplot(grid[row_index, ref_index])
            image = ax.imshow(
                cube[:, :, channel, ref_index],
                origin="upper",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
            )
            show_reference_marker = mode == "split" or channel == ref_channel
            if show_reference_marker:
                ax.plot(
                    ref_col,
                    ref_row,
                    marker="x",
                    color="black",
                    markersize=4.0,
                    markeredgewidth=0.8,
                    linestyle="",
                )
                value = float(cube[ref_row, ref_col, channel, ref_index])
                text_x = ref_col + (1.0 if ref_col < layout.width / 2 else -1.0)
                text_y = ref_row + (1.0 if ref_row < layout.height / 2 else -1.0)
                ax.text(
                    text_x,
                    text_y,
                    f"{value:+.2f}",
                    color="black",
                    fontsize=6,
                    ha="left" if text_x > ref_col else "right",
                    va="top" if text_y > ref_row else "bottom",
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.7, "linewidth": 0},
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                channel_label = layout.channel_names[ref_channel]
                ax.set_title(
                    f"unit {int(ref_unit)}\n"
                    f"r={ref_row}, c={ref_col}, ch={channel_label}",
                    fontsize=8,
                )
            if ref_index == 0:
                ylabel = (
                    layout.channel_names[channel]
                    if mode == "split"
                    else f"ref {layout.channel_names[channel]}"
                )
                ax.set_ylabel(ylabel)

    if image is not None:
        cax = fig.add_subplot(grid[:, -1])
        fig.colorbar(image, cax=cax, label=r"$\cos(W_{u,:}, W_{\mathrm{ref},:})$")
    fig.suptitle(title, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def job_output_dir(base: Path, run_dir: Path) -> Path:
    parts = [base]
    task = task_name(run_dir)
    if task:
        parts.append(Path(task))
    parts.append(Path(run_dir.parent.name))
    return Path(*parts)


def output_dir(base: Path, run_dir: Path, layer: str) -> Path:
    return job_output_dir(base, run_dir) / run_dir.name / layer / "sup_frames"


def input_space_output_dir(base: Path, run_dir: Path, layer: str) -> Path:
    return job_output_dir(base, run_dir) / run_dir.name / layer / "input_space_frames"


def sorted_series_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda item: tuple(width_key(label_from_run(item["run"]))))


def series_colors(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[tuple[Any, str]]]]:
    scheme_groups = {
        "standard": [entry for entry in entries if scheme_from_run(entry["run"]) == "standard"],
        "muP": [entry for entry in entries if scheme_from_run(entry["run"]) == "muP"],
    }
    color_map = {
        "standard": plt.cm.autumn(np.linspace(0, 1, max(len(scheme_groups["standard"]), 1))),
        "muP": plt.cm.winter(np.linspace(0, 1, max(len(scheme_groups["muP"]), 1))),
    }
    color_indices = {"standard": 0, "muP": 0}
    colors: dict[str, Any] = {}
    legend_entries: dict[str, list[tuple[Any, str]]] = {"standard": [], "muP": []}
    for entry in sorted_series_entries(entries):
        scheme = scheme_from_run(entry["run"])
        colors[entry["run"]] = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1
    return colors, legend_entries


def apply_combined_legend(
    ax: plt.Axes,
    legend_entries: dict[str, list[tuple[Any, str]]],
) -> None:
    handles, labels, header_indices = combined_legend(legend_entries)
    if handles:
        legend = ax.legend(handles, labels, fontsize=8, frameon=False)
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")


def save_frob_norm_plots(series: list[dict[str, Any]], dpi: int) -> None:
    if not series:
        return

    layers = {entry["layer"] for entry in series}
    groups: dict[tuple[Path, str], list[dict[str, Any]]] = {}
    for entry in series:
        groups.setdefault((entry["job_dir"], entry["layer"]), []).append(entry)

    for (job_dir, layer), entries in groups.items():
        if layer == "Dense_0":
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.6), constrained_layout=True)
            ax, corrected_ax, compute_ax = axes
        else:
            fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
            corrected_ax = None
            compute_ax = None

        colors, legend_entries = series_colors(entries)
        for entry in sorted_series_entries(entries):
            scheme = scheme_from_run(entry["run"])
            label = label_from_run(entry["run"])
            color = colors[entry["run"]]
            line, = ax.plot(
                entry["steps"],
                entry["frob_norms"],
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=color,
            )
            legend_entries[scheme].append((line, label))
            if corrected_ax is not None:
                corrected = baseline_subtracted_frobenius(entry)
                corrected_ax.plot(
                    entry["steps"],
                    corrected,
                    marker="o",
                    markersize=3.0,
                    linewidth=1.2,
                    color=color,
                )
                assert compute_ax is not None
                compute_ax.plot(
                    training_compute(entry),
                    corrected,
                    marker="o",
                    markersize=3.0,
                    linewidth=1.2,
                    color=color,
                )

        all_steps = np.concatenate([entry["steps"] for entry in entries])
        set_nonnegative_loglike_xscale(ax, [all_steps])
        if corrected_ax is not None:
            set_nonnegative_loglike_xscale(corrected_ax, [all_steps])
        ax.set_xlabel("Training Step")
        ax.set_ylabel(r"$||S - \mathrm{diag}(S)||_F$")
        ax.set_title(f"Off-diagonal superposition Frobenius norm - {layer}")
        ax.grid(True, which="both", alpha=0.3)
        apply_combined_legend(ax, legend_entries)

        if corrected_ax is not None:
            corrected_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
            corrected_ax.set_xlabel("Training Step")
            corrected_ax.set_ylabel(
                r"$||S - \mathrm{diag}(S)||_F - \sqrt{d(d-1)/N}$"
            )
            corrected_ax.set_title("Baseline-subtracted off-diagonal Frobenius norm")
            corrected_ax.grid(True, which="both", alpha=0.3)
            assert compute_ax is not None
            all_compute = np.concatenate([training_compute(entry) for entry in entries])
            set_nonnegative_loglike_xscale(compute_ax, [all_compute])
            compute_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
            compute_ax.set_xlabel(r"Training Compute $c_i$")
            compute_ax.set_ylabel(
                r"$||S - \mathrm{diag}(S)||_F - \sqrt{d(d-1)/N}$"
            )
            compute_ax.set_title("Baseline-subtracted off-diagonal Frobenius norm vs compute")
            compute_ax.grid(True, which="both", alpha=0.3)
        else:
            ax.set_xlabel("Training Step")

        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / ("frob_norms.png" if len(layers) == 1 else f"frob_norms_{layer}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Saved Frobenius norm plot to {path}")


def save_effective_participation_ratio_plots(series: list[dict[str, Any]], dpi: int) -> None:
    dense0_series = [entry for entry in series if entry["layer"] == "Dense_0"]
    if not dense0_series:
        return

    groups: dict[Path, list[dict[str, Any]]] = {}
    for entry in dense0_series:
        groups.setdefault(entry["job_dir"], []).append(entry)

    for job_dir, entries in groups.items():
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True, constrained_layout=True)
        ax, corrected_ax = axes
        colors, legend_entries = series_colors(entries)
        for entry in sorted_series_entries(entries):
            scheme = scheme_from_run(entry["run"])
            label = label_from_run(entry["run"])
            values = effective_participation_ratio(
                int(entry["input_dim"]),
                entry["frob_norms"],
            )
            corrected_values = effective_participation_ratio(
                int(entry["input_dim"]),
                baseline_subtracted_frobenius(entry),
            )
            line, = ax.plot(
                entry["steps"],
                values,
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=colors[entry["run"]],
            )
            legend_entries[scheme].append((line, label))
            corrected_ax.plot(
                entry["steps"],
                corrected_values,
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=colors[entry["run"]],
            )

        all_steps = np.concatenate([entry["steps"] for entry in entries])
        set_nonnegative_loglike_xscale(ax, [all_steps])
        set_nonnegative_loglike_xscale(corrected_ax, [all_steps])
        ax.set_ylabel(r"$d^2 / (d + ||S - \mathrm{diag}(S)||_F^2)$")
        ax.set_title("Effective participation ratio - Dense_0")
        ax.grid(True, which="both", alpha=0.3)
        apply_combined_legend(ax, legend_entries)
        corrected_ax.set_xlabel("Training Step")
        corrected_ax.set_ylabel(
            r"$d^2 / (d + (||S - \mathrm{diag}(S)||_F - \sqrt{d(d-1)/N})^2)$"
        )
        corrected_ax.set_title("Effective participation ratio from baseline-subtracted Frobenius norm")
        corrected_ax.grid(True, which="both", alpha=0.3)

        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / "effective_participation_ratio_Dense_0.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Saved effective participation ratio plot to {path}")


def linear_fit(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x_values, np.ones_like(x_values)])
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    return float(slope), float(intercept)


def save_baseline_subtracted_frob_scaling_fits(series: list[dict[str, Any]], dpi: int) -> None:
    dense0_series = [entry for entry in series if entry["layer"] == "Dense_0"]
    if not dense0_series:
        return

    groups: dict[Path, list[dict[str, Any]]] = {}
    for entry in dense0_series:
        groups.setdefault(entry["job_dir"], []).append(entry)

    for job_dir, entries in groups.items():
        steps = sorted({int(step) for entry in entries for step in entry["steps"]})
        if not steps:
            continue

        corrected_by_run = {
            entry["run"]: {
                int(step): float(value)
                for step, value in zip(entry["steps"], baseline_subtracted_frobenius(entry))
            }
            for entry in entries
        }
        fit_dir = job_dir / "baseline_subtracted_frob_logN_fits_Dense_0"
        fit_dir.mkdir(parents=True, exist_ok=True)
        fit_records: list[dict[str, Any]] = []
        colors, _ = series_colors(entries)

        for step in steps:
            widths = []
            values = []
            labels = []
            skipped = 0
            for entry in sorted_series_entries(entries):
                value = corrected_by_run[entry["run"]].get(step)
                if value is None:
                    continue
                if value <= 0:
                    skipped += 1
                    continue
                widths.append(float(entry["output_dim"]))
                values.append(float(value))
                labels.append(label_from_run(entry["run"]))

            fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
            if widths:
                log_widths = np.log(np.asarray(widths, dtype=float))
                log_values = np.log(np.asarray(values, dtype=float))
                for log_width, log_value, label, entry in zip(
                    log_widths,
                    log_values,
                    labels,
                    sorted_series_entries([entry for entry in entries if corrected_by_run[entry["run"]].get(step, -1.0) > 0]),
                ):
                    ax.scatter(
                        log_width,
                        log_value,
                        color=colors[entry["run"]],
                        s=30.0,
                        label=label,
                    )

                if len(log_widths) >= 2:
                    slope, intercept = linear_fit(log_widths, log_values)
                    x_fit = np.linspace(float(log_widths.min()), float(log_widths.max()), 200)
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, color="black", linewidth=1.2)
                    fit_records.append(
                        {
                            "step": int(step),
                            "exponent": slope,
                            "intercept": intercept,
                            "points": int(len(log_widths)),
                            "skipped_nonpositive": int(skipped),
                        }
                    )
                    text = (
                        rf"$\log y = {slope:.4f}\log N + {intercept:.4f}$"
                        "\n"
                        rf"exponent $= {slope:.4f}$"
                    )
                else:
                    slope = float("nan")
                    intercept = float("nan")
                    fit_records.append(
                        {
                            "step": int(step),
                            "exponent": None,
                            "intercept": None,
                            "points": int(len(log_widths)),
                            "skipped_nonpositive": int(skipped),
                        }
                    )
                    text = "Insufficient positive points for fit"
                if skipped:
                    text += f"\nskipped nonpositive: {skipped}"
                ax.text(
                    0.04,
                    0.96,
                    text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
                )
            else:
                fit_records.append(
                    {
                        "step": int(step),
                        "exponent": None,
                        "intercept": None,
                        "points": 0,
                        "skipped_nonpositive": int(skipped),
                    }
                )
                ax.text(
                    0.5,
                    0.5,
                    "No positive baseline-subtracted values",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

            ax.set_xlabel(r"$\log N$")
            ax.set_ylabel(
                r"$\log(||S - \mathrm{diag}(S)||_F - \sqrt{d(d-1)/N})$"
            )
            ax.set_title(f"Baseline-subtracted Frobenius scaling - Dense_0 - step {step}")
            ax.grid(True, alpha=0.3)
            handles, labels_for_legend = ax.get_legend_handles_labels()
            if handles:
                unique: dict[str, Any] = {}
                for handle, label in zip(handles, labels_for_legend):
                    unique.setdefault(label, handle)
                ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False)
            fig.savefig(fit_dir / f"baseline_subtracted_frob_logN_fit_step_{step}.png", dpi=dpi)
            plt.close(fig)

        with (fit_dir / "fit_exponents.json").open("w", encoding="utf-8") as file:
            json.dump(fit_records, file, indent=2)
        print(f"Saved {len(steps)} baseline-subtracted Frobenius scaling fits to {fit_dir}")


def save_eigenvalue_spectrum_plots(series: list[dict[str, Any]], dpi: int) -> None:
    groups: dict[tuple[Path, str], list[dict[str, Any]]] = {}
    for entry in series:
        if entry.get("eigenvalues_by_step"):
            groups.setdefault((entry["job_dir"], entry["layer"]), []).append(entry)

    for (job_dir, layer), entries in groups.items():
        steps = sorted(
            {
                int(step)
                for entry in entries
                for step, _ in entry["eigenvalues_by_step"]
            }
        )
        if not steps:
            continue

        spectra_by_run = {
            entry["run"]: {int(step): values for step, values in entry["eigenvalues_by_step"]}
            for entry in entries
        }
        spectrum_dir = job_dir / f"eigenvalue_spectra_{layer}"
        spectrum_dir.mkdir(parents=True, exist_ok=True)

        for step in steps:
            fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
            colors, legend_entries = series_colors(entries)
            for entry in sorted_series_entries(entries):
                eigenvalues = spectra_by_run[entry["run"]].get(step)
                if eigenvalues is None:
                    continue
                scheme = scheme_from_run(entry["run"])
                label = label_from_run(entry["run"])
                ranks = np.arange(1, eigenvalues.size + 1)
                line, = ax.plot(
                    ranks,
                    eigenvalues,
                    linewidth=1.2,
                    color=colors[entry["run"]],
                )
                legend_entries[scheme].append((line, label))

            ax.set_xlabel("Eigenvalue rank")
            ax.set_ylabel(r"Eigenvalue of $S$ (zero tail omitted)")
            ax.set_title(f"Superposition nonzero eigenvalue spectrum - {layer} - step {step}")
            ax.grid(True, which="both", alpha=0.3)
            apply_combined_legend(ax, legend_entries)
            path = spectrum_dir / f"eigenvalues_step_{step}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
        print(f"Saved {len(steps)} eigenvalue spectrum plots to {spectrum_dir}")


def import_utils_module() -> Any:
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import utils  # type: ignore[import-not-found]

    return utils


def split_counts(total: int, test_split: float) -> tuple[int, int]:
    if total <= 1:
        return total, 0
    n_test = int(round(total * float(test_split)))
    n_test = max(1, min(total - 1, n_test))
    return total - n_test, n_test


def generic_random_eval_split(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(inputs, dtype=np.float32).reshape(inputs.shape[0], -1)
    outputs = np.asarray(outputs, dtype=np.float32).reshape(outputs.shape[0], -1)
    _, n_test = split_counts(int(inputs.shape[0]), test_split)
    permutation = np.random.default_rng(seed=seed).permutation(inputs.shape[0])
    test_idx = permutation[:n_test]
    if test_idx.size == 0:
        test_idx = permutation[-1:]
    return inputs[test_idx].astype(np.float32), outputs[test_idx].astype(np.float32)


def load_reconstruction_eval_data(
    run_dir: Path,
    cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    utils = import_utils_module()
    config = load_yaml(run_dir / "simulation_config.yaml")
    training = config.get("training", {})
    training_data = training.get("training_data", {})
    task = training_data.get("task")
    if task is None:
        raise ValueError(f"Missing training_data.task in {run_dir / 'simulation_config.yaml'}")

    dataset_path = utils.resolve_dataset_path(task)
    test_split = float(training_data.get("test_split", training.get("test_split", 0.2)))
    batch_seed = training.get("batch_seed")
    seed_label = str(batch_seed) if batch_seed is not None else "predefined_or_seed0"
    cache_key = (str(dataset_path), seed_label, test_split)
    if cache_key in cache:
        return cache[cache_key]

    if dataset_path.is_dir() and (dataset_path / "train_32x32.mat").is_file():
        _, _, x_test, y_test = utils.load_svhn(dataset_path)
    elif dataset_path.is_dir() and (dataset_path / "data_batch_1").is_file():
        _, _, x_test, y_test = utils.load_all_cifar10_data(dataset_path)
    elif dataset_path.is_dir() and (dataset_path / "train-images-idx3-ubyte").is_file():
        _, _, x_test, y_test = utils.load_mnist(dataset_path)
    elif dataset_path.is_dir() and (
        (dataset_path / "emnist-byclass-test-images-idx3-ubyte").is_file()
        or (dataset_path / "byclass" / "emnist-byclass-test-images-idx3-ubyte").is_file()
    ):
        _, _, x_test, y_test = utils.load_emnist_byclass(dataset_path)
    elif (dataset_path if dataset_path.is_file() else dataset_path / "data.npz").is_file():
        data_path = dataset_path if dataset_path.is_file() else dataset_path / "data.npz"
        with np.load(data_path) as data:
            files = set(data.files)
        if {"inputs", "outputs", "train_idx", "test_idx"} <= files:
            _, _, x_test, y_test = utils.load_classification_dataset(dataset_path, one_hot=True)
        else:
            inputs, outputs = utils.load_teacher_dataset(dataset_path)
            seed = int(batch_seed) if batch_seed is not None and str(batch_seed) != "random" else 0
            x_test, y_test = generic_random_eval_split(
                inputs,
                outputs,
                test_split=test_split,
                seed=seed,
            )
    else:
        inputs, outputs = utils.load_teacher_dataset(dataset_path)
        seed = int(batch_seed) if batch_seed is not None and str(batch_seed) != "random" else 0
        x_test, y_test = generic_random_eval_split(
            inputs,
            outputs,
            test_split=test_split,
            seed=seed,
        )

    result = (np.asarray(x_test, dtype=np.float32), np.asarray(y_test, dtype=np.float32))
    cache[cache_key] = result
    return result


def numpy_activation(name: str) -> Any:
    if name == "identity":
        return lambda x: x
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh
    if name == "gelu":
        return lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if name == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"Unsupported activation {name!r} for spectral reconstruction")


def classification_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    labels_array = np.asarray(labels)
    if labels_array.ndim == 2 and labels_array.shape[-1] == logits.shape[-1]:
        labels_onehot = labels_array.astype(np.float32)
    elif labels_array.ndim == 2 and labels_array.shape[-1] == 1:
        labels_onehot = np.eye(logits.shape[-1], dtype=np.float32)[labels_array[:, 0].astype(np.int64)]
    else:
        labels_onehot = np.eye(logits.shape[-1], dtype=np.float32)[labels_array.reshape(-1).astype(np.int64)]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return float(-np.mean(np.sum(labels_onehot * log_probs, axis=1)))


def mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def evaluate_dense_two_layer_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense0_kernel: np.ndarray,
    dense1_kernel: np.ndarray,
    *,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> float:
    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:stop]
        logits = activation1(activation0(xb @ dense0_kernel) @ dense1_kernel)
        yb = y_test[start:stop]
        if loss_type == "cross_entropy":
            loss = classification_cross_entropy(logits, yb)
        elif loss_type == "mse":
            loss = mean_squared_error(logits, yb)
        else:
            raise ValueError(f"Unsupported loss_type {loss_type!r}")
        losses.append(loss)
        weights.append(stop - start)
    return float(np.average(losses, weights=weights))


def top_svd(matrix: np.ndarray, max_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_modes = max(1, min(int(max_modes), min(matrix.shape)))
    min_dim = min(matrix.shape)
    try:
        if max_modes < min_dim and max_modes <= max(1, min_dim // 2):
            from scipy.sparse.linalg import svds

            u, s, vt = svds(matrix, k=max_modes, which="LM")
            order = np.argsort(s)[::-1]
            return (
                np.asarray(u[:, order], dtype=np.float32),
                np.asarray(s[order], dtype=np.float32),
                np.asarray(vt[order, :], dtype=np.float32),
            )
    except Exception as exc:
        print(f"Falling back to full SVD for spectral reconstruction: {exc}")

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (
        np.asarray(u[:, :max_modes], dtype=np.float32),
        np.asarray(s[:max_modes], dtype=np.float32),
        np.asarray(vt[:max_modes, :], dtype=np.float32),
    )


def spectral_basis_for_dense0(
    dense0_kernel: np.ndarray,
    max_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    row_norms = np.linalg.norm(dense0_kernel, axis=1).astype(np.float32)
    normalized = normalized_input_vectors(dense0_kernel)
    nonzero_rows = int(np.count_nonzero(row_norms > 0))
    u, singular_values, vt = top_svd(normalized, max_modes)
    return row_norms, u, singular_values, vt, nonzero_rows


def spectral_reconstructed_superposition_frobenius(
    u: np.ndarray,
    singular_values: np.ndarray,
    modes: int,
) -> float:
    k_eff = min(int(modes), singular_values.size)
    if k_eff < 1:
        return 0.0
    left = u[:, :k_eff] * singular_values[:k_eff][None, :]
    row_norm_sq = np.sum(np.square(left, dtype=np.float64), axis=1)
    row_norm = np.sqrt(np.maximum(row_norm_sq, 0.0))
    normalized_left = np.divide(
        left,
        row_norm[:, None],
        out=np.zeros_like(left, dtype=np.float32),
        where=row_norm[:, None] > 0,
    )
    cosine = normalized_left @ normalized_left.T
    diagonal_frobenius_sq = float(np.count_nonzero(row_norm > 0))
    frob_sq = float(np.sum(np.square(cosine, dtype=np.float64)))
    return float(np.sqrt(max(frob_sq - diagonal_frobenius_sq, 0.0)))


def evaluate_reconstructed_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense1_kernel: np.ndarray,
    *,
    row_norms: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    modes: int,
    renormalize: bool,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    k_eff = min(int(modes), singular_values.size)
    u_k = u[:, :k_eff]
    s_k = singular_values[:k_eff]
    vt_k = vt[:k_eff, :]
    retained_per_feature = np.sum((u_k * s_k[None, :]) ** 2, axis=1)
    retained_total = float(np.sum(s_k.astype(np.float64) ** 2))
    total_norm_sq = float(np.count_nonzero(row_norms > 0))
    err_q = float(np.sqrt(max(total_norm_sq - retained_total, 0.0) / max(total_norm_sq, 1.0)))

    if renormalize:
        reconstructed_norms = np.sqrt(np.maximum(retained_per_feature, 0.0)).astype(np.float32)
        scale = np.divide(
            row_norms,
            reconstructed_norms,
            out=np.zeros_like(row_norms, dtype=np.float32),
            where=reconstructed_norms > 0,
        )
    else:
        scale = row_norms

    left = u_k * s_k[None, :]
    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:stop]
        hidden_pre = ((xb * scale[None, :]) @ left) @ vt_k
        logits = activation1(activation0(hidden_pre) @ dense1_kernel)
        yb = y_test[start:stop]
        if loss_type == "cross_entropy":
            loss = classification_cross_entropy(logits, yb)
        elif loss_type == "mse":
            loss = mean_squared_error(logits, yb)
        else:
            raise ValueError(f"Unsupported loss_type {loss_type!r}")
        losses.append(loss)
        weights.append(stop - start)

    diagnostics = {
        "k_effective": float(k_eff),
        "retained_mean": float(np.mean(retained_per_feature)),
        "retained_min": float(np.min(retained_per_feature)),
        "retained_max": float(np.max(retained_per_feature)),
        "err_Q": err_q,
    }
    return float(np.average(losses, weights=weights)), diagnostics


def spectral_reconstructed_dense0_kernel(
    row_norms: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    modes: int,
) -> tuple[np.ndarray, int]:
    k_eff = min(int(modes), singular_values.size)
    left = u[:, :k_eff] * singular_values[:k_eff][None, :]
    reconstructed = left @ vt[:k_eff, :]
    reconstructed = reconstructed * row_norms[:, None]
    return np.asarray(reconstructed, dtype=np.float32), k_eff


def save_spectral_reconstruction_input_space_frames(
    *,
    run_dir: Path,
    base_output: Path,
    step: int,
    mode_counts: Sequence[int],
    row_norms: np.ndarray,
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    dpi: int,
) -> None:
    unit_count = row_norms.size
    layout = infer_image_layout(run_dir, unit_count)
    if layout is None:
        print(
            f"Skipping spectral reconstruction input-space frames for {run_dir}: "
            f"could not infer image layout for {unit_count} units"
        )
        return

    reference_units = selected_reference_units(unit_count, 5)
    out_dir = spectral_reconstruction_output_dir(base_output, run_dir) / "input_space_frames" / run_dir.name
    for modes in mode_counts:
        reconstructed, k_eff = spectral_reconstructed_dense0_kernel(
            row_norms,
            u,
            singular_values,
            vt,
            int(modes),
        )
        columns = cosine_columns(reconstructed, reference_units)
        plot_input_space_frame(
            columns,
            reference_units,
            layout,
            out_dir / f"input_space_spectral_k_{int(modes):03d}_step_{int(step)}.png",
            title=(
                f"{run_dir.parent.name}/{run_dir.name}/Dense_0 spectral reconstruction "
                f"k={int(modes)} (effective {k_eff}) - Training Step: {int(step)}"
            ),
            channel_mode="split",
            dpi=dpi,
        )
    print(f"Saved {len(mode_counts)} spectral reconstruction input-space frames to {out_dir}")


def collective_modes_output_dir(base: Path, run_dir: Path) -> Path:
    return job_output_dir(base, run_dir) / "collective_modes_Dense_0" / run_dir.name


def orient_modes_for_display(modes: np.ndarray) -> np.ndarray:
    oriented = np.asarray(modes, dtype=np.float32).copy()
    for index in range(oriented.shape[1]):
        column = oriented[:, index]
        if column.size == 0:
            continue
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            oriented[:, index] *= -1.0
    return oriented


def mode_cube(modes: np.ndarray, layout: ImageLayout, mode_count: int) -> np.ndarray:
    expected = layout.height * layout.width * layout.channels
    if modes.shape[0] != expected:
        raise ValueError(
            f"Cannot reshape {modes.shape[0]} mode entries as "
            f"{layout.height}x{layout.width}x{layout.channels}."
        )
    return modes[:, :mode_count].reshape(layout.height, layout.width, layout.channels, mode_count)


def pixel_energy_from_vector(values: np.ndarray, layout: ImageLayout) -> np.ndarray:
    cube = np.asarray(values, dtype=np.float32).reshape(layout.height, layout.width, layout.channels)
    return np.sum(np.square(cube, dtype=np.float64), axis=2)


def pixel_mass_density(values: np.ndarray, layout: ImageLayout) -> np.ndarray:
    cube = np.asarray(values, dtype=np.float32).reshape(layout.height, layout.width, layout.channels)
    pixel_mass = np.sum(cube, axis=2)
    total = float(np.sum(pixel_mass))
    if total > 0:
        pixel_mass = pixel_mass / total
    return pixel_mass


def image_for_display(flat_image: np.ndarray, layout: ImageLayout) -> np.ndarray:
    image = np.asarray(flat_image, dtype=np.float32).reshape(
        layout.height,
        layout.width,
        layout.channels,
    )
    if layout.channels == 1:
        image = image[:, :, 0]
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        return np.zeros_like(image)
    return np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)


def label_text(label: np.ndarray) -> str:
    array = np.asarray(label)
    if array.ndim > 0 and array.size > 1:
        return str(int(np.argmax(array)))
    if array.size:
        value = float(array.reshape(-1)[0])
        return str(int(value)) if value.is_integer() else f"{value:.3g}"
    return "n/a"


def save_collective_mode_signed_maps(
    *,
    run_dir: Path,
    out_dir: Path,
    step: int,
    layout: ImageLayout,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    top_modes: int,
    dpi: int,
) -> None:
    mode_count = min(int(top_modes), modes.shape[1])
    if mode_count < 1:
        return
    cube = mode_cube(modes, layout, mode_count)
    signed_span = float(np.max(np.abs(cube))) if cube.size else 1.0
    signed_span = max(signed_span, 1e-12)

    col_count = layout.channels
    fig, axes = plt.subplots(
        mode_count,
        col_count,
        figsize=(2.55 * col_count, 2.05 * mode_count + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    signed_image = None
    for mode_index in range(mode_count):
        for channel in range(layout.channels):
            ax = axes[mode_index, channel]
            signed_image = ax.imshow(
                cube[:, :, channel, mode_index],
                origin="upper",
                cmap="coolwarm",
                vmin=-signed_span,
                vmax=signed_span,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if mode_index == 0:
                ax.set_title(layout.channel_names[channel])
            if channel == 0:
                ax.set_ylabel(
                    f"mode {mode_index + 1}\n"
                    rf"$\lambda$={float(eigenvalues[mode_index]):.3g}",
                    fontsize=8,
                )

    if signed_image is not None:
        fig.colorbar(signed_image, ax=axes.ravel().tolist(), shrink=0.72, label="mode entry")
    fig.suptitle(
        f"{run_dir.parent.name}/{run_dir.name}/Dense_0 signed input-space collective modes - step {step}",
        fontsize=10,
    )
    plot_dir = out_dir / "signed_channel_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"signed_mode_maps_step_{int(step)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_collective_mode_energy_maps(
    *,
    run_dir: Path,
    out_dir: Path,
    step: int,
    layout: ImageLayout,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    top_modes: int,
    dpi: int,
) -> None:
    mode_count = min(int(top_modes), modes.shape[1])
    if mode_count < 1:
        return
    energy_maps = [pixel_energy_from_vector(modes[:, index], layout) for index in range(mode_count)]
    energy_max = max((float(np.max(values)) for values in energy_maps if values.size), default=1.0)
    energy_max = max(energy_max, 1e-12)

    ncols = min(3, mode_count)
    nrows = int(np.ceil(mode_count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 2.75 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    image = None
    for mode_index in range(mode_count):
        ax = axes_flat[mode_index]
        image = ax.imshow(
            energy_maps[mode_index],
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=energy_max,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"mode {mode_index + 1}, "
            rf"$\lambda$={float(eigenvalues[mode_index]):.3g}",
            fontsize=8,
        )
    for ax in axes_flat[mode_count:]:
        ax.set_visible(False)

    if image is not None:
        fig.colorbar(
            image,
            ax=[ax for ax in axes_flat[:mode_count]],
            shrink=0.78,
            label="channel-summed squared mode entry",
        )
    fig.suptitle(
        f"{run_dir.parent.name}/{run_dir.name}/Dense_0 per-mode input-space energy - step {step}",
        fontsize=10,
    )
    plot_dir = out_dir / "mode_energy_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"mode_energy_maps_step_{int(step)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_collective_mode_leverage_maps(
    *,
    run_dir: Path,
    out_dir: Path,
    step: int,
    layout: ImageLayout,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    leverage_ks: Sequence[int],
    dpi: int,
) -> None:
    k_values = [min(int(k), modes.shape[1]) for k in leverage_ks if int(k) > 0]
    k_values = sorted(set(k for k in k_values if k > 0))
    if not k_values:
        return

    unweighted_maps = []
    weighted_maps = []
    for k in k_values:
        subspace_mass = np.sum(np.square(modes[:, :k], dtype=np.float64), axis=1)
        weighted_mass = np.sum(
            np.square(modes[:, :k], dtype=np.float64) * eigenvalues[:k][None, :],
            axis=1,
        )
        unweighted_maps.append(pixel_mass_density(subspace_mass, layout))
        weighted_maps.append(pixel_mass_density(weighted_mass, layout))

    row_labels = ("top-k leverage density", "eigenvalue-weighted density")
    fig, axes = plt.subplots(
        2,
        len(k_values),
        figsize=(3.0 * len(k_values), 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    vmax_unweighted = max(float(np.max(values)) for values in unweighted_maps)
    vmax_weighted = max(float(np.max(values)) for values in weighted_maps)
    images = []
    for col, k in enumerate(k_values):
        image = axes[0, col].imshow(
            unweighted_maps[col],
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=max(vmax_unweighted, 1e-12),
            interpolation="nearest",
        )
        images.append(image)
        axes[0, col].set_title(f"k={k}")
        axes[1, col].imshow(
            weighted_maps[col],
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=max(vmax_weighted, 1e-12),
            interpolation="nearest",
        )
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[0, col].set_ylabel(row_labels[0] if col == 0 else "")
        axes[1, col].set_ylabel(row_labels[1] if col == 0 else "")

    fig.colorbar(images[0], ax=axes[0, :], shrink=0.8, label="fraction of top-k subspace mass")
    weighted_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=max(vmax_weighted, 1e-12)),
        cmap="magma",
    )
    weighted_mappable.set_array([])
    fig.colorbar(
        weighted_mappable,
        ax=axes[1, :],
        shrink=0.8,
        label="fraction of eigenvalue-weighted mass",
    )
    fig.suptitle(
        f"{run_dir.parent.name}/{run_dir.name}/Dense_0 input-space subspace leverage - step {step}",
        fontsize=10,
    )
    plot_dir = out_dir / "subspace_leverage_maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"subspace_leverage_step_{int(step)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_collective_mode_response_examples(
    *,
    run_dir: Path,
    out_dir: Path,
    step: int,
    layout: ImageLayout,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    top_modes: int,
    example_count: int,
    example_pool: int,
    dpi: int,
) -> None:
    mode_count = min(int(top_modes), modes.shape[1])
    if mode_count < 1 or example_count < 1 or x_eval.size == 0:
        return
    x_flat = np.asarray(x_eval, dtype=np.float32).reshape(x_eval.shape[0], -1)
    y_array = np.asarray(y_eval)
    if example_pool > 0 and x_flat.shape[0] > example_pool:
        indices = np.linspace(0, x_flat.shape[0] - 1, int(example_pool)).astype(np.int64)
        x_flat = x_flat[indices]
        y_array = y_array[indices]

    modes_for_scores = modes[:, :mode_count]
    scores = x_flat @ modes_for_scores
    per_side = min(int(example_count), max(1, x_flat.shape[0] // 2))
    col_count = 1 + 2 * per_side
    fig, axes = plt.subplots(
        mode_count,
        col_count,
        figsize=(2.05 * col_count, 2.05 * mode_count + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    for mode_index in range(mode_count):
        mode_scores = np.asarray(scores[:, mode_index], dtype=float)
        order = np.argsort(mode_scores)
        negative_indices = order[:per_side]
        positive_indices = order[-per_side:][::-1]

        ax = axes[mode_index, 0]
        energy = pixel_energy_from_vector(modes[:, mode_index], layout)
        ax.imshow(energy, origin="upper", cmap="magma", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(
            f"mode {mode_index + 1}\n"
            rf"$\lambda$={float(eigenvalues[mode_index]):.3g}",
            fontsize=8,
        )
        if mode_index == 0:
            ax.set_title("mode energy")

        for offset, sample_index in enumerate(negative_indices):
            ax = axes[mode_index, 1 + offset]
            image = image_for_display(x_flat[int(sample_index)], layout)
            ax.imshow(image, cmap="gray" if layout.channels == 1 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"neg {offset + 1}\n"
                f"s={mode_scores[int(sample_index)]:+.2g}, y={label_text(y_array[int(sample_index)])}",
                fontsize=7,
            )

        for offset, sample_index in enumerate(positive_indices):
            ax = axes[mode_index, 1 + per_side + offset]
            image = image_for_display(x_flat[int(sample_index)], layout)
            ax.imshow(image, cmap="gray" if layout.channels == 1 else None, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"pos {offset + 1}\n"
                f"s={mode_scores[int(sample_index)]:+.2g}, y={label_text(y_array[int(sample_index)])}",
                fontsize=7,
            )

    fig.suptitle(
        f"{run_dir.parent.name}/{run_dir.name}/Dense_0 test images ranked by mode response - step {step}",
        fontsize=10,
    )
    plot_dir = out_dir / "mode_response_examples"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"mode_response_examples_step_{int(step)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_collective_mode_visualization_store(
    zarr_path: Path,
    base_output: Path,
    min_step: int,
    top_modes: int,
    leverage_ks: Sequence[int],
    example_count: int,
    example_pool: int,
    dpi: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> int:
    run_dir = zarr_path.parent
    dense0_dir = zarr_path / "Dense_0" / "kernel"
    if not dense0_dir.is_dir():
        print(f"Skipping collective mode visualization for {zarr_path}: missing Dense_0 kernel")
        return 0

    dense0_meta = load_json(dense0_dir / "zarr.json")
    dense0_shape = tuple(int(v) for v in dense0_meta["shape"])
    if len(dense0_shape) != 3:
        print(f"Skipping collective mode visualization for {zarr_path}: unexpected Dense_0 shape {dense0_shape}")
        return 0
    input_dim = int(dense0_shape[1])
    layout = infer_image_layout(run_dir, input_dim)
    if layout is None:
        print(
            f"Skipping collective mode visualization for {zarr_path}: "
            f"could not infer an image layout for {input_dim} input units"
        )
        return 0

    times = selected_times(dense0_shape[0], training_step_stride(run_dir), min_step)
    if not times:
        print(f"Skipping collective mode visualization for {zarr_path}: no selected times")
        return 0

    requested_ks = tuple(sorted({int(k) for k in leverage_ks if int(k) > 0}))
    max_modes = max(
        [int(top_modes), int(example_count > 0) * int(top_modes), *requested_ks],
        default=int(top_modes),
    )
    max_modes = max(1, max_modes)

    x_eval: np.ndarray | None = None
    y_eval: np.ndarray | None = None
    if example_count > 0:
        x_eval, y_eval = load_reconstruction_eval_data(run_dir, data_cache)

    out_dir = collective_modes_output_dir(base_output, run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    metadata = {
        "run": run_dir.name,
        "input_dim": input_dim,
        "output_dim": int(dense0_shape[2]),
        "top_modes": int(top_modes),
        "leverage_k_values": list(requested_ks),
        "example_count_per_sign": int(example_count),
        "steps": [],
    }
    for time_index, step in times:
        dense0_kernel = read_snapshot(dense0_dir, dense0_meta, time_index)
        _, u, singular_values, _, _ = spectral_basis_for_dense0(dense0_kernel, max_modes)
        eigenvalues = np.square(singular_values.astype(np.float64))
        modes = orient_modes_for_display(u)
        save_collective_mode_signed_maps(
            run_dir=run_dir,
            out_dir=out_dir,
            step=int(step),
            layout=layout,
            modes=modes,
            eigenvalues=eigenvalues,
            top_modes=top_modes,
            dpi=dpi,
        )
        save_collective_mode_energy_maps(
            run_dir=run_dir,
            out_dir=out_dir,
            step=int(step),
            layout=layout,
            modes=modes,
            eigenvalues=eigenvalues,
            top_modes=top_modes,
            dpi=dpi,
        )
        save_collective_mode_leverage_maps(
            run_dir=run_dir,
            out_dir=out_dir,
            step=int(step),
            layout=layout,
            modes=modes,
            eigenvalues=eigenvalues,
            leverage_ks=requested_ks,
            dpi=dpi,
        )
        if x_eval is not None and y_eval is not None:
            save_collective_mode_response_examples(
                run_dir=run_dir,
                out_dir=out_dir,
                step=int(step),
                layout=layout,
                modes=modes,
                eigenvalues=eigenvalues,
                x_eval=x_eval,
                y_eval=y_eval,
                top_modes=top_modes,
                example_count=example_count,
                example_pool=example_pool,
                dpi=dpi,
            )
        metadata["steps"].append(
            {
                "step": int(step),
                "top_eigenvalues": [float(value) for value in eigenvalues[: int(top_modes)]],
            }
        )
        saved += 3 + int(x_eval is not None and y_eval is not None and example_count > 0)

    with (out_dir / "collective_mode_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved {saved} collective mode visualization files to {out_dir}")
    return saved


def spectral_reconstruction_output_dir(base: Path, run_dir: Path) -> Path:
    return job_output_dir(base, run_dir) / "spectral_reconstruction_Dense_0"


def process_spectral_reconstruction_store(
    zarr_path: Path,
    base_output: Path,
    min_step: int,
    mode_counts: Sequence[int],
    batch_size: int,
    dpi: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    run_dir = zarr_path.parent
    dense0_dir = zarr_path / "Dense_0" / "kernel"
    dense1_dir = zarr_path / "Dense_1" / "kernel"
    if not dense0_dir.is_dir() or not dense1_dir.is_dir():
        print(f"Skipping spectral reconstruction for {zarr_path}: missing Dense_0 or Dense_1 kernel")
        return None

    config = load_yaml(run_dir / "simulation_config.yaml")
    network = config.get("network", {})
    training = config.get("training", {})
    configured_batch_size = float(training.get("batch_size", 1.0))
    total_params = float(network.get("total_params", 1.0))
    activations = network.get("activations_per_layer", {})
    activation0 = numpy_activation(str(activations.get("Dense_0", "relu")))
    activation1 = numpy_activation(str(activations.get("Dense_1", "identity")))
    loss_type = str(training.get("loss_type", "mse"))

    dense0_meta = load_json(dense0_dir / "zarr.json")
    dense1_meta = load_json(dense1_dir / "zarr.json")
    dense0_shape = tuple(int(v) for v in dense0_meta["shape"])
    dense1_shape = tuple(int(v) for v in dense1_meta["shape"])
    times = selected_times(dense0_shape[0], training_step_stride(run_dir), min_step)
    if dense1_shape[0] < dense0_shape[0]:
        times = [(time_index, step) for time_index, step in times if time_index < dense1_shape[0]]
    if not times:
        print(f"Skipping spectral reconstruction for {zarr_path}: no selected times")
        return None

    x_test, y_test = load_reconstruction_eval_data(run_dir, data_cache)
    requested_modes = tuple(sorted({int(value) for value in mode_counts if int(value) > 0}))
    if not requested_modes:
        raise ValueError("At least one positive spectral reconstruction mode count is required")
    max_modes = min(max(requested_modes), min(dense0_shape[1], dense0_shape[2]))

    record: dict[str, Any] = {
        "job_dir": spectral_reconstruction_output_dir(base_output, run_dir).parent,
        "run": run_dir.name,
        "steps": [],
        "original_loss": [],
        "input_dim": int(dense0_shape[1]),
        "output_dim": int(dense0_shape[2]),
        "batch_size": configured_batch_size,
        "total_params": total_params,
        "mode_counts": list(requested_modes),
        "losses": {
            "unnormalized": {str(k): [] for k in requested_modes},
        },
        "frob_norms": {
            "unnormalized": {str(k): [] for k in requested_modes},
        },
        "diagnostics": {
            "unnormalized": {str(k): [] for k in requested_modes},
        },
    }

    last_step = None
    last_basis = None
    for time_index, step in times:
        dense0_kernel = read_snapshot(dense0_dir, dense0_meta, time_index)
        dense1_kernel = read_snapshot(dense1_dir, dense1_meta, time_index)
        row_norms, u, singular_values, vt, _ = spectral_basis_for_dense0(dense0_kernel, max_modes)
        last_step = int(step)
        last_basis = (row_norms, u, singular_values, vt)
        original_loss = evaluate_dense_two_layer_loss(
            x_test,
            y_test,
            dense0_kernel,
            dense1_kernel,
            activation0=activation0,
            activation1=activation1,
            loss_type=loss_type,
            batch_size=batch_size,
        )
        record["steps"].append(int(step))
        record["original_loss"].append(float(original_loss))
        for k in requested_modes:
            record["frob_norms"]["unnormalized"][str(k)].append(
                spectral_reconstructed_superposition_frobenius(
                    u,
                    singular_values,
                    k,
                )
            )
            loss, diagnostics = evaluate_reconstructed_loss(
                x_test,
                y_test,
                dense1_kernel,
                row_norms=row_norms,
                u=u,
                singular_values=singular_values,
                vt=vt,
                modes=k,
                renormalize=False,
                activation0=activation0,
                activation1=activation1,
                loss_type=loss_type,
                batch_size=batch_size,
            )
            record["losses"]["unnormalized"][str(k)].append(float(loss))
            record["diagnostics"]["unnormalized"][str(k)].append(diagnostics)
    if last_step is not None and last_basis is not None:
        row_norms, u, singular_values, vt = last_basis
        save_spectral_reconstruction_input_space_frames(
            run_dir=run_dir,
            base_output=base_output,
            step=last_step,
            mode_counts=requested_modes,
            row_norms=row_norms,
            u=u,
            singular_values=singular_values,
            vt=vt,
            dpi=dpi,
        )
    print(f"Computed spectral reconstructions for {len(times)} checkpoints in {zarr_path}")
    return record


def spectral_reconstruction_frob_values(entry: dict[str, Any], modes: int) -> np.ndarray:
    return np.asarray(
        entry.get("frob_norms", {})
        .get("unnormalized", {})
        .get(str(int(modes)), []),
        dtype=float,
    )


def spectral_reconstruction_baseline_subtracted_frob(
    entry: dict[str, Any],
    modes: int,
) -> np.ndarray:
    values = spectral_reconstruction_frob_values(entry, modes)
    if "input_dim" not in entry or "output_dim" not in entry:
        return np.empty(0, dtype=float)
    return subtract_random_superposition_baseline(
        int(entry["input_dim"]),
        int(entry["output_dim"]),
        values,
    )


def set_padded_ylim(ax: plt.Axes, arrays: Sequence[np.ndarray]) -> None:
    finite = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in arrays]
    finite = [values for values in finite if values.size]
    if not finite:
        return
    values = np.concatenate(finite)
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin
    pad = max(0.08 * span, 0.02 * max(abs(ymin), abs(ymax), 1.0), 1e-3)
    ax.set_ylim(ymin - pad, ymax + pad)


def save_spectral_reconstruction_frob_grid(
    entries: list[dict[str, Any]],
    out_dir: Path,
    mode_counts: Sequence[int],
    dpi: int,
    *,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
    y_values: Any,
    x_values: Any,
    zero_line: bool = False,
) -> None:
    if not mode_counts:
        return

    ncols = min(2, max(1, len(mode_counts)))
    nrows = int(np.ceil(len(mode_counts) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 4.1 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    colors, legend_entries = series_colors(entries)
    legend_runs: set[str] = set()

    for panel_index, modes in enumerate(mode_counts):
        ax = axes_flat[panel_index]
        panel_values: list[np.ndarray] = []
        panel_x_values: list[np.ndarray] = []
        for entry in sorted_series_entries(entries):
            x = np.asarray(x_values(entry), dtype=float)
            y = np.asarray(y_values(entry, modes), dtype=float)
            count = min(x.size, y.size)
            if count < 1:
                continue
            x = x[:count]
            y = y[:count]
            line, = ax.plot(
                x,
                y,
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=colors[entry["run"]],
            )
            panel_values.append(y)
            panel_x_values.append(x)
            if entry["run"] not in legend_runs:
                legend_entries[scheme_from_run(entry["run"])].append(
                    (line, label_from_run(entry["run"]))
                )
                legend_runs.add(entry["run"])

        set_nonnegative_loglike_xscale(ax, panel_x_values)
        if zero_line:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        set_padded_ylim(ax, panel_values)
        ax.set_title(f"k={int(modes)}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes_flat[len(mode_counts):]:
        ax.set_visible(False)

    fig.suptitle(title)
    handles, labels, header_indices = combined_legend(legend_entries)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            frameon=False,
        )
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
        for index in header_indices:
            legend.get_texts()[index].set_fontweight("bold")
    fig.savefig(out_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_spectral_reconstruction_frob_norm_plots(
    entries: list[dict[str, Any]],
    out_dir: Path,
    mode_counts: Sequence[int],
    dpi: int,
) -> None:
    if not any(entry.get("frob_norms") for entry in entries):
        return

    save_spectral_reconstruction_frob_grid(
        entries,
        out_dir,
        mode_counts,
        dpi,
        filename="frob_norms_spectral_reconstruction_Dense_0.png",
        title="Dense_0 spectral reconstruction off-diagonal Frobenius norm",
        x_label="Training Step",
        y_label=r"$||S_k - \mathrm{diag}(S_k)||_F$",
        y_values=spectral_reconstruction_frob_values,
        x_values=lambda entry: entry["steps"],
    )
    save_spectral_reconstruction_frob_grid(
        entries,
        out_dir,
        mode_counts,
        dpi,
        filename="frob_norms_baseline_subtracted_spectral_reconstruction_Dense_0.png",
        title="Dense_0 spectral reconstruction baseline-subtracted Frobenius norm",
        x_label="Training Step",
        y_label=r"$||S_k - \mathrm{diag}(S_k)||_F - \sqrt{d(d-1)/N}$",
        y_values=spectral_reconstruction_baseline_subtracted_frob,
        x_values=lambda entry: entry["steps"],
        zero_line=True,
    )
    save_spectral_reconstruction_frob_grid(
        entries,
        out_dir,
        mode_counts,
        dpi,
        filename="frob_norms_baseline_subtracted_vs_compute_spectral_reconstruction_Dense_0.png",
        title="Dense_0 spectral reconstruction baseline-subtracted Frobenius norm vs compute",
        x_label=r"Training Compute $c_i$",
        y_label=r"$||S_k - \mathrm{diag}(S_k)||_F - \sqrt{d(d-1)/N}$",
        y_values=spectral_reconstruction_baseline_subtracted_frob,
        x_values=training_compute,
        zero_line=True,
    )
    print(f"Saved spectral reconstruction Frobenius norm plots to {out_dir}")


def save_spectral_reconstruction_results(
    records: list[dict[str, Any]],
    dpi: int,
) -> None:
    if not records:
        return

    groups: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["job_dir"], []).append(record)

    for job_dir, entries in groups.items():
        entries = sorted_series_entries(entries)
        mode_counts = [int(value) for value in entries[0]["mode_counts"]]
        colors, _ = series_colors(entries)
        out_dir = job_dir / "spectral_reconstruction_Dense_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_spectral_reconstruction_frob_norm_plots(entries, out_dir, mode_counts, dpi)

        ncols = min(2, max(1, len(mode_counts)))
        nrows = int(np.ceil(len(mode_counts) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.4 * ncols, 4.2 * nrows),
            sharex=False,
            sharey=False,
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.ravel()

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        for col_index, k in enumerate(mode_counts):
            ax = axes_flat[col_index]
            panel_values: list[np.ndarray] = []
            for entry in entries:
                steps = np.asarray(entry["steps"], dtype=float)
                color = colors[entry["run"]]
                label = label_from_run(entry["run"])
                reconstructed_loss = np.asarray(
                    entry["losses"]["unnormalized"][str(k)],
                    dtype=float,
                )
                original_loss = np.asarray(entry["original_loss"], dtype=float)
                line, = ax.plot(
                    steps,
                    reconstructed_loss,
                    color=color,
                    linewidth=1.2,
                    marker="o",
                    markersize=2.5,
                )
                ax.plot(
                    steps,
                    original_loss,
                    color=color,
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.65,
                )
                panel_values.extend(
                    (
                        reconstructed_loss[np.isfinite(reconstructed_loss)],
                        original_loss[np.isfinite(original_loss)],
                    )
                )
                if col_index == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)
            nonempty_values = [values for values in panel_values if values.size]
            if nonempty_values:
                finite_values = np.concatenate(nonempty_values)
                ymin = float(np.min(finite_values))
                ymax = float(np.max(finite_values))
                span = ymax - ymin
                pad = max(0.08 * span, 0.02 * max(abs(ymin), abs(ymax), 1.0), 1e-3)
                ax.set_ylim(ymin - pad, ymax + pad)
            set_nonnegative_loglike_xscale(
                ax,
                [np.asarray(entry["steps"], dtype=float) for entry in entries],
            )
            ax.set_title(f"Unnormalized, k={k}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Test loss")
            ax.grid(True, which="both", alpha=0.3)

        for ax in axes_flat[len(mode_counts):]:
            ax.set_visible(False)

        fig.suptitle(
            "Dense_0 spectral reconstruction test loss\n"
            "solid: reconstructed Dense_0, dashed: original checkpoint"
        )
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=1,
                frameon=False,
            )
        fig.savefig(
            out_dir / "test_loss_spectral_reconstruction_Dense_0.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        serializable = []
        for entry in entries:
            serializable.append(
                {
                    "run": entry["run"],
                    "steps": entry["steps"],
                    "original_loss": entry["original_loss"],
                    "input_dim": entry.get("input_dim"),
                    "output_dim": entry.get("output_dim"),
                    "batch_size": entry.get("batch_size"),
                    "total_params": entry.get("total_params"),
                    "mode_counts": entry["mode_counts"],
                    "losses": entry["losses"],
                    "frob_norms": entry.get("frob_norms", {}),
                    "diagnostics": entry["diagnostics"],
                }
            )
        with (out_dir / "spectral_reconstruction_results.json").open("w", encoding="utf-8") as file:
            json.dump(serializable, file, indent=2)
        print(f"Saved spectral reconstruction results to {out_dir}")


def dense_run_info_from_store(
    zarr_path: Path,
    base_output: Path,
    min_step: int,
) -> DenseRunInfo | None:
    run_dir = zarr_path.parent
    dense0_dir = zarr_path / "Dense_0" / "kernel"
    dense1_dir = zarr_path / "Dense_1" / "kernel"
    if not dense0_dir.is_dir() or not dense1_dir.is_dir():
        print(f"Skipping largest-network spectral reconstruction for {zarr_path}: missing Dense_0 or Dense_1 kernel")
        return None

    dense0_meta = load_json(dense0_dir / "zarr.json")
    dense1_meta = load_json(dense1_dir / "zarr.json")
    dense0_shape = tuple(int(value) for value in dense0_meta["shape"])
    dense1_shape = tuple(int(value) for value in dense1_meta["shape"])
    times = selected_times(dense0_shape[0], training_step_stride(run_dir), min_step)
    times = [(time_index, step) for time_index, step in times if time_index < dense1_shape[0]]
    if not times:
        print(f"Skipping largest-network spectral reconstruction for {zarr_path}: no selected times")
        return None

    config = load_yaml(run_dir / "simulation_config.yaml")
    network = config.get("network", {})
    training = config.get("training", {})
    activations = network.get("activations_per_layer", {})
    input_dim = int(np.prod(dense0_shape[1:-1]))

    return DenseRunInfo(
        zarr_path=zarr_path,
        run_dir=run_dir,
        job_dir=job_output_dir(base_output, run_dir),
        dense0_dir=dense0_dir,
        dense1_dir=dense1_dir,
        dense0_meta=dense0_meta,
        dense1_meta=dense1_meta,
        dense0_shape=dense0_shape,
        dense1_shape=dense1_shape,
        step_to_time_index={int(step): int(time_index) for time_index, step in times},
        input_dim=input_dim,
        output_dim=int(dense0_shape[-1]),
        total_params=float(network.get("total_params", 1.0)),
        batch_size=float(training.get("batch_size", 1.0)),
        activation0=numpy_activation(str(activations.get("Dense_0", "relu"))),
        activation1=numpy_activation(str(activations.get("Dense_1", "identity"))),
        loss_type=str(training.get("loss_type", "mse")),
    )


def read_dense0_matrix(info: DenseRunInfo, step: int) -> np.ndarray:
    return weight_vectors_by_input_unit(
        read_snapshot(info.dense0_dir, info.dense0_meta, info.step_to_time_index[int(step)])
    )


def read_dense1_matrix(info: DenseRunInfo, step: int) -> np.ndarray:
    return read_snapshot(info.dense1_dir, info.dense1_meta, info.step_to_time_index[int(step)])


def evaluate_largest_basis_reconstructed_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense0_kernel: np.ndarray,
    dense1_kernel: np.ndarray,
    *,
    source_u: np.ndarray,
    source_singular_values: np.ndarray,
    modes: int,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    row_norms = np.linalg.norm(dense0_kernel, axis=1).astype(np.float32)
    normalized = normalized_input_vectors(dense0_kernel)
    k_eff = min(int(modes), source_singular_values.size, source_u.shape[1])
    if k_eff < 1:
        raise ValueError("Largest-network spectral reconstruction requires at least one source mode")

    source_u_k = source_u[:, :k_eff]
    source_s_k = source_singular_values[:k_eff]
    projected = source_u_k.T @ normalized
    hidden_modes_t = np.divide(
        projected,
        source_s_k[:, None],
        out=np.zeros_like(projected, dtype=np.float32),
        where=source_s_k[:, None] > 0,
    )
    left = source_u_k * source_s_k[None, :]

    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:stop]
        hidden_pre = ((xb * row_norms[None, :]) @ left) @ hidden_modes_t
        logits = activation1(activation0(hidden_pre) @ dense1_kernel)
        yb = y_test[start:stop]
        if loss_type == "cross_entropy":
            loss = classification_cross_entropy(logits, yb)
        elif loss_type == "mse":
            loss = mean_squared_error(logits, yb)
        else:
            raise ValueError(f"Unsupported loss_type {loss_type!r}")
        losses.append(loss)
        weights.append(stop - start)

    total_norm_sq = float(np.count_nonzero(row_norms > 0))
    retained_total = float(np.sum(np.square(projected, dtype=np.float64)))
    err_q = float(np.sqrt(max(total_norm_sq - retained_total, 0.0) / max(total_norm_sq, 1.0)))
    diagnostics = {
        "k_effective": float(k_eff),
        "source_eigenvalue_sum": float(np.sum(np.square(source_s_k, dtype=np.float64))),
        "retained_fraction": float(retained_total / max(total_norm_sq, 1.0)),
        "err_Q": err_q,
    }
    return float(np.average(losses, weights=weights)), diagnostics


def largest_network_spectral_reconstruction_output_dir(base: Path, run_dir: Path) -> Path:
    return job_output_dir(base, run_dir) / "largest_network_spectral_reconstruction_Dense_0"


def process_largest_network_spectral_reconstruction_group(
    infos: list[DenseRunInfo],
    batch_size: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    if not infos:
        return []

    infos = sorted(infos, key=lambda info: tuple(width_key(label_from_run(info.run_dir.name))))
    source = max(infos, key=lambda info: (info.output_dim, info.total_params))
    source_steps = sorted(source.step_to_time_index)
    max_target_modes = max(
        (
            min(info.output_dim, source.input_dim, source.output_dim)
            for info in infos
            if info.run_dir != source.run_dir and info.input_dim == source.input_dim
        ),
        default=0,
    )

    records: dict[str, dict[str, Any]] = {
        info.run_dir.name: {
            "job_dir": info.job_dir,
            "run": info.run_dir.name,
            "source_run": source.run_dir.name,
            "source_output_dim": source.output_dim,
            "source_total_params": source.total_params,
            "input_dim": info.input_dim,
            "output_dim": info.output_dim,
            "total_params": info.total_params,
            "batch_size": info.batch_size,
            "steps": [],
            "original_loss": [],
            "largest_spectrum_reconstructed_loss": [],
            "k_modes": [],
            "diagnostics": [],
        }
        for info in infos
    }

    for step in source_steps:
        source_u = None
        source_s = None
        if max_target_modes > 0:
            source_dense0 = read_dense0_matrix(source, step)
            _, source_u, source_s, _, _ = spectral_basis_for_dense0(source_dense0, max_target_modes)

        for info in infos:
            if step not in info.step_to_time_index:
                continue
            if info.input_dim != source.input_dim:
                print(
                    f"Skipping {info.run_dir.name} at step {step}: input dimension "
                    f"{info.input_dim} differs from source {source.input_dim}"
                )
                continue

            dense0_kernel = read_dense0_matrix(info, step)
            dense1_kernel = read_dense1_matrix(info, step)
            x_test, y_test = load_reconstruction_eval_data(info.run_dir, data_cache)
            original_loss = evaluate_dense_two_layer_loss(
                x_test,
                y_test,
                dense0_kernel,
                dense1_kernel,
                activation0=info.activation0,
                activation1=info.activation1,
                loss_type=info.loss_type,
                batch_size=batch_size,
            )

            if info.run_dir == source.run_dir:
                reconstructed_loss = original_loss
                diagnostics = {
                    "k_effective": float(min(info.output_dim, info.input_dim)),
                    "source_reference": True,
                    "retained_fraction": 1.0,
                    "err_Q": 0.0,
                }
                k_modes = min(info.output_dim, info.input_dim)
            else:
                if source_u is None or source_s is None:
                    raise RuntimeError("Missing source spectral basis for non-source reconstruction")
                k_modes = min(info.output_dim, source_s.size)
                reconstructed_loss, diagnostics = evaluate_largest_basis_reconstructed_loss(
                    x_test,
                    y_test,
                    dense0_kernel,
                    dense1_kernel,
                    source_u=source_u,
                    source_singular_values=source_s,
                    modes=k_modes,
                    activation0=info.activation0,
                    activation1=info.activation1,
                    loss_type=info.loss_type,
                    batch_size=batch_size,
                )

            record = records[info.run_dir.name]
            record["steps"].append(int(step))
            record["original_loss"].append(float(original_loss))
            record["largest_spectrum_reconstructed_loss"].append(float(reconstructed_loss))
            record["k_modes"].append(int(k_modes))
            record["diagnostics"].append(diagnostics)

    result = [record for record in records.values() if record["steps"]]
    print(
        f"Computed largest-network spectral reconstruction for {len(result)} runs "
        f"using source {source.run_dir.name}"
    )
    return result


def safe_plot_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def save_largest_network_spectral_reconstruction_results(
    records: list[dict[str, Any]],
    dpi: int,
) -> None:
    if not records:
        return

    groups: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["job_dir"], []).append(record)

    for job_dir, entries in groups.items():
        entries = sorted_series_entries(entries)
        source_run = str(entries[0]["source_run"])
        out_dir = job_dir / "largest_network_spectral_reconstruction_Dense_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        colors, legend_entries = series_colors(entries)

        fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        panel_values: list[np.ndarray] = []
        for entry in entries:
            steps = np.asarray(entry["steps"], dtype=float)
            if entry["run"] == source_run:
                continue
            losses = np.asarray(entry["largest_spectrum_reconstructed_loss"], dtype=float)
            line, = ax.plot(
                steps,
                losses,
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=colors[entry["run"]],
            )
            legend_entries[scheme_from_run(entry["run"])].append((line, label_from_run(entry["run"])))
            panel_values.append(losses)

        source_entry = next((entry for entry in entries if entry["run"] == source_run), None)
        if source_entry is not None:
            source_steps = np.asarray(source_entry["steps"], dtype=float)
            source_loss = np.asarray(source_entry["original_loss"], dtype=float)
            ax.plot(
                source_steps,
                source_loss,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=f"{label_from_run(source_run)} original source",
            )
            panel_values.append(source_loss)

        all_steps = np.concatenate([np.asarray(entry["steps"], dtype=float) for entry in entries])
        set_nonnegative_loglike_xscale(ax, [all_steps])
        set_padded_ylim(ax, panel_values)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Test loss")
        ax.set_title(
            "Dense_0 reconstruction from largest-network spectral modes\n"
            "solid: smaller runs reconstructed from source spectrum, dashed: largest original"
        )
        ax.grid(True, which="both", alpha=0.3)
        handles, labels, header_indices = combined_legend(legend_entries)
        source_handles, source_labels = ax.get_legend_handles_labels()
        handles.extend(source_handles)
        labels.extend(source_labels)
        if handles:
            legend = ax.legend(handles, labels, fontsize=8, frameon=False)
            legend._legend_box.align = "left"  # type: ignore[attr-defined]
            for index in header_indices:
                legend.get_texts()[index].set_fontweight("bold")
        fig.savefig(out_dir / "test_loss_largest_network_spectral_reconstruction_Dense_0.png", dpi=dpi)
        plt.close(fig)

        per_run_dir = out_dir / "per_run_test_loss_comparison"
        per_run_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
            steps = np.asarray(entry["steps"], dtype=float)
            original = np.asarray(entry["original_loss"], dtype=float)
            reconstructed = np.asarray(entry["largest_spectrum_reconstructed_loss"], dtype=float)
            ax.plot(
                steps,
                original,
                color="black",
                linewidth=1.2,
                marker="o",
                markersize=3.0,
                label="original checkpoint",
            )
            ax.plot(
                steps,
                reconstructed,
                color=colors[entry["run"]],
                linewidth=1.2,
                linestyle="--",
                marker="o",
                markersize=3.0,
                label="largest-spectrum reconstruction",
            )
            set_nonnegative_loglike_xscale(ax, [steps])
            set_padded_ylim(ax, [original, reconstructed])
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Test loss")
            ax.set_title(
                f"{label_from_run(entry['run'])}: original vs largest-spectrum reconstruction\n"
                f"source: {label_from_run(source_run)}"
            )
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, frameon=False)
            fig.savefig(
                per_run_dir / f"{safe_plot_name(entry['run'])}_test_loss_comparison.png",
                dpi=dpi,
            )
            plt.close(fig)

        serializable = []
        for entry in entries:
            serializable.append(
                {
                    "run": entry["run"],
                    "source_run": entry["source_run"],
                    "source_output_dim": entry["source_output_dim"],
                    "source_total_params": entry["source_total_params"],
                    "input_dim": entry["input_dim"],
                    "output_dim": entry["output_dim"],
                    "total_params": entry["total_params"],
                    "batch_size": entry["batch_size"],
                    "steps": entry["steps"],
                    "original_loss": entry["original_loss"],
                    "largest_spectrum_reconstructed_loss": entry["largest_spectrum_reconstructed_loss"],
                    "k_modes": entry["k_modes"],
                    "diagnostics": entry["diagnostics"],
                }
            )
        with (out_dir / "largest_network_spectral_reconstruction_results.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(serializable, file, indent=2)
        print(f"Saved largest-network spectral reconstruction results to {out_dir}")


def process_largest_network_spectral_reconstruction(
    stores: Sequence[Path],
    base_output: Path,
    min_step: int,
    batch_size: int,
    dpi: int,
) -> None:
    groups: dict[Path, list[DenseRunInfo]] = {}
    for zarr_path in stores:
        info = dense_run_info_from_store(zarr_path, base_output, min_step)
        if info is None:
            continue
        groups.setdefault(info.job_dir, []).append(info)

    all_records: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
    for infos in groups.values():
        all_records.extend(
            process_largest_network_spectral_reconstruction_group(
                infos,
                batch_size,
                data_cache,
            )
        )
    save_largest_network_spectral_reconstruction_results(all_records, dpi)


def process_store(
    zarr_path: Path,
    base_output: Path,
    min_step: int,
    dpi: int,
    layer_filter: str | None,
    input_space: bool,
    input_space_only: bool,
    aggregate_only: bool,
    input_space_layer: str,
    input_space_reference_count: int,
    input_space_channel_mode: str,
) -> tuple[int, list[dict[str, Any]]]:
    run_dir = zarr_path.parent
    keys = matrix_keys(zarr_path)
    if not keys:
        print(f"Skipping {zarr_path}: no matrix-valued param_keys found")
        return 0, []

    step_stride = training_step_stride(run_dir)
    simulation_config = load_yaml(run_dir / "simulation_config.yaml")
    training_config = simulation_config.get("training", {})
    network_config = simulation_config.get("network", {})
    batch_size = float(training_config.get("batch_size", 1.0))
    total_params = float(network_config.get("total_params", 1.0))
    frame_count = 0
    frob_series: list[dict[str, Any]] = []
    for key in keys:
        layer = layer_name(key)
        if layer_filter is not None and layer != layer_filter:
            continue
        array_dir = zarr_path / key
        meta = load_json(array_dir / "zarr.json")
        shape = tuple(int(v) for v in meta["shape"])
        times = selected_times(shape[0], step_stride, min_step)
        if not times:
            print(f"Skipping {zarr_path}/{key}: no saved steps >= {min_step} match the log grid")
            continue
        frames_dir = output_dir(base_output, run_dir, layer)
        unit_count = int(np.prod(shape[1:-1]))
        layout = None
        reference_units = np.empty(0, dtype=np.int64)
        plot_input_space = input_space and layer == input_space_layer
        if plot_input_space:
            layout = infer_image_layout(run_dir, unit_count)
            if layout is None:
                print(
                    f"Skipping input-space frames for {zarr_path}/{key}: "
                    f"could not infer an image layout for {unit_count} input units"
                )
                plot_input_space = False
            else:
                reference_units = selected_reference_units(
                    unit_count,
                    input_space_reference_count,
                )
                if reference_units.size == 0:
                    plot_input_space = False

        if input_space_only and not plot_input_space:
            continue

        input_frames_dir = input_space_output_dir(base_output, run_dir, layer)

        if aggregate_only:
            steps = []
            frob_norms = []
            eigenvalues_by_step = []
            for time_index, step in times:
                matrix = weight_vectors_by_input_unit(read_snapshot(array_dir, meta, time_index))
                frob_norm, eigenvalues = superposition_summary(matrix)
                steps.append(step)
                frob_norms.append(frob_norm)
                eigenvalues_by_step.append((step, eigenvalues))
            frob_series.append(
                {
                    "job_dir": job_output_dir(base_output, run_dir),
                    "layer": layer,
                    "run": run_dir.name,
                    "steps": np.asarray(steps, dtype=np.int64),
                    "frob_norms": np.asarray(frob_norms, dtype=float),
                    "input_dim": unit_count,
                    "output_dim": int(shape[-1]),
                    "batch_size": batch_size,
                    "total_params": total_params,
                    "eigenvalues_by_step": eigenvalues_by_step,
                }
            )
            print(f"Computed {len(times)} aggregate summaries for {zarr_path}/{key}")
            continue

        if input_space_only:
            input_frame_count = 0
            for frame_index, (time_index, step) in enumerate(times):
                matrix = weight_vectors_by_input_unit(read_snapshot(array_dir, meta, time_index))
                columns = cosine_columns(matrix, reference_units)
                assert layout is not None
                plot_input_space_frame(
                    columns,
                    reference_units,
                    layout,
                    input_frames_dir / f"input_space_{frame_index:03d}_step_{step}.png",
                    title=(
                        f"{run_dir.parent.name}/{run_dir.name}/{layer} "
                        f"input-space superposition - Training Step: {step}"
                    ),
                    channel_mode=input_space_channel_mode,
                    dpi=dpi,
                )
                input_frame_count += 1
                frame_count += 1
            print(
                f"Saved {input_frame_count} input-space frames for "
                f"{zarr_path}/{key} to {input_frames_dir}"
            )
            continue

        frame_data = []
        steps = []
        frob_norms = []
        eigenvalues_by_step = []
        overlap_span = 0.0
        for time_index, step in times:
            matrix = weight_vectors_by_input_unit(read_snapshot(array_dir, meta, time_index))
            cosine, norms, overlap = cosine_from_input_vectors(matrix)
            input_columns = cosine[:, reference_units] if plot_input_space else None
            overlap_span = max(overlap_span, float(np.max(np.abs(overlap))) if overlap.size else 0.0)
            steps.append(step)
            frob_norms.append(offdiag_frobenius(cosine))
            _, eigenvalues = superposition_summary(matrix)
            eigenvalues_by_step.append((step, eigenvalues))
            frame_data.append((step, cosine, norms, overlap, input_columns))

        input_frame_count = 0
        for frame_index, (step, cosine, norms, overlap, input_columns) in enumerate(frame_data):
            plot_frame(
                cosine,
                norms,
                overlap,
                frames_dir / f"sup_{frame_index:03d}_step_{step}.png",
                title=f"{run_dir.parent.name}/{run_dir.name}/{layer} - Training Step: {step}",
                overlap_span=overlap_span,
                dpi=dpi,
            )
            frame_count += 1
            if plot_input_space and input_columns is not None:
                assert layout is not None
                plot_input_space_frame(
                    input_columns,
                    reference_units,
                    layout,
                    input_frames_dir / f"input_space_{frame_index:03d}_step_{step}.png",
                    title=(
                        f"{run_dir.parent.name}/{run_dir.name}/{layer} "
                        f"input-space superposition - Training Step: {step}"
                    ),
                    channel_mode=input_space_channel_mode,
                    dpi=dpi,
                )
                input_frame_count += 1
                frame_count += 1
        frob_series.append(
            {
                "job_dir": job_output_dir(base_output, run_dir),
                "layer": layer,
                "run": run_dir.name,
                "steps": np.asarray(steps, dtype=np.int64),
                "frob_norms": np.asarray(frob_norms, dtype=float),
                "input_dim": unit_count,
                "output_dim": int(shape[-1]),
                "batch_size": batch_size,
                "total_params": total_params,
                "eigenvalues_by_step": eigenvalues_by_step,
            }
        )
        print(f"Saved {len(times)} frames for {zarr_path}/{key} to {frames_dir}")
        if plot_input_space:
            print(
                f"Saved {input_frame_count} input-space frames for "
                f"{zarr_path}/{key} to {input_frames_dir}"
            )
    return frame_count, frob_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot column-cosine superposition matrices from weights.zarr training runs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures_superposition"))
    parser.add_argument("--min-step", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--layer", type=str, default=None, help="Optional layer filter, e.g. Dense_0.")
    parser.add_argument(
        "--input-space",
        action="store_true",
        help=(
            "Also plot selected Dense_0 superposition columns in the original image "
            "coordinate system when an image layout can be inferred."
        ),
    )
    parser.add_argument(
        "--input-space-only",
        action="store_true",
        help=(
            "Generate only the image-coordinate superposition frames. This implies "
            "--input-space and skips the full matrix frames and Frobenius plots."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Compute only aggregate plots: Frobenius norms, effective participation "
            "ratio, and eigenvalue spectra. Skips per-checkpoint superposition frames."
        ),
    )
    parser.add_argument(
        "--spectral-reconstruction",
        action="store_true",
        help=(
            "Evaluate Dense_0 spectral reconstructions using top collective modes and "
            "plot reconstructed-network test losses."
        ),
    )
    parser.add_argument(
        "--spectral-reconstruction-only",
        action="store_true",
        help=(
            "Run only Dense_0 spectral reconstruction test-loss analysis. Skips all "
            "superposition frame and aggregate plots."
        ),
    )
    parser.add_argument(
        "--largest-network-spectral-reconstruction",
        action="store_true",
        help=(
            "Evaluate Dense_0 reconstructions where each smaller run is projected onto "
            "the largest-width run's top spectral modes at each selected checkpoint."
        ),
    )
    parser.add_argument(
        "--largest-network-spectral-reconstruction-only",
        action="store_true",
        help=(
            "Run only the largest-network spectral reconstruction analysis. Skips "
            "superposition frame, aggregate, and standard spectral reconstruction plots."
        ),
    )
    parser.add_argument(
        "--collective-mode-visualization",
        action="store_true",
        help=(
            "Visualize Dense_0 input-space collective modes: signed channel maps, "
            "top-k subspace leverage maps, and test examples ranked by mode response."
        ),
    )
    parser.add_argument(
        "--collective-mode-visualization-only",
        action="store_true",
        help=(
            "Run only the Dense_0 collective mode visualization analysis. Skips "
            "superposition frames, aggregate plots, and spectral reconstruction."
        ),
    )
    parser.add_argument(
        "--spectral-reconstruction-modes",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="Collective mode counts to use for Dense_0 spectral reconstruction.",
    )
    parser.add_argument(
        "--spectral-reconstruction-batch-size",
        type=int,
        default=1024,
        help="Evaluation batch size for spectral reconstruction test-loss computation.",
    )
    parser.add_argument(
        "--input-space-layer",
        type=str,
        default="Dense_0",
        help="Layer for image-coordinate superposition frames.",
    )
    parser.add_argument(
        "--input-space-reference-count",
        type=int,
        default=5,
        help="Number of equidistant flattened input units to use as reference columns.",
    )
    parser.add_argument(
        "--input-space-channel-mode",
        choices=("split", "reference"),
        default="split",
        help=(
            "For RGB inputs, 'split' shows R/G/B heatmap rows for each reference unit; "
            "'reference' shows only the channel of the selected reference unit."
        ),
    )
    parser.add_argument(
        "--collective-mode-top-modes",
        type=int,
        default=6,
        help="Number of largest-eigenvalue input-space modes to visualize individually.",
    )
    parser.add_argument(
        "--collective-mode-leverage-k",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="Top-k mode counts used for subspace leverage maps.",
    )
    parser.add_argument(
        "--collective-mode-example-count",
        type=int,
        default=4,
        help="Number of most positive and most negative test examples to show per mode.",
    )
    parser.add_argument(
        "--collective-mode-example-pool",
        type=int,
        default=0,
        help=(
            "Optional cap on examples scored per mode. Use 0 to score the full test "
            "set loaded by the run configuration."
        ),
    )
    args = parser.parse_args()
    if args.input_space_reference_count < 1:
        raise ValueError("--input-space-reference-count must be positive")
    if args.aggregate_only and args.input_space_only:
        raise ValueError("--aggregate-only and --input-space-only cannot be combined")
    if args.spectral_reconstruction_batch_size < 1:
        raise ValueError("--spectral-reconstruction-batch-size must be positive")
    if args.collective_mode_top_modes < 1:
        raise ValueError("--collective-mode-top-modes must be positive")
    if args.collective_mode_example_count < 0:
        raise ValueError("--collective-mode-example-count cannot be negative")
    if args.collective_mode_example_pool < 0:
        raise ValueError("--collective-mode-example-pool cannot be negative")
    if args.spectral_reconstruction_only:
        args.spectral_reconstruction = True
    if args.largest_network_spectral_reconstruction_only:
        args.largest_network_spectral_reconstruction = True
    if args.collective_mode_visualization_only:
        args.collective_mode_visualization = True

    stores = zarr_paths(args.input_dir.resolve())
    if not stores:
        raise FileNotFoundError(f"No weights.zarr directories found under {args.input_dir}")

    total = 0
    frob_series: list[dict[str, Any]] = []
    input_space = args.input_space or args.input_space_only
    if (
        not args.spectral_reconstruction_only
        and not args.largest_network_spectral_reconstruction_only
        and not args.collective_mode_visualization_only
    ):
        for path in stores:
            count, series = process_store(
                path,
                args.output_dir,
                args.min_step,
                args.dpi,
                args.layer,
                input_space,
                args.input_space_only,
                args.aggregate_only,
                args.input_space_layer,
                args.input_space_reference_count,
                args.input_space_channel_mode,
            )
            total += count
            frob_series.extend(series)
        save_frob_norm_plots(frob_series, args.dpi)
        save_effective_participation_ratio_plots(frob_series, args.dpi)
        save_baseline_subtracted_frob_scaling_fits(frob_series, args.dpi)
        save_eigenvalue_spectrum_plots(frob_series, args.dpi)

    if args.spectral_reconstruction:
        data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
        reconstruction_records = []
        for path in stores:
            record = process_spectral_reconstruction_store(
                path,
                args.output_dir,
                args.min_step,
                args.spectral_reconstruction_modes,
                args.spectral_reconstruction_batch_size,
                args.dpi,
                data_cache,
            )
            if record is not None:
                reconstruction_records.append(record)
        save_spectral_reconstruction_results(reconstruction_records, args.dpi)
    if args.largest_network_spectral_reconstruction:
        process_largest_network_spectral_reconstruction(
            stores,
            args.output_dir,
            args.min_step,
            args.spectral_reconstruction_batch_size,
            args.dpi,
        )
    if args.collective_mode_visualization:
        data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
        for path in stores:
            total += process_collective_mode_visualization_store(
                path,
                args.output_dir,
                args.min_step,
                args.collective_mode_top_modes,
                args.collective_mode_leverage_k,
                args.collective_mode_example_count,
                args.collective_mode_example_pool,
                args.dpi,
                data_cache,
            )
    print(f"Saved {total} output files from {len(stores)} weights.zarr stores.")


if __name__ == "__main__":
    main()
