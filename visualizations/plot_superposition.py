#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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


def zarr_paths(input_dir: Path) -> list[Path]:
    if input_dir.name == "weights.zarr" and input_dir.is_dir():
        return [input_dir]
    return sorted(path for path in input_dir.rglob("weights.zarr") if path.is_dir())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def keep_log_step(step: int, min_step: int) -> bool:
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


def cosine_over_outputs(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_dim = weights.shape[-1]
    matrix = weights.reshape(-1, output_dim)
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    gram = matrix @ matrix.T
    denom = norms[:, None] * norms[None, :]
    cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    overlap = cosine.sum(axis=1) - np.diag(cosine)
    return cosine, norms, overlap


def offdiag_frobenius(cosine: np.ndarray) -> float:
    offdiag = cosine.copy()
    np.fill_diagonal(offdiag, 0.0)
    return float(np.linalg.norm(offdiag))


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


def job_output_dir(base: Path, run_dir: Path) -> Path:
    parts = [base]
    task = task_name(run_dir)
    if task:
        parts.append(Path(task))
    parts.append(Path(run_dir.parent.name))
    return Path(*parts)


def output_dir(base: Path, run_dir: Path, layer: str) -> Path:
    return job_output_dir(base, run_dir) / run_dir.name / layer / "sup_frames"


def save_frob_norm_plots(series: list[dict[str, Any]], dpi: int) -> None:
    if not series:
        return

    layers = {entry["layer"] for entry in series}
    groups: dict[tuple[Path, str], list[dict[str, Any]]] = {}
    for entry in series:
        groups.setdefault((entry["job_dir"], entry["layer"]), []).append(entry)

    for (job_dir, layer), entries in groups.items():
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        for entry in sorted(entries, key=lambda item: item["run"]):
            ax.plot(
                entry["steps"],
                entry["frob_norms"],
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                label=entry["run"],
            )

        all_steps = np.concatenate([entry["steps"] for entry in entries])
        if np.all(all_steps > 0):
            ax.set_xscale("log")
        ax.set_xlabel("Training Step")
        ax.set_ylabel(r"$||S||_F$")
        # ax.set_ylabel(r"$\exp(||S - \mathrm{diag}(S)||_F)$")
        ax.set_title(f"Exponential off-diagonal superposition Frobenius norm - {layer}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / ("frob_norms.png" if len(layers) == 1 else f"frob_norms_{layer}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print(f"Saved Frobenius norm plot to {path}")


def process_store(
    zarr_path: Path,
    base_output: Path,
    min_step: int,
    dpi: int,
    layer_filter: str | None,
) -> tuple[int, list[dict[str, Any]]]:
    run_dir = zarr_path.parent
    keys = matrix_keys(zarr_path)
    if not keys:
        print(f"Skipping {zarr_path}: no matrix-valued param_keys found")
        return 0, []

    step_stride = training_step_stride(run_dir)
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

        frame_data = []
        steps = []
        frob_norms = []
        overlap_span = 0.0
        for time_index, step in times:
            cosine, norms, overlap = cosine_over_outputs(read_snapshot(array_dir, meta, time_index))
            overlap_span = max(overlap_span, float(np.max(np.abs(overlap))) if overlap.size else 0.0)
            steps.append(step)
            frob_norms.append(offdiag_frobenius(cosine))
            frame_data.append((step, cosine, norms, overlap))

        for frame_index, (step, cosine, norms, overlap) in enumerate(frame_data):
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
        frob_series.append(
            {
                "job_dir": job_output_dir(base_output, run_dir),
                "layer": layer,
                "run": run_dir.name,
                "steps": np.asarray(steps, dtype=np.int64),
                "frob_norms": np.asarray(frob_norms, dtype=float),
            }
        )
        print(f"Saved {len(times)} frames for {zarr_path}/{key} to {frames_dir}")
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
    args = parser.parse_args()

    stores = zarr_paths(args.input_dir.resolve())
    if not stores:
        raise FileNotFoundError(f"No weights.zarr directories found under {args.input_dir}")

    total = 0
    frob_series: list[dict[str, Any]] = []
    for path in stores:
        count, series = process_store(path, args.output_dir, args.min_step, args.dpi, args.layer)
        total += count
        frob_series.extend(series)
    save_frob_norm_plots(frob_series, args.dpi)
    print(f"Saved {total} superposition frames from {len(stores)} weights.zarr stores.")


if __name__ == "__main__":
    main()
