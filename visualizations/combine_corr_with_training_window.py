from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

MPL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


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


def load_yaml_as_dict(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


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


def collect_weight_metrics(weight_metrics_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    weight_metrics = load_training_log(weight_metrics_path)
    training_metrics = weight_metrics.get("training_metrics", {})
    step_norms = training_metrics.get("step_norms")
    similarities = training_metrics.get("similarities")
    if not step_norms:
        raise ValueError(f"No step norms history found in {weight_metrics_path}")
    if not similarities:
        raise ValueError(f"No similarities history found in {weight_metrics_path}")
    return np.asarray(step_norms), np.asarray(similarities)


def collect_loss_history(log_path: Path) -> np.ndarray:
    log_data = load_training_log(log_path)
    history = log_data.get("final_metrics", {}).get("history", {})
    losses = history.get("train_loss")
    if not losses:
        raise ValueError(f"No loss history found in {log_path}")
    return np.asarray(losses)


def find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_job_id(*paths: Path) -> str | None:
    pattern = re.compile(r"job-\d+")
    for path in paths:
        for part in path.parts:
            match = pattern.fullmatch(part)
            if match:
                return match.group(0)
            match = pattern.search(part)
            if match:
                return match.group(0)
    return None


def infer_job_dir(repo_root: Path, job_dir: Path | None, *context_paths: Path) -> Path:
    if job_dir is not None:
        if not job_dir.exists():
            raise FileNotFoundError(f"Job directory does not exist: {job_dir}")
        return job_dir

    job_id = parse_job_id(*context_paths)
    if job_id is None:
        raise ValueError(
            "Could not infer the job id from the provided paths. Pass --job-dir explicitly."
        )

    candidates = [
        path
        for path in repo_root.rglob(job_id)
        if path.is_dir() and path.parent.name == "logs"
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a logs directory for {job_id}. Pass --job-dir explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple log directories for {job_id}: {candidates}. Pass --job-dir explicitly."
        )
    return candidates[0]


def infer_layer(corr_dir: Path, metrics_image: Path, layer: str | None) -> str:
    if layer:
        return layer

    metric_match = re.search(r"weight_metrics_(.+?)(?:_c)?\.png$", metrics_image.name)
    if metric_match:
        return metric_match.group(1)

    if corr_dir.name.startswith("Dense_") or corr_dir.name == "all_weights":
        return corr_dir.name

    raise ValueError("Could not infer the target layer. Pass --layer explicitly.")


def infer_run_name(corr_dir: Path, run_name: str | None) -> str:
    if run_name:
        return run_name
    if corr_dir.parent.name:
        return corr_dir.parent.name
    raise ValueError("Could not infer the training run directory. Pass --run-name explicitly.")


def load_save_loss_frequency(simulation_config_path: Path) -> int:
    simulation_info = load_yaml_as_dict(simulation_config_path)
    value = simulation_info.get("training", {}).get("save_loss_frequency", 1)
    if isinstance(value, str):
        if value.strip().lower() == "epoch":
            return 1
        try:
            return int(float(value))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported save_loss_frequency value in {simulation_config_path}: {value}"
            ) from exc
    return int(value)


def sorted_corr_frames(corr_dir: Path) -> List[Path]:
    frame_paths = sorted(corr_dir.glob("corr_*_log.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No correlation frames found in {corr_dir}")
    return frame_paths


def keep_log_decade_frame(window_start: int) -> bool:
    if window_start == 0:
        return True
    if window_start < 100:
        return False
    step = 10 ** (len(str(window_start)) - 1)
    return window_start % step == 0


def infer_delta_t(
    run_dir: Path,
    layer: str,
    delta_t: int | None,
) -> int:
    if delta_t is not None:
        return delta_t

    weight_metrics_dir = run_dir / "weight_metrics"
    xj_path = weight_metrics_dir / f"xj_{layer}.npy"
    cov_path = weight_metrics_dir / f"cov_{layer}.npy"
    if not xj_path.exists() or not cov_path.exists():
        raise FileNotFoundError(
            "Could not infer delta_t because one of the required files is missing: "
            f"{xj_path} {cov_path}. Pass --delta-t explicitly."
        )

    xj_length = int(np.load(xj_path, mmap_mode="r").shape[0])
    cov_length = int(np.load(cov_path, mmap_mode="r").shape[0])
    inferred_delta_t = xj_length - cov_length
    if inferred_delta_t < 1:
        raise ValueError(
            f"Expected xj length > covariance length for {layer}, got {xj_length} and {cov_length}."
        )
    return inferred_delta_t


def infer_snapshot_stride(
    run_dir: Path,
    layer: str,
    num_frames: int,
    snapshot_stride: int | None,
) -> int:
    if snapshot_stride is not None:
        return snapshot_stride

    cov_path = run_dir / "weight_metrics" / f"cov_{layer}.npy"
    if not cov_path.exists():
        raise FileNotFoundError(
            f"Could not infer snapshot stride because the covariance file is missing: {cov_path}"
        )

    cov_length = int(np.load(cov_path, mmap_mode="r").shape[0])
    matching_strides = [
        stride
        for stride in range(1, cov_length + 1)
        if len(range(0, cov_length, stride)) == num_frames
    ]
    if not matching_strides:
        raise ValueError(
            f"Could not infer a snapshot stride for {num_frames} frames from covariance length {cov_length}. "
            "Pass --snapshot-stride explicitly."
        )
    if len(matching_strides) > 1:
        raise ValueError(
            f"Multiple snapshot strides match {num_frames} frames and covariance length {cov_length}: "
            f"{matching_strides}. Pass --snapshot-stride explicitly."
        )
    return matching_strides[0]


def build_metrics_figure(
    job_dir: Path,
    layer: str,
    compute_flag: bool,
) -> Tuple[plt.Figure, List[plt.Axes]]:
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

    legend_entries: Dict[str, List[Tuple[Any, str]]] = {"standard": [], "muP": []}
    fig, ((cos_log, loss_log), (ax_step_norms, ax_step_norms_ratio)) = plt.subplots(
        2, 2, figsize=(15, 10)
    )

    prev_step_norms: np.ndarray | None = None
    ratio: np.ndarray | None = None
    smallest: int | None = None
    second_largest: int | None = None
    x_axis_for_ratio: np.ndarray | None = None
    training_info: Dict[str, Any] | None = None
    task_name: str | None = None
    save_loss_frequency_for_title: int | None = None

    for i, log_path in enumerate(log_paths):
        simulation_config_path = collect_files_with_ending(log_path.parent, "simulation_config.yaml")[0]
        weight_metrics_path = log_path.parent / "weight_metrics" / f"{layer}.json"
        simulation_info = load_yaml_as_dict(simulation_config_path)

        training_info = simulation_info.get("training", {})
        network_info = simulation_info.get("network", {})
        task_path = training_info.get("training_data", {}).get("task", "")
        task_name = Path(task_path).name

        num_nodes = network_info.get("nodes_per_layer", {}).get("Dense_0")
        if i == 0:
            smallest = num_nodes
        if i == len(log_paths) - 2:
            second_largest = num_nodes

        save_loss_frequency = load_save_loss_frequency(simulation_config_path)
        save_loss_frequency_for_title = save_loss_frequency

        losses = collect_loss_history(log_path)
        step_norms, similarities = collect_weight_metrics(weight_metrics_path)

        scheme = _scheme_from_path(log_path)
        color = color_map[scheme][color_indices[scheme]]
        color_indices[scheme] += 1

        label = extract_after_char(str(log_path), "-", "/")

        if compute_flag:
            parameters = network_info.get("total_params")
            batch_size = training_info.get("batch_size")
            x_axis = np.arange(0, len(losses)) * save_loss_frequency * batch_size * parameters
            x_label = r"Training Compute $c_{i}$ [log]"
        else:
            x_axis = np.arange(0, len(losses)) * save_loss_frequency
            x_label = r"Training Steps $t_{i}$ [log]"

        line, = loss_log.plot(x_axis, losses, color=color)
        line, = cos_log.plot(x_axis[:-2], similarities, color=color, label=label)
        ax_step_norms.plot(x_axis[:-1], step_norms, color=color)
        legend_entries[scheme].append((line, label))

        if prev_step_norms is not None:
            ratio = step_norms / prev_step_norms
            ax_step_norms_ratio.plot(x_axis[:-1], ratio, color=color)

        prev_step_norms = step_norms
        x_axis_for_ratio = x_axis[:-1]

    if training_info is None or task_name is None or save_loss_frequency_for_title is None:
        raise ValueError(f"Could not build the metrics figure for {job_dir}")

    loss_log.set_xscale("log")
    loss_log.set_yscale("log")
    loss_log.set_xlabel(x_label, fontsize=16)
    loss_log.set_ylabel(x_label, fontsize=16)
    loss_log.grid(True, which="both", alpha=0.3)
    loss_log.tick_params(axis="both", labelsize=13)

    cos_log.set_xlabel(x_label, fontsize=16)
    cos_log.set_ylabel(
        r"$\cos(\Delta \vec{W}_{t_{i+1}}^{\," + layer[-1] + r"}, \Delta \vec{W}_{t_{i}}^{\," + layer[-1] + r"})$",
        fontsize=16,
    )
    cos_log.grid(True, alpha=0.3)
    cos_log.set_xscale("log", base=10)
    cos_log.tick_params(axis="both", labelsize=13)

    ax_step_norms.set_xlabel(x_label, fontsize=16)
    ax_step_norms.set_ylabel(
        r"$\| \Delta \vec{W}_{t_{i}}^{\," + layer[-1] + r"} \| = \| \vec{W}_{t_{i+1}}^{\," + layer[-1] + r"} - \vec{W}_{t_{i}}^{\," + layer[-1] + r"}\|$",
        fontsize=16,
    )
    ax_step_norms.grid(True, alpha=0.3)
    ax_step_norms.set_xscale("log", base=10)
    ax_step_norms.tick_params(axis="both", labelsize=13)

    ax_step_norms_ratio.set_xlabel(x_label, fontsize=16)
    ax_step_norms_ratio.set_ylabel(r"$R(t)$", fontsize=16)
    ax_step_norms_ratio.grid(True, alpha=0.3)
    ax_step_norms_ratio.set_xscale("log", base=10)
    ax_step_norms_ratio.tick_params(axis="both", labelsize=13)
    ax_step_norms_ratio.set_ylim(0, 2)

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
        0.1,
        0.1,
        (
            f"lr = {training_info.get('lr')} \n"
            f"epochs = {training_info.get('epochs')} \n"
            f"batch size = {training_info.get('batch_size')}"
        ),
        transform=loss_log.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"),
    )

    if layer == "all_weights":
        layer_label = "All Weights of The Network"
    else:
        layer_label = f"The Weights in Layer {layer}"

    fig.suptitle(
        r"$\vec{\theta}_{t}$ Describes "
        + f"{layer_label}"
        + f"\n {save_loss_frequency_for_title} SGD Update Steps are Conducted Between Subsequent Data Points"
        + r" $t_{i+1}$ and $t_{i}$",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    return fig, [cos_log, loss_log, ax_step_norms, ax_step_norms_ratio]


def add_window_highlight(
    axes: Sequence[plt.Axes],
    window_start: float,
    window_end: float,
    *,
    line_color: str,
    fill_color: str,
    fill_alpha: float,
) -> List[Any]:
    artists: List[Any] = []
    for axis in axes:
        visible_start = window_start
        visible_end = window_end
        if axis.get_xscale() == "log":
            x_min, x_max = axis.get_xlim()
            visible_start = max(window_start, x_min)
            visible_end = min(window_end, x_max)
        if visible_end <= visible_start:
            continue

        span = axis.axvspan(
            visible_start,
            visible_end,
            color=fill_color,
            alpha=fill_alpha,
            zorder=0.5,
        )
        left_line = axis.axvline(
            visible_start,
            color=line_color,
            linewidth=1.8,
            alpha=0.95,
            zorder=5,
        )
        right_line = axis.axvline(
            visible_end,
            color=line_color,
            linewidth=1.8,
            alpha=0.95,
            zorder=5,
        )
        artists.extend([span, left_line, right_line])
    return artists


def remove_artists(artists: Sequence[Any]) -> None:
    for artist in artists:
        artist.remove()


def render_figure(fig: plt.Figure, dpi: int) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    with Image.open(buffer) as image:
        rendered = image.convert("RGB")
    buffer.close()
    return rendered


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image.copy()
    scale = target_height / image.height
    target_width = max(1, int(round(image.width * scale)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def compose_frame(
    corr_image: Image.Image,
    metrics_image: Image.Image,
    gap_px: int,
    background_color: Tuple[int, int, int],
) -> Image.Image:
    target_height = max(corr_image.height, metrics_image.height)
    corr_panel = resize_to_height(corr_image, target_height)
    metrics_panel = resize_to_height(metrics_image, target_height)

    canvas = Image.new(
        "RGB",
        (corr_panel.width + gap_px + metrics_panel.width, target_height),
        background_color,
    )
    canvas.paste(corr_panel, (0, 0))
    canvas.paste(metrics_panel, (corr_panel.width + gap_px, 0))
    corr_panel.close()
    metrics_panel.close()
    return canvas


def save_gif_with_ffmpeg(
    frames_dir: Path,
    output_gif: Path,
    frame_duration_ms: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to build the GIF in a streaming fashion, but it was not found on PATH."
        )

    fps = 1000.0 / frame_duration_ms
    pattern = str(frames_dir / "frame_%06d.png")
    palette_path = frames_dir / "palette.png"

    palette_command = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        pattern,
        "-vf",
        "palettegen=stats_mode=full",
        str(palette_path),
    ]
    gif_command = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        pattern,
        "-i",
        str(palette_path),
        "-lavfi",
        "paletteuse=dither=bayer:bayer_scale=5",
        "-loop",
        "0",
        str(output_gif),
    ]

    subprocess.run(palette_command, check=True, capture_output=True, text=True)
    subprocess.run(gif_command, check=True, capture_output=True, text=True)

    if palette_path.exists():
        palette_path.unlink()


def default_output_gif(corr_dir: Path, metrics_image: Path) -> Path:
    return corr_dir / f"combined_{metrics_image.stem}.gif"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def select_frame_records(
    frame_paths: Sequence[Path],
    *,
    save_loss_frequency: int,
    snapshot_stride: int,
    delta_t: int,
    sampling_mode: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for source_frame_number, corr_frame_path in enumerate(frame_paths):
        snapshot_index = source_frame_number * snapshot_stride
        window_start = snapshot_index * save_loss_frequency
        window_end = (snapshot_index + delta_t - 1) * save_loss_frequency

        if sampling_mode == "all":
            keep = True
        elif sampling_mode == "log-decades":
            keep = keep_log_decade_frame(window_start)
        else:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

        if not keep:
            continue

        records.append(
            {
                "source_frame": source_frame_number,
                "corr_frame": corr_frame_path.name,
                "snapshot_index": snapshot_index,
                "window_start": window_start,
                "window_end": window_end,
            }
        )

    if not records:
        raise ValueError("Frame sampling removed all frames.")
    return records


def generate_combined_frames(
    corr_dir: Path,
    job_dir: Path,
    run_name: str,
    layer: str,
    output_gif: Path,
    *,
    frame_duration_ms: int,
    save_loss_frequency: int,
    snapshot_stride: int,
    delta_t: int,
    dpi: int,
    gap_px: int,
    keep_frames_dir: Path | None,
    line_color: str,
    fill_color: str,
    fill_alpha: float,
    sampling_mode: str,
) -> None:
    all_frame_paths = sorted_corr_frames(corr_dir)
    run_dir = job_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    fig, axes = build_metrics_figure(job_dir, layer, compute_flag=False)

    output_gif = output_gif.resolve()
    ensure_parent(output_gif)

    if keep_frames_dir is not None:
        frames_dir = keep_frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in frames_dir.glob("frame_*.png"):
            stale_frame.unlink()
        palette_path = frames_dir / "palette.png"
        if palette_path.exists():
            palette_path.unlink()
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="combined_corr_frames_", dir=str(output_gif.parent))
        frames_dir = Path(temp_context.name)

    highlight_artists: List[Any] = []
    selected_records = select_frame_records(
        all_frame_paths,
        save_loss_frequency=save_loss_frequency,
        snapshot_stride=snapshot_stride,
        delta_t=delta_t,
        sampling_mode=sampling_mode,
    )

    try:
        manifest_path = frames_dir / "selected_frames.json"
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(selected_records, manifest_file, indent=2)

        for output_frame_number, record in enumerate(selected_records):
            if highlight_artists:
                remove_artists(highlight_artists)
                highlight_artists = []

            window_start = int(record["window_start"])
            window_end = int(record["window_end"])
            highlight_artists = add_window_highlight(
                axes,
                window_start,
                window_end,
                line_color=line_color,
                fill_color=fill_color,
                fill_alpha=fill_alpha,
            )

            rendered_metrics = render_figure(fig, dpi=dpi)
            metrics_panel = rendered_metrics

            corr_frame_path = corr_dir / str(record["corr_frame"])
            with Image.open(corr_frame_path) as corr_frame:
                corr_panel = corr_frame.convert("RGB")
            combined = compose_frame(
                corr_panel,
                metrics_panel,
                gap_px=gap_px,
                background_color=(255, 255, 255),
            )

            frame_output_path = frames_dir / f"frame_{output_frame_number:06d}.png"
            combined.save(frame_output_path)
            combined.close()
            corr_panel.close()
            metrics_panel.close()

        if highlight_artists:
            remove_artists(highlight_artists)
        save_gif_with_ffmpeg(frames_dir, output_gif, frame_duration_ms)
    finally:
        plt.close(fig)
        if temp_context is not None:
            temp_context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined GIF from covariance frames and the corresponding "
            "weight-metrics plot with an animated training-step window."
        )
    )
    parser.add_argument(
        "--corr-dir",
        type=Path,
        required=True,
        help="Directory containing correlation frames such as corr_000_log.png.",
    )
    parser.add_argument(
        "--metrics-image",
        type=Path,
        required=True,
        help=(
            "Existing weight metrics PNG. This is used to infer the job and layer "
            "when possible and to match the intended visualization target."
        ),
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        default=None,
        help="Logs directory for the job, for example experiments/test/logs/job-13160402.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Training run directory inside the job, for example scheme-muP_widths-64x1.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer name, for example Dense_0.",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to <corr-dir>/combined_<metrics-image-stem>.gif.",
    )
    parser.add_argument(
        "--keep-frames-dir",
        type=Path,
        default=None,
        help="Optional directory for the combined PNG frames. Temporary files are used otherwise.",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=1000,
        help="Duration of each GIF frame in milliseconds.",
    )
    parser.add_argument(
        "--save-loss-frequency",
        type=int,
        default=None,
        help="Override the training-step spacing between saved metric points.",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=None,
        help="Override the covariance snapshot stride used for the correlation frames.",
    )
    parser.add_argument(
        "--delta-t",
        type=int,
        default=None,
        help="Override the covariance window width in saved metric snapshots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI used when rendering the metrics figure for each frame.",
    )
    parser.add_argument(
        "--gap-px",
        type=int,
        default=24,
        help="Horizontal gap in pixels between the correlation frame and the metrics panel.",
    )
    parser.add_argument(
        "--line-color",
        type=str,
        default="#cc5500",
        help="Color of the vertical window boundary lines.",
    )
    parser.add_argument(
        "--fill-color",
        type=str,
        default="#f6b26b",
        help="Color of the highlighted window fill.",
    )
    parser.add_argument(
        "--fill-alpha",
        type=float,
        default=0.18,
        help="Alpha value used for the highlighted window fill.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("all", "log-decades"),
        default="all",
        help=(
            "Frame selection strategy. 'log-decades' keeps the first frame and "
            "then only frames at decade landmarks such as 100, 200, ..., 1000, "
            "2000, ..., 10000, 20000, ..."
        ),
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    corr_dir = args.corr_dir.resolve()
    metrics_image = args.metrics_image.resolve()

    if not corr_dir.exists():
        raise FileNotFoundError(f"Correlation directory does not exist: {corr_dir}")
    if not metrics_image.exists():
        raise FileNotFoundError(f"Metrics image does not exist: {metrics_image}")

    job_dir = infer_job_dir(repo_root, args.job_dir, corr_dir, metrics_image)
    run_name = infer_run_name(corr_dir, args.run_name)
    layer = infer_layer(corr_dir, metrics_image, args.layer)
    run_dir = job_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist inside the job: {run_dir}")

    simulation_config_path = run_dir / "simulation_config.yaml"
    if not simulation_config_path.exists():
        raise FileNotFoundError(f"Simulation config is missing: {simulation_config_path}")

    frame_paths = sorted_corr_frames(corr_dir)
    save_loss_frequency = (
        args.save_loss_frequency
        if args.save_loss_frequency is not None
        else load_save_loss_frequency(simulation_config_path)
    )
    delta_t = infer_delta_t(run_dir, layer, args.delta_t)
    snapshot_stride = infer_snapshot_stride(run_dir, layer, len(frame_paths), args.snapshot_stride)
    output_gif = args.output_gif.resolve() if args.output_gif else default_output_gif(corr_dir, metrics_image)

    generate_combined_frames(
        corr_dir,
        job_dir,
        run_name,
        layer,
        output_gif,
        frame_duration_ms=args.frame_duration_ms,
        save_loss_frequency=save_loss_frequency,
        snapshot_stride=snapshot_stride,
        delta_t=delta_t,
        dpi=args.dpi,
        gap_px=args.gap_px,
        keep_frames_dir=args.keep_frames_dir,
        line_color=args.line_color,
        fill_color=args.fill_color,
        fill_alpha=args.fill_alpha,
        sampling_mode=args.sampling_mode,
    )

    print(f"Saved combined GIF to {output_gif}")
    print(
        "Resolved metadata: "
        f"job_dir={job_dir} run_name={run_name} layer={layer} "
        f"save_loss_frequency={save_loss_frequency} snapshot_stride={snapshot_stride} "
        f"delta_t={delta_t} sampling_mode={args.sampling_mode}"
    )


if __name__ == "__main__":
    main()
