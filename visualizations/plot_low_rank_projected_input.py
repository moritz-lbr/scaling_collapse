#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import plot_superposition as ps


PROJECTION_FRACTIONS = (0.05, 0.10, 0.25, 0.50)


def read_optional_snapshot(info: ps.DenseRunInfo, key: str, step: int) -> np.ndarray | None:
    array_dir = info.zarr_path / key
    if not array_dir.is_dir():
        return None
    meta = ps.load_json(array_dir / "zarr.json")
    return ps.read_snapshot(array_dir, meta, info.step_to_time_index[int(step)])


def loss_from_logits(logits: np.ndarray, targets: np.ndarray, loss_type: str) -> float:
    if loss_type == "cross_entropy":
        return ps.classification_cross_entropy(logits, targets)
    if loss_type == "mse":
        return ps.mean_squared_error(logits, targets)
    raise ValueError(f"Unsupported loss_type {loss_type!r}")


def evaluate_full_model_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense0_kernel: np.ndarray,
    dense1_kernel: np.ndarray,
    *,
    dense0_bias: np.ndarray | None,
    dense1_bias: np.ndarray | None,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> float:
    x_test = np.asarray(x_test, dtype=np.float32).reshape(x_test.shape[0], -1)
    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        hidden_pre = x_test[start:stop] @ dense0_kernel
        if dense0_bias is not None:
            hidden_pre = hidden_pre + dense0_bias
        logits = activation0(hidden_pre) @ dense1_kernel
        if dense1_bias is not None:
            logits = logits + dense1_bias
        logits = activation1(logits)
        losses.append(loss_from_logits(logits, y_test[start:stop], loss_type))
        weights.append(stop - start)
    return float(np.average(losses, weights=weights))


def input_space_modes(dense0_kernel: np.ndarray, max_modes: int) -> np.ndarray:
    hidden_weights = np.asarray(dense0_kernel.T, dtype=np.float32)
    normalized, _ = normalize_rows(hidden_weights)
    _, _, vt = ps.top_svd(normalized, max_modes)
    return vt.astype(np.float32, copy=False).T


def normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
    normalized = np.divide(
        matrix,
        norms[:, None],
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms[:, None] > 0,
    )
    return normalized, norms


def evaluate_projected_input_loss(
    x_test: np.ndarray,
    y_test: np.ndarray,
    dense0_kernel: np.ndarray,
    dense1_kernel: np.ndarray,
    v_k: np.ndarray,
    *,
    dense0_bias: np.ndarray | None,
    dense1_bias: np.ndarray | None,
    activation0: Any,
    activation1: Any,
    loss_type: str,
    batch_size: int,
) -> tuple[float, float]:
    x_test = np.asarray(x_test, dtype=np.float32).reshape(x_test.shape[0], -1)
    hidden_weights = np.asarray(dense0_kernel.T, dtype=np.float32)
    reduced_dense0 = hidden_weights @ v_k

    if x_test.shape[0] > 0:
        sample = x_test[: min(batch_size, x_test.shape[0])]
        efficient = (sample @ v_k) @ reduced_dense0.T
        explicit = sample @ (v_k @ reduced_dense0.T)
        numerical_error = float(np.max(np.abs(efficient - explicit)))
    else:
        numerical_error = 0.0

    losses = []
    weights = []
    for start in range(0, x_test.shape[0], batch_size):
        stop = min(start + batch_size, x_test.shape[0])
        z = x_test[start:stop] @ v_k
        hidden_pre = z @ reduced_dense0.T
        if dense0_bias is not None:
            hidden_pre = hidden_pre + dense0_bias
        logits = activation0(hidden_pre) @ dense1_kernel
        if dense1_bias is not None:
            logits = logits + dense1_bias
        logits = activation1(logits)
        losses.append(loss_from_logits(logits, y_test[start:stop], loss_type))
        weights.append(stop - start)
    return float(np.average(losses, weights=weights)), numerical_error


def projection_rank(width: int, input_dim: int, fraction: float) -> int:
    return max(1, min(int(round(float(fraction) * int(width))), int(input_dim)))


def analytical_parameter_estimates(
    batch_size: int,
    input_dim: int,
    width: int,
    output_dim: int,
    rank: int,
) -> dict[str, float]:
    full = float(batch_size * width * input_dim)
    projected_on_the_fly = float(batch_size * rank * (input_dim + width))
    projected_precomputed = float(batch_size * width * rank)
    total_full = full + float(batch_size * width * output_dim)
    total_projected_on_the_fly = projected_on_the_fly + float(batch_size * width * output_dim)
    total_projected_precomputed = projected_precomputed + float(batch_size * width * output_dim)
    return {
        "analytic_parameter_estimate_full": full,
        "analytic_parameter_estimate_projected_on_the_fly": projected_on_the_fly,
        "analytic_parameter_estimate_projected_precomputed": projected_precomputed,
        "analytic_parameter_ratio_on_the_fly": projected_on_the_fly / full if full > 0 else float("nan"),
        "analytic_parameter_ratio_precomputed": projected_precomputed / full if full > 0 else float("nan"),
        "analytic_total_parameter_estimate_full": total_full,
        "analytic_total_parameter_estimate_projected_on_the_fly": total_projected_on_the_fly,
        "analytic_total_parameter_estimate_projected_precomputed": total_projected_precomputed,
        "analytic_total_parameter_ratio_on_the_fly": (
            total_projected_on_the_fly / total_full if total_full > 0 else float("nan")
        ),
        "analytic_total_parameter_ratio_precomputed": (
            total_projected_precomputed / total_full if total_full > 0 else float("nan")
        ),
    }


def process_group(
    infos: list[ps.DenseRunInfo],
    *,
    fractions: tuple[float, ...],
    batch_size: int,
    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    sorted_entries = ps.sorted_series_entries([{"run": info.run_dir.name, "info": info} for info in infos])
    run_infos = [entry["info"] for entry in sorted_entries]
    if not run_infos:
        return []

    records = []
    for info in run_infos:
        x_test, y_test = ps.load_reconstruction_eval_data(info.run_dir, data_cache)
        x_test = np.asarray(x_test, dtype=np.float32).reshape(x_test.shape[0], -1)
        eval_batch = min(int(batch_size), int(x_test.shape[0]))
        output_dim = int(info.dense1_shape[-1])
        ranks = {fraction: projection_rank(info.output_dim, info.input_dim, fraction) for fraction in fractions}
        max_rank = max(ranks.values())
        parameter_cache = {
            fraction: analytical_parameter_estimates(
                eval_batch,
                info.input_dim,
                info.output_dim,
                output_dim,
                rank,
            )
            for fraction, rank in ranks.items()
        }

        print(f"Processing low-rank projected-input evaluation for {info.run_dir.name}")
        for step in sorted(info.step_to_time_index):
            dense0 = ps.read_dense0_matrix(info, step)
            dense1 = ps.read_dense1_matrix(info, step)
            dense0_bias = read_optional_snapshot(info, "Dense_0/bias", step)
            dense1_bias = read_optional_snapshot(info, "Dense_1/bias", step)
            original_loss = evaluate_full_model_loss(
                x_test,
                y_test,
                dense0,
                dense1,
                dense0_bias=dense0_bias,
                dense1_bias=dense1_bias,
                activation0=info.activation0,
                activation1=info.activation1,
                loss_type=info.loss_type,
                batch_size=batch_size,
            )
            modes = input_space_modes(dense0, max_rank)
            for fraction, rank in ranks.items():
                v_k = modes[:, :rank]
                projected_loss, reconstruction_error = evaluate_projected_input_loss(
                    x_test,
                    y_test,
                    dense0,
                    dense1,
                    v_k,
                    dense0_bias=dense0_bias,
                    dense1_bias=dense1_bias,
                    activation0=info.activation0,
                    activation1=info.activation1,
                    loss_type=info.loss_type,
                    batch_size=batch_size,
                )
                records.append(
                    {
                        "job_dir": info.job_dir,
                        "run": info.run_dir.name,
                        "step": int(step),
                        "width_N": int(info.output_dim),
                        "input_dim": int(info.input_dim),
                        "output_dim": output_dim,
                        "k": int(rank),
                        "fraction": float(fraction),
                        "test_loss_original": float(original_loss),
                        "test_loss_projected": float(projected_loss),
                        "projection_equivalence_max_abs_error": reconstruction_error,
                        **parameter_cache[fraction],
                    }
                )
    return records


def records_by_job(records: list[dict[str, Any]]) -> dict[Path, list[dict[str, Any]]]:
    groups: dict[Path, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["job_dir"], []).append(record)
    return groups


def colormap_for_records(records: list[dict[str, Any]]) -> Any:
    if not records:
        return ps.plt.cm.winter
    scheme = ps.scheme_from_run(str(records[0]["run"]))
    return ps.plt.cm.autumn if scheme == "standard" else ps.plt.cm.winter


def save_json_and_csv(out_dir: Path, records: list[dict[str, Any]]) -> None:
    serializable = []
    for record in records:
        item = dict(record)
        item.pop("job_dir", None)
        serializable.append(item)
    with (out_dir / "low_rank_projected_input_results.json").open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)

    if not serializable:
        return
    fieldnames = [
        "step",
        "run",
        "width_N",
        "input_dim",
        "output_dim",
        "k",
        "fraction",
        "test_loss_original",
        "test_loss_projected",
        "analytic_parameter_estimate_full",
        "analytic_parameter_estimate_projected_on_the_fly",
        "analytic_parameter_estimate_projected_precomputed",
        "analytic_parameter_ratio_on_the_fly",
        "analytic_parameter_ratio_precomputed",
        "analytic_total_parameter_estimate_full",
        "analytic_total_parameter_estimate_projected_on_the_fly",
        "analytic_total_parameter_estimate_projected_precomputed",
        "analytic_total_parameter_ratio_on_the_fly",
        "analytic_total_parameter_ratio_precomputed",
        "projection_equivalence_max_abs_error",
    ]
    with (out_dir / "low_rank_projected_input_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in serializable:
            writer.writerow({field: record.get(field) for field in fieldnames})


def save_largest_model_plot(
    out_dir: Path,
    records: list[dict[str, Any]],
    dpi: int,
    *,
    on_the_fly_ratio_field: str = "analytic_parameter_ratio_on_the_fly",
    precomputed_ratio_field: str = "analytic_parameter_ratio_precomputed",
    parameter_ylabel: str = "Relative parameter estimate vs full first layer",
    parameter_title: str = "Analytical relative parameter estimates",
    output_name: str = "largest_model_loss_and_parameter_estimate_low_rank_projected_input_Dense_0.png",
) -> None:
    largest_width = max(int(record["width_N"]) for record in records)
    largest_records = [record for record in records if int(record["width_N"]) == largest_width]
    if not largest_records:
        return

    fig, axes = ps.plt.subplots(3, 1, figsize=(9.0, 11.0), sharex=True, constrained_layout=True)
    loss_ax, ratio_ax, parameter_ax = axes
    steps = sorted({int(record["step"]) for record in largest_records})
    original_by_step: dict[int, float] = {}
    for record in largest_records:
        original_by_step.setdefault(int(record["step"]), float(record["test_loss_original"]))
    original_losses = np.asarray([original_by_step[step] for step in steps], dtype=float)
    loss_ax.plot(
        steps,
        original_losses,
        color="black",
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        label="full original",
    )

    fractions = sorted({float(record["fraction"]) for record in largest_records})
    colors = colormap_for_records(largest_records)(np.linspace(0, 1, max(len(fractions), 1)))
    for color, fraction in zip(colors, fractions):
        frac_records = {
            int(record["step"]): record
            for record in largest_records
            if float(record["fraction"]) == fraction
        }
        ordered = [frac_records[step] for step in steps]
        projected_losses = np.asarray(
            [float(record["test_loss_projected"]) for record in ordered],
            dtype=float,
        )
        loss_ax.plot(
            steps,
            projected_losses,
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=3.0,
            label=f"k={fraction:.0%} N",
        )
        loss_ratio = np.divide(
            projected_losses,
            original_losses,
            out=np.full_like(projected_losses, np.nan),
            where=original_losses != 0,
        )
        ratio_ax.plot(
            steps,
            loss_ratio,
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=3.0,
            label=f"k={fraction:.0%} N",
        )
        parameter_ax.plot(
            steps,
            [float(record[on_the_fly_ratio_field]) for record in ordered],
            color=color,
            linewidth=1.4,
            label=f"on-the-fly {fraction:.0%}",
        )
        parameter_ax.plot(
            steps,
            [float(record[precomputed_ratio_field]) for record in ordered],
            color=color,
            linewidth=1.4,
            linestyle="-.",
            label=f"precomputed {fraction:.0%}",
        )

    ps.set_nonnegative_loglike_xscale(loss_ax, [steps])
    ps.set_nonnegative_loglike_xscale(ratio_ax, [steps])
    ps.set_nonnegative_loglike_xscale(parameter_ax, [steps])
    loss_ax.set_ylabel("Test loss")
    loss_ax.set_title(f"Largest model low-rank projected-input loss, N={largest_width}")
    loss_ax.grid(True, which="both", alpha=0.3)
    loss_ax.legend(fontsize=8, frameon=False)
    ratio_ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ratio_ax.set_ylabel("Projected / original loss")
    ratio_ax.set_title("Projected-input loss ratio")
    ratio_ax.grid(True, which="both", alpha=0.3)
    ratio_ax.legend(fontsize=8, frameon=False, ncol=2)
    parameter_ax.set_xlabel("Training Step")
    parameter_ax.set_ylabel(parameter_ylabel)
    parameter_ax.set_title(parameter_title)
    parameter_ax.grid(True, which="both", alpha=0.3)
    parameter_ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.savefig(out_dir / output_name, dpi=dpi)
    ps.plt.close(fig)


def save_largest_projected_vs_original_widths_plot(
    out_dir: Path,
    records: list[dict[str, Any]],
    dpi: int,
) -> None:
    widths = sorted({int(record["width_N"]) for record in records})
    fractions = sorted({float(record["fraction"]) for record in records})
    if not widths or not fractions:
        return

    largest_width = max(widths)
    smaller_widths = [width for width in widths if width != largest_width]
    if not smaller_widths:
        return

    original_by_width: dict[int, dict[int, float]] = {width: {} for width in smaller_widths}
    run_by_width: dict[int, str] = {}
    projected_largest: dict[float, dict[int, float]] = {fraction: {} for fraction in fractions}
    k_by_fraction: dict[float, int] = {}

    for record in records:
        width = int(record["width_N"])
        step = int(record["step"])
        fraction = float(record["fraction"])
        if width in original_by_width:
            original_by_width[width].setdefault(step, float(record["test_loss_original"]))
            run_by_width.setdefault(width, str(record["run"]))
        if width == largest_width:
            projected_largest[fraction][step] = float(record["test_loss_projected"])
            k_by_fraction[fraction] = int(record["k"])

    ncols = min(2, len(fractions))
    nrows = int(np.ceil(len(fractions) / ncols))
    fig, axes = ps.plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 4.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    color_entries = [
        {"run": run_by_width[width]}
        for width in smaller_widths
        if width in run_by_width
    ]
    width_color_map, _ = ps.series_colors(color_entries)

    for index, fraction in enumerate(fractions):
        ax = axes_flat[index]
        panel_values: list[np.ndarray] = []
        projected_steps = sorted(projected_largest[fraction])
        projected_values = np.asarray(
            [projected_largest[fraction][step] for step in projected_steps],
            dtype=float,
        )
        ax.plot(
            projected_steps,
            projected_values,
            color="black",
            linewidth=1.6,
            marker="o",
            markersize=3.2,
            label=f"projected largest, k={k_by_fraction.get(fraction, 0)} ({fraction:.0%}N)",
        )
        panel_values.append(projected_values)

        for width in smaller_widths:
            steps = sorted(original_by_width[width])
            if not steps:
                continue
            values = np.asarray([original_by_width[width][step] for step in steps], dtype=float)
            run = run_by_width.get(width, f"N={width}")
            color = width_color_map.get(run, "black")
            label = ps.label_from_run(run)
            ax.plot(
                steps,
                values,
                color=color,
                linewidth=1.2,
                marker="o",
                markersize=2.8,
                label=f"original {label}",
            )
            panel_values.append(values)

        all_steps = np.asarray(
            projected_steps + [step for width in smaller_widths for step in original_by_width[width]],
            dtype=float,
        )
        ps.set_nonnegative_loglike_xscale(ax, [all_steps])
        ps.set_padded_ylim(ax, panel_values)
        ax.set_title(f"Largest projected with k={fraction:.0%}N")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Test loss")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

    for ax in axes_flat[len(fractions):]:
        ax.set_visible(False)
    fig.suptitle(
        "Projected largest model vs independently trained smaller models\n"
        "black: low-rank projected largest model; colored: original smaller full models"
    )
    fig.savefig(
        out_dir / "largest_projected_vs_original_widths_low_rank_projected_input_Dense_0.png",
        dpi=dpi,
    )
    ps.plt.close(fig)


def save_parameter_heatmaps(
    out_dir: Path,
    records: list[dict[str, Any]],
    dpi: int,
    *,
    on_the_fly_estimate_field: str = "analytic_parameter_estimate_projected_on_the_fly",
    precomputed_estimate_field: str = "analytic_parameter_estimate_projected_precomputed",
    full_estimate_field: str = "analytic_parameter_estimate_full",
    title: str | None = None,
    colorbar_label: str = "Projected estimate / reference full estimate",
    output_name: str = "parameter_ratio_heatmaps_low_rank_projected_input_Dense_0.png",
) -> None:
    widths = sorted({int(record["width_N"]) for record in records})
    fractions = sorted({float(record["fraction"]) for record in records})
    representative: dict[tuple[int, float], dict[str, Any]] = {}
    for record in records:
        representative.setdefault((int(record["width_N"]), float(record["fraction"])), record)
    if not widths or not fractions:
        return

    fig, axes = ps.plt.subplots(
        2,
        len(fractions),
        figsize=(4.9 * len(fractions), 7.8),
        squeeze=False,
        constrained_layout=True,
    )
    variants = (
        (on_the_fly_estimate_field, "on-the-fly projection"),
        (precomputed_estimate_field, "precomputed input"),
    )
    input_dim = int(records[0]["input_dim"])
    output_dim = int(records[0]["output_dim"])
    for row, (field, label) in enumerate(variants):
        for col, fraction in enumerate(fractions):
            matrix = np.full((len(widths), len(widths)), np.nan, dtype=float)
            for i, projected_width in enumerate(widths):
                record = representative.get((projected_width, fraction))
                if record is None:
                    continue
                projected_estimate = float(record[field])
                for j, reference_width in enumerate(widths):
                    full_estimate = (
                        float(record[full_estimate_field])
                        * reference_width
                        / projected_width
                    )
                    if full_estimate > 0:
                        matrix[i, j] = projected_estimate / full_estimate
            ax = axes[row, col]
            image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma_r")
            ax.set_xticks(np.arange(len(widths)))
            ax.set_xticklabels([str(width) for width in widths], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(widths)))
            ax.set_yticklabels([str(width) for width in widths])
            ax.set_xlabel("Reference full width")
            ax.set_ylabel("Projected model width")
            ax.set_title(f"{label}, k={fraction:.0%}N")
            fig.colorbar(image, ax=ax, label=colorbar_label)
    if title is None:
        title = (
            "Analytical relative parameter estimates across widths\n"
            f"full first-layer estimate is B*N_ref*d with d={input_dim}"
        )
    fig.suptitle(title.format(input_dim=input_dim, output_dim=output_dim))
    fig.savefig(out_dir / output_name, dpi=dpi)
    ps.plt.close(fig)


def save_results(records: list[dict[str, Any]], dpi: int) -> None:
    for job_dir, entries in records_by_job(records).items():
        out_dir = job_dir / "low_rank_projected_input_Dense_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json_and_csv(out_dir, entries)
        save_largest_model_plot(out_dir, entries, dpi)
        save_largest_model_plot(
            out_dir,
            entries,
            dpi,
            on_the_fly_ratio_field="analytic_total_parameter_ratio_on_the_fly",
            precomputed_ratio_field="analytic_total_parameter_ratio_precomputed",
            parameter_ylabel="Relative total parameter estimate vs full model",
            parameter_title="Analytical relative total parameter estimates",
            output_name="largest_model_loss_and_total_parameter_estimate_low_rank_projected_input_Dense_0.png",
        )
        save_largest_projected_vs_original_widths_plot(out_dir, entries, dpi)
        save_parameter_heatmaps(out_dir, entries, dpi)
        save_parameter_heatmaps(
            out_dir,
            entries,
            dpi,
            on_the_fly_estimate_field="analytic_total_parameter_estimate_projected_on_the_fly",
            precomputed_estimate_field="analytic_total_parameter_estimate_projected_precomputed",
            full_estimate_field="analytic_total_parameter_estimate_full",
            title=(
                "Analytical relative total parameter estimates across widths\n"
                "full estimate is B*N_ref*(d + out_dim) with d={input_dim}, out_dim={output_dim}"
            ),
            output_name="total_parameter_ratio_heatmaps_low_rank_projected_input_Dense_0.png",
        )
        print(f"Saved low-rank projected-input results to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dense_0 low-rank projected-input models and compare test loss "
            "with analytical relative parameter estimates."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures_superposition"))
    parser.add_argument("--min-step", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(PROJECTION_FRACTIONS),
        help="Projection fractions used as k=max(1, round(frac*N)), clipped to d.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    fractions = tuple(float(value) for value in args.fractions)
    if not fractions or any(value <= 0 for value in fractions):
        raise ValueError("--fractions must contain positive values")

    stores = ps.zarr_paths(args.input_dir.resolve())
    if not stores:
        raise FileNotFoundError(f"No weights.zarr directories found under {args.input_dir}")

    groups: dict[Path, list[ps.DenseRunInfo]] = {}
    for zarr_path in stores:
        info = ps.dense_run_info_from_store(zarr_path, args.output_dir, args.min_step)
        if info is None:
            continue
        groups.setdefault(info.job_dir, []).append(info)

    data_cache: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]] = {}
    all_records: list[dict[str, Any]] = []
    for infos in groups.values():
        all_records.extend(
            process_group(
                infos,
                fractions=fractions,
                batch_size=args.batch_size,
                data_cache=data_cache,
            )
        )
    save_results(all_records, args.dpi)


if __name__ == "__main__":
    main()
