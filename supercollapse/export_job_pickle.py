from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

IGNORED_DIR_NAMES = {"pareto_frontier"}
IGNORED_DIR_PREFIXES = ("slurm_output",)


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in IGNORED_DIR_NAMES:
        return False
    if any(path.name.startswith(prefix) for prefix in IGNORED_DIR_PREFIXES):
        return False
    return (path / "simulation_config.yaml").is_file() and (path / "training_log.json").is_file()


def iter_run_dirs(job_dir: Path) -> Iterable[Path]:
    for path in sorted(job_dir.iterdir()):
        if is_run_dir(path):
            yield path


def get_hidden_width(nodes_per_layer: dict[str, Any]) -> int:
    if "Dense_0" in nodes_per_layer:
        return int(nodes_per_layer["Dense_0"])
    if not nodes_per_layer:
        raise ValueError("nodes_per_layer is missing from simulation_config.yaml")
    first_key = sorted(nodes_per_layer)[0]
    return int(nodes_per_layer[first_key])


def load_run_row(run_dir: Path, c_opt, params, schedule: str, seed: int, outer_test_loss: float) -> dict[str, Any]:
    config_path = run_dir / "simulation_config.yaml"
    log_path = run_dir / "training_log.json"

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with log_path.open("r", encoding="utf-8") as handle:
        log = json.load(handle)

    network = cfg.get("network", {})
    training = cfg.get("training", {})
    data_cfg = training.get("training_data", {})
    history_dict = log.get("final_metrics", {}).get("history", {})

    train_loss = np.asarray(history_dict.get("train_loss", []), dtype=float)
    test_loss = np.asarray(history_dict.get("test_loss", []), dtype=float)
    if train_loss.size == 0 or test_loss.size == 0:
        raise ValueError(f"Missing train/test loss history in {log_path}")
    if train_loss.size != test_loss.size:
        raise ValueError(f"Mismatched train/test history lengths in {log_path}")

    total_params = int(network["total_params"])
    batch_size = int(training["batch_size"])
    save_loss_frequency = training["save_loss_frequency"]
    if not isinstance(save_loss_frequency, int):
        raise ValueError(
            f"Expected integer save_loss_frequency in {config_path}, got {save_loss_frequency!r}"
        )

    step = np.arange(test_loss.size, dtype=float)
    compute = step * float(batch_size * save_loss_frequency * total_params)
    lr = np.full(test_loss.shape, float(training["lr"]), dtype=float)

    idx = np.where(np.array(params) == total_params)
    print(params, total_params, idx, idx[0][0])
    c = c_opt[idx[0][0]]

    print(run_dir)
    print(c)

    history = pd.DataFrame(
        {
            "step": step[compute < c],
            "compute": compute[compute < c],
            "train_loss": train_loss[compute < c],
            "test_loss": test_loss[compute < c],
            "lr": lr[compute < c],
        }
    )

    # history = pd.DataFrame(
    #     {
    #         "step": step,
    #         "compute": compute,
    #         "train_loss": train_loss,
    #         "test_loss": test_loss,
    #         "lr": lr,
    #     }
    # )

    nodes_per_layer = network.get("nodes_per_layer", {})
    return {
        "history": history,
        "num_params": total_params,
        "N": int(network.get("num_hidden_layers", len(nodes_per_layer))),
        "V": int(data_cfg.get("input_dimension", -1)),
        "L": int(data_cfg.get("output_dimension", -1)),
        "alpha": -1,
        "beta": -1,
        "D": get_hidden_width(nodes_per_layer),
        "B": batch_size,
        "lr": float(training["lr"]),
        "P": total_params,
        "schedule": schedule,
        "decay_frac": 1,
        "test_loss": outer_test_loss,
        "seed": seed,
        "opt_C": float(compute[-1]),
        "opt_L": float(test_loss[-1]),
        "color": np.nan,
    }


def add_colors(df: pd.DataFrame, color_max: float) -> pd.DataFrame:
    df = df.sort_values("P").reset_index(drop=True)
    if df.empty:
        return df
    if len(df) == 1:
        df["color"] = 0.0
        return df

    pmin = float(df["P"].min())
    pmax = float(df["P"].max())
    colors = color_max * (np.log(df["P"]) - math.log(pmin)) / (math.log(pmax) - math.log(pmin))
    df["color"] = np.clip(colors, 0.0, color_max)
    return df


def build_dataframe(job_dir: Path, schedule: str, seed: int, outer_test_loss: float, color_max: float) -> pd.DataFrame:
    comp_opt_path = job_dir / "pareto_frontier" / "compute_optimal_points.json"
    with comp_opt_path.open("r", encoding="utf-8") as file:
        compute_opt_points = json.load(file)
        l_opt, c_opt, params = compute_opt_points.get("min_loss"), compute_opt_points.get("opt_compute"), compute_opt_points.get("parameters")

    # rows = []
    # for run_dir in iter_run_dirs(job_dir):
    #     rows.append(load_run_row(run_dir, c_opt, params, schedule=schedule, seed=seed, outer_test_loss=outer_test_loss))        

    rows = [
        load_run_row(run_dir, c_opt, params, schedule=schedule, seed=seed, outer_test_loss=outer_test_loss)
        for run_dir, c in zip(iter_run_dirs(job_dir), c_opt)
    ]
    if not rows:
        raise ValueError(
            f"No run directories with training_log.json and simulation_config.yaml were found under {job_dir}"
        )

    columns = [
        "history",
        "num_params",
        "N",
        "V",
        "L",
        "alpha",
        "beta",
        "D",
        "B",
        "lr",
        "P",
        "schedule",
        "decay_frac",
        "test_loss",
        "seed",
        "opt_C",
        "opt_L",
        "color",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return add_colors(df, color_max=color_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a scaling_collapse job directory into a supercollapse-style "
            "pickle dataframe with one row per training run."
        )
    )
    parser.add_argument("job_dir", type=Path, help="Path to a job directory like job-13274968.")
    parser.add_argument("output_path", type=Path, help="Path of the output .pkl file.")
    parser.add_argument(
        "--schedule",
        default="const",
        help="Value written to the outer 'schedule' column. Default: const.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Value written to the outer 'seed' column for every run. Default: 0.",
    )
    parser.add_argument(
        "--outer-test-loss",
        type=float,
        default=-1.0,
        help="Value written to the outer 'test_loss' column. Default: -1.",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=0.9,
        help="Maximum value used for the log-scaled color column. Default: 0.9.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job_dir = args.job_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not job_dir.is_dir():
        raise ValueError(f"Job directory does not exist: {job_dir}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --force to overwrite it.")

    df = build_dataframe(
        job_dir=job_dir,
        schedule=args.schedule,
        seed=args.seed,
        outer_test_loss=args.outer_test_loss,
        color_max=args.color_max,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(df, handle)

    print(f"Wrote {len(df)} runs to {output_path}")
    print(f"Columns: {', '.join(df.columns.astype(str))}")
    if not df.empty:
        first_history = df.iloc[0]["history"]
        print(f"History columns: {', '.join(first_history.columns.astype(str))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
