import argparse
import json
from collections import OrderedDict
from pathlib import Path
import yaml
from tqdm import tqdm
import pynvml
# from jax import config
# config.update("jax_enable_x64", True)

from config import create_config, load_simulation_parameters
from train import run_once
from utils import format_widths, parameter_summary, tree_to_python, _sorted_nodes, collect_files_with_ending, load_yaml_as_dict


def run_experiment(config_path: Path, output_dir: Path) -> None:
    # Load Config that 
    sim_config = load_simulation_parameters(config_path)
    cfg = create_config(sim_config)

    # Collect data from training configs and create config class
    data_config_path = collect_files_with_ending(cfg.task, "dataset_overview.yaml")[0]
    data_config = load_yaml_as_dict(data_config_path)["training"]

    widths = cfg.widths
    cfg.num_input_features, cfg.num_output_features = data_config["input_dimension"], data_config["output_dimension"]

    run_name = f"scheme-{cfg.param_scheme}_widths-{format_widths(widths)}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pbar = None

    def update_progress(*, epoch: int, total: int, train_loss: float, test_loss: float) -> None:
        nonlocal pbar
        if pbar is None:
            # Create the bar only when the first metric is available so the initial render
            # reflects step 1/total instead of an empty 0/1 placeholder.
            pbar = tqdm(
                total=total,
                desc=f"{run_name} training",
                unit="step",
                leave=False,
                initial=epoch,
            )
        pbar.total = total
        pbar.n = epoch
        pbar.set_postfix(train=f"{train_loss:.4f}", test=f"{test_loss:.4f}", refresh=False)
        pbar.refresh()

    try:
        result = run_once(cfg, run_dir, progress=update_progress)
    finally:
        if pbar is not None:
            pbar.close()

    dataset_info = result.get("dataset_info")
    dataset_info["input_dimension"] = cfg.num_input_features
    dataset_info["output_dimension"] = cfg.num_output_features
    dataset_info["task"] = cfg.task
    per_layer_counts, total_params = parameter_summary(result["final_params"])
    num_hidden_layers = len(widths)

    # ----------------------------
    # Logs and Config_summary 
    # ----------------------------

    training_log = {
        "final_metrics": {
            "final_train_loss": result["final_train_loss"],
            "final_test_loss": result["final_test_loss"],
            "history": result["history"],
        },
        "final_params": tree_to_python(result["final_params"]),
    }
    

    with (run_dir / "training_log.json").open("w") as f:
        json.dump(training_log, f, indent=2)


    network_section = OrderedDict([
        ("num_hidden_layers", num_hidden_layers),
        ("activations_per_layer", cfg.activations_per_layer),
        ("nodes_per_layer", dict(_sorted_nodes(cfg.nodes_per_layer))),
        ("params_per_layer", per_layer_counts),
        ("total_params", int(total_params)),
        ("param_scheme", cfg.param_scheme),
    ])

    training_section = OrderedDict([
        ("lr", cfg.lr),
        ("wd", cfg.wd),
        ("epochs", cfg.epochs),
        ("batch_size", cfg.batch_size),
        ("save_loss_frequency", cfg.save_loss_frequency),
        ("training_data", dataset_info),
    ])

    simulation_config = {
        "network": dict(network_section),
        "training": dict(training_section),
    }

    with (run_dir / "simulation_config.yaml").open("w") as f:
        yaml.safe_dump(simulation_config, f, sort_keys=False)

    print(f"→ results saved to {run_dir}")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    name = pynvml.nvmlDeviceGetName(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

    print("Device name:", name)
    print("Compute capability:", cap)
    print("Total memory (GB):", mem.total / 1e9)
    print("Free memory  (GB):", mem.free / 1e9)
    print("Used memory  (GB):", mem.used / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single MLP experiment defined in config.yaml.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                        help="Path to YAML configuration file.")
    parser.add_argument("--output_dir", type=Path, default=Path("logs"),
                        help="Directory to store run outputs.")
    args = parser.parse_args()

    run_experiment(args.config, args.output_dir)

if __name__ == "__main__":
    main()
