import argparse
import json
from collections import OrderedDict
from pathlib import Path
import yaml
from tqdm import tqdm

from config import create_config, load_simulation_parameters
from train import run_once
from utils import format_widths, parameter_summary, tree_to_python, _sorted_nodes, collect_files_with_ending, load_yaml_as_dict


def run_experiment(config_path: Path, output_dir: Path) -> None:
    # Load Config that 
    sim_config, _ = load_simulation_parameters(config_path)
    cfg = create_config(sim_config)

    # Collect data from training configs and create config class
    data_config_dir = Path("datasets").joinpath(cfg.task)
    print(data_config_dir)
    data_config_path = collect_files_with_ending(data_config_dir, "dataset_overview.yaml")[0]
    data_config = load_yaml_as_dict(data_config_path)["training"]

    widths = cfg.widths
    # print(data_config["input_shape"], type(data_config["input_shape"]))
    # Fix the issue with data_config["input_shape"]!!!
    cfg.num_input_features, cfg.num_output_features = data_config["input_shape"][0], data_config["output_shape"][0]

    run_name = f"scheme-{cfg.param_scheme}_widths-{format_widths(widths)}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=cfg.epochs, desc=f"{run_name} training", unit="epoch", leave=False)

    def update_progress(*, epoch: int, total: int, train_loss: float, test_loss: float) -> None:
        pbar.total = total
        pbar.n = epoch
        pbar.set_postfix(train=f"{train_loss:.4f}", test=f"{test_loss:.4f}")
        pbar.refresh()
        tqdm.write(f"Epoch {epoch}/{total} • train_loss={train_loss:.6f} • test_loss={test_loss:.6f}")

    try:
        result = run_once(cfg, progress=update_progress)
    finally:
        pbar.close()

    dataset_info = result.get("dataset_info", {})
    per_layer_counts, total_params = parameter_summary(result["final_params"])
    num_hidden_layers = len(widths)
    num_layers_total = num_hidden_layers + 1


    # ----------------------------
    # Logs and Config_summary 
    # ----------------------------

    training_log = {
        "dataset_info": {
            "n_train": dataset_info.get("n_train"),
            "n_test": dataset_info.get("n_test"),
            "total": dataset_info.get("total"),
            "test_split": dataset_info.get("test_split"),
            "save_loss_frequency": dataset_info.get("save_loss_frequency"),
        },
        "parameter_summary": {
            "num_hidden_layers": num_hidden_layers,
            "num_layers_total": num_layers_total,
            "nodes_per_layer": dict(_sorted_nodes(cfg.nodes_per_layer)),
            "params_per_layer": per_layer_counts,
            "total_params": int(total_params),
        },
        "final_metrics": {
            "train_loss": result["final_train_loss"],
            "test_loss": result["final_test_loss"],
            "history": result["history"],
        },
        "final_params": tree_to_python(result["final_params"]),
    }

    

    with (run_dir / "training_log.json").open("w") as f:
        json.dump(training_log, f, indent=2)

    


    network_section = OrderedDict([
        ("num_layers_total", num_layers_total),
        ("num_hidden_layers", num_hidden_layers),
        ("activations_per_layer", cfg.activations_per_layer),
        ("nodes_per_layer", dict(_sorted_nodes(cfg.nodes_per_layer))),
        ("params_per_layer", per_layer_counts),
        ("total_params", int(total_params)),
        ("param_scheme", cfg.param_scheme),
    ])

    training_section = OrderedDict([
        ("lr", cfg.lr),
        ("task", cfg.task),
        ("test_split", cfg.test_split),
        ("batch_size", cfg.batch_size),
        ("epochs", cfg.epochs),
        ("save_loss_frequency", cfg.save_loss_frequency),
        ("training_data", data_config),
    ])

    simulation_config = {
        "network": dict(network_section),
        "training": dict(training_section),
    }

    with (run_dir / "simulation_config.yaml").open("w") as f:
        yaml.safe_dump(simulation_config, f, sort_keys=False)

    print(f"→ results saved to {run_dir}")


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
