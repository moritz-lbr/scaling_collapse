import argparse
from pathlib import Path
from datetime import datetime
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from main import run_experiment


def assemble(config_path):
    # Initialize Hydra manually
    root = Path(__file__).parent.parent
    name = config_path.name
    path = config_path.parent
    dir = root.joinpath(path)
    
    with initialize_config_dir(config_dir=str(dir), version_base=None):
        cfg = compose(config_name=name)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(resolved, dict):
            resolved.pop("_master", None)
    return resolved

def run_experiments(input_path: Path, output_path: Path = None) -> int:

    print("\nNote that the first entry of the weight log file contains the weights at initialization before the training has started.")
    print("\nNote that the total number of training steps displayed in the progress bar refers to the number of training steps that are saved during training.")
    print("The actual total number of training steps conducted in this training process can be obtained by multiplying the total number of steps displayed in the\n" \
    "progress bar with the save_loss_frequency assigned in the training config, as only every save_loss_frequency step losses and weights are saved.")
    print(f"\nRunning training for {input_path.name}") 
    
    if str(input_path).endswith(".yaml"):
        if not output_path:
            output_dir = Path("./logs") 
        else:
            output_dir = output_path
        config = assemble(input_path)

    else:
        configs = input_path.joinpath("configs")
        config_files = sorted(configs.glob("*.yaml"))
        config_files = [f for f in config_files if f.name != "master_config.yaml"]

        if output_path:
            output_dir = output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = input_path.joinpath(f"logs/run_{timestamp}")

        for path in config_files:
            config = assemble(path)
    
    run_experiment(config, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Parser to run experiments from command line.")

    parser.add_argument("--path", type=Path, help="Path for the experiment config files", required=True)
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to store run outputs.")

    args = parser.parse_args()

    return run_experiments(args.path, args.output_dir)

if __name__ == "__main__":
    main()