import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    cfg_path = path
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def sample_power_law(num_terms: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    if exponent >= -1:
        raise ValueError("Power-law exponent must be less than -1 for a proper tail.")
    alpha = -(exponent + 1.0)  # exponent=-2 -> alpha=1 pareto tail
    return rng.pareto(alpha, size=num_terms) + 1.0


def random_unit_vectors(num_terms: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    vecs = rng.normal(size=(num_terms, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def sample_fourier_parameters(
    num_terms: int, input_dim: int, exponent: float, bias_bool: bool, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = sample_power_law(num_terms, exponent, rng)
    dirs = random_unit_vectors(num_terms, input_dim, rng)
    modes = np.rint(scales[:, None] * dirs).astype(np.int64)
    weights = rng.normal(loc=0.0, scale=1.0, size=num_terms)
    if bias_bool:
        biases = rng.binomial(1, 0.5, size=num_terms)
    else:
        biases = np.array(0.0, ndmin=num_terms)
    return weights, biases, modes


def evaluate_target(x: np.ndarray, weights: np.ndarray, biases: np.ndarray, modes: np.ndarray, non_linearity: str) -> np.ndarray:
    phase = (x @ modes.T) + biases
    if non_linearity == "identity":
        terms = phase
    elif non_linearity == "x**2":
        terms = phase**2
    elif non_linearity == "x**3":
        terms = phase**3
    elif non_linearity == "cos":
        terms = np.cos(phase)
    else: 
        raise ValueError("The non_linearity selected in the regression_config.yaml could not be found." \
        " Available non-linearities are: [identity, x**2, cos].")
    return (terms @ weights)


def make_regression_data(cfg_path: Path, output_file: Path | None) -> None:
    name = Path("data.npz") if output_file is None else Path(output_file)
    cfg = load_config(cfg_path)
    sim_cfg = cfg.get("simulation_parameters", {})
    data_cfg = sim_cfg.get("data", cfg.get("data", {}))
    if not data_cfg:
        raise ValueError("Config must contain a 'data' section with generation parameters.")

    data_dir = Path(data_cfg["regression_data_dir"])
    n_samples = int(data_cfg["samples"])
    input_dim = int(data_cfg.get("input_dim"))
    output_dim = int(data_cfg.get("output_dim"))
    non_linearity = str(data_cfg.get("non_linearity", "identity"))
    if output_dim != 1:
        raise ValueError("This regression generator produces scalar outputs; set output_dim to 1.")
    num_terms = int(data_cfg.get("num_terms"))
    power_exp = float(data_cfg.get("power_law_exponent"))
    noise_std = data_cfg.get("noise_std")
    bias_bool = bool(data_cfg.get("bias"))
    rng_seed = data_cfg.get("seed")

    save_dir = data_dir

    if save_dir.exists():
        base = save_dir
        for i in range(100):
            candidate = Path(f"{base}_{i:02d}")
            if not candidate.exists():
                save_dir = candidate
                break

    save_dir.mkdir(parents=True, exist_ok=False)
            
    if rng_seed:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()

    weights, biases, modes = sample_fourier_parameters(num_terms, input_dim, power_exp, bias_bool, rng)
    inputs = rng.uniform(-0.5, 0.5, size=(n_samples, input_dim)).astype(np.float32)
    outputs = evaluate_target(inputs, weights, biases, modes, non_linearity).astype(np.float32).reshape(n_samples, output_dim)
    if noise_std:
        outputs = outputs + np.random.normal(loc=0, scale=noise_std, size=outputs.shape)

    np.savez_compressed(save_dir / name, inputs=inputs, outputs=outputs)

    training_overview = {
        "data_samples": n_samples,
        "input_dimension": input_dim,
        "output_dimension": output_dim,
        "target_function": {
            "num_terms": num_terms,
            "power_law_exponent": power_exp,
            "non_linearity": non_linearity,
            "sample_domain": f"[-0.5, 0.5]^{input_dim}",
            "bias": bias_bool,
        },
        "noise_std": noise_std,
    }

    overview = {"training": training_overview}
    with (save_dir / "dataset_overview.yaml").open("w") as f:
        yaml.safe_dump(overview, f, sort_keys=False)

    print(f"Saved dataset to {save_dir / name}")
    print(f"Saved overview to {save_dir / 'dataset_overview.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate regression data with Fourier-based targets.")
    parser.add_argument(
        "--config",
        type=Path,
        default="datasets/regression_config.yaml",
        help="Path to YAML configuration file relative to the datasets/ directory.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Optional filename for the compressed dataset (defaults to data.npz).",
    )
    args = parser.parse_args()

    print(f"Using config file: {args.config} for regression data generation.")
    make_regression_data(args.config, args.output_file)


if __name__ == "__main__":
    main()
