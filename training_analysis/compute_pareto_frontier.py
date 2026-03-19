from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import json
import os
import yaml
from scipy.interpolate import interp1d
import pdb
import matplotlib.pyplot as plt



def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches

def load_training_log(log_path: Path) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def collect_single_loss_history(log_path: Path) -> np.ndarray:
    log_data = load_training_log(log_path)
    history = log_data.get("final_metrics").get("history")
    losses = history.get("test_loss")
    if not losses:
        raise ValueError(f"No loss history found in {log_path}")

    return np.asarray(losses, dtype=float)

def get_training_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        training_config = yaml.safe_load(file)
        network_info = training_config.get("network")
        training_info = training_config.get("training")
        batch_size = training_info.get("batch_size")
        save_loss_frequency = training_info.get("save_loss_frequency")
        total_params = network_info.get("total_params")
    return batch_size, save_loss_frequency, total_params

def collect_losses_and_compute(directory: Path) -> List[np.ndarray]:
    log_files = collect_files_with_ending(directory, "training_log.json")
    log_histories = np.empty(len(log_files), dtype=object)
    training_compute = np.empty(len(log_files), dtype=object)
    model_size = np.empty(len(log_files), dtype=object)
    for i, log_file in enumerate(log_files):
        log_histories[i] = collect_single_loss_history(log_file)
        batch_size, save_loss_frequency, total_params = get_training_info(log_file.parent / "simulation_config.yaml")
        training_compute[i] = np.arange(len(log_histories[i]))*batch_size*save_loss_frequency*total_params
        model_size[i] = total_params
    return log_histories, training_compute, model_size

def get_pareto_frontier(losses: np.ndarray, compute_costs: np.ndarray, parameters: np.ndarray, c_min: float, c_max: float, n_points: int) -> List[np.ndarray]:
    # Assuming losses are 1D arrays and we want to minimize them
    L_min = np.full(n_points, np.inf, dtype=float)
    C_opt = np.full(n_points, np.nan, dtype=float)
    P_opt = np.full(n_points, np.nan, dtype=float)

    for i, (loss, compute) in enumerate(zip(losses, compute_costs)):
        model_c_range = np.logspace(np.log10(c_min), np.log10(c_max), n_points)
        interp = interp1d(np.log10(compute), np.log10(loss), bounds_error=False, fill_value=np.inf)
        L_new = 10**interp(np.log10(model_c_range))
        mask = L_new < L_min
        L_min[mask] = L_new[mask]
        C_opt[mask] = model_c_range[mask]
        P_opt[mask] = parameters[i]

    L_min = L_min[~np.isnan(P_opt)]
    P_opt = P_opt[~np.isnan(P_opt)]
    C_opt = C_opt[~np.isnan(C_opt)]
    return L_min, P_opt, C_opt

def fit_power_law(x, y):
    coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
    b = coeffs[0]
    c = coeffs[1]
    a = 10**(-c/b)
    
    # Calculate R²
    y_pred_log = b * np.log10(x) + c
    r2 = np.corrcoef(np.log10(y), y_pred_log)[0,1]**2
    
    return a, b, r2, 10**y_pred_log

def save_pareto_frontier(directory, L_min, P_opt, C_opt):
    data = {
        "L_min": L_min.tolist(),
        "P_opt": P_opt.tolist(),
        "C_opt": C_opt.tolist()
    }
    
    file_path = directory / "pareto_frontier" / "frontier_samples.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def plot_param_compute(file_path, parameters, compute_costs, compute_pred):
    fig, ax = plt.subplots()
    ax.scatter(parameters.flatten(), compute_costs.flatten())
    ax.plot(parameters.flatten(), compute_pred, color='red', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Model Size (Number of Parameters)')
    plt.ylabel('Compute Cost (FLOPs)')
    plt.title('Loss vs Compute Cost for Different Model Sizes')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path / "pareto_frontier" / "frontier_fit.png")

def main(log_path, c_min, c_max, n_points):
    losses, compute, params = collect_losses_and_compute(Path(log_path))
    l_min, p_opt, c_opt = get_pareto_frontier(losses, compute, params, c_min, c_max, n_points)
    fit_params = fit_power_law(p_opt, c_opt)
    save_pareto_frontier(Path(log_path), l_min, p_opt, c_opt)
    plot_param_compute(Path(log_path), p_opt, c_opt, fit_params[3])


if __name__ == "__main__":
    log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-13274968"
    c_min = 1.5e8
    c_max = 3e10
    n_points = 1000
    main(log_path, c_min, c_max, n_points)
