from pathlib import Path
import numpy as np
from typing import Dict, Any, Iterable, List
import json
import os
import yaml
from scipy.interpolate import interp1d
import pdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm



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
    if "final_metrics" in log_data:
        final_metrics = log_data.get("final_metrics")
    else: 
        final_metrics = log_data.get("avg_final_metrics")
    history = final_metrics.get("history")
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
    loss_histories = np.empty(len(log_files), dtype=object)
    training_compute = np.empty(len(log_files), dtype=object)
    model_size = np.empty(len(log_files), dtype=object)
    for i, log_file in enumerate(log_files):
        loss_histories[i] = collect_single_loss_history(log_file)
        batch_size, save_loss_frequency, total_params = get_training_info(log_file.parent / "simulation_config.yaml")
        training_compute[i] = np.arange(len(loss_histories[i]))*batch_size*save_loss_frequency*total_params
        model_size[i] = total_params
        
    idx = np.argsort(model_size)
    model_size = model_size[idx]
    loss_histories = loss_histories[idx]
    training_compute = training_compute[idx]
    return loss_histories, training_compute, model_size

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
    y_pred = 10**y_pred_log
    r2 = np.corrcoef(np.log10(y), y_pred_log)[0,1]**2

    return {
            'power_law': "opt_compute = (params/a)^b",
            'a': a,
            'b': b,
            'r2': r2,
            'fit': {
                'params': np.unique(x).tolist(),
                'opt_compute': np.unique(y_pred).tolist()
                }
        }
    
    # return a, b, r2, 10**y_pred_log

def fit_loss_compute_power_law(C, L, num_inits=10, L0=None):
    def power_law_const(params, x):
        if L0 is None:
            a, b, L0_fit = params
            return a * x**(-b) + L0_fit
        else:
            a, b = params
            return a * x**(-b) + L0

    def huber_loss(residual, delta=1e-3):
        mask = np.abs(residual) <= delta
        return np.where(mask, 
                       0.5 * residual**2,
                       delta * (np.abs(residual) - 0.5 * delta))

    def objective(params):
        pred = power_law_const(params, C)
        residuals = np.log(pred) - np.log(L)
        return np.mean(huber_loss(residuals))
        
     # Try all combinations of parameter initializations
    best_loss = np.inf
    best_params = None
    
    # Parameter ranges for initialization
    # a_range = [1e4, 1e5]
    # b_range = [0.1, 1]
    
    # # Create evenly spaced initializations for each parameter
    # a_inits = np.logspace(np.log10(a_range[0]), np.log10(a_range[1]), num_inits)
    # b_inits = np.linspace(b_range[0], b_range[1], num_inits)

    # a_inits = np.logspace(7, 9, 10)
    # b_inits = np.linspace(0.2, 2.2, 10)

    # L0_range = [0.0, 0.013]
    # L0_inits = np.linspace(0.0, 0.013, 10)

    a_inits = np.logspace(-1, 0, 10)
    b_inits = np.linspace(0.01, 0.1, 10)

    L0_range = [0.01, 0.15]
    L0_inits = np.linspace(0.01, 0.15, 10)

    
    if L0 is None:
        L0_range = [min(L)*0.1, min(L)]
        L0_inits = np.linspace(L0_range[0], L0_range[1], num_inits)
        # Try all combinations of a, b, and L0
        for a, b, L0_init in tqdm(product(a_inits, b_inits, L0_inits)):
            init_params = [a, b, L0_init]
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                bounds=[(0, None), (0, None), (0, None)]  # Changed bounds to >= 0
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
    else:
        # Try all combinations of a and b
        for a, b in product(a_inits, b_inits):
            init_params = [a, b]
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                bounds=[(0, None), (0, None)]  # Changed bounds to >= 0
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
            
    # Compute R^2 score
    y_pred = power_law_const(best_params, C)
    log_L = np.log(L)
    log_pred = np.log(y_pred)
    ss_res = np.sum((log_L - log_pred) ** 2)
    ss_tot = np.sum((log_L - np.mean(log_L)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    if L0 is None:
        return {
            'power_law': "min_loss = a * C^(-b) + L0",
            'a': best_params[0],
            'b': best_params[1],
            'L0': best_params[2],
            'r2': r2,
            'fit': {
                'compute': C.tolist(),
                'min_loss': y_pred.tolist()
            }
        }
    else:
        return {
            'power_law': "min_loss = a * C^(-b) + L0",
            'a': best_params[0],
            'b': best_params[1],
            'L0': L0,
            'r2': r2,
            'fit': {
                'compute': C.tolist(),
                'min_loss': y_pred.tolist()
            }
        }

def get_curve_optimal_points(parameters, comp_param_fit, losses, compute_costs):
    opt_compute = (np.asarray(parameters, dtype=float) / comp_param_fit["a"]) ** comp_param_fit["b"]
    opt_loss = []
    for loss, compute, c_opt in zip(losses, compute_costs, opt_compute):
        interp = interp1d(
            np.log10(compute),
            np.log10(loss),
            bounds_error=False,
            fill_value=np.inf,
        )
        opt_loss.append(10**interp(np.log10(c_opt)))
    return np.asarray(opt_loss, dtype=float), np.asarray(opt_compute, dtype=float)

def save_pareto_frontier_samples(directory, L_min, P_opt, C_opt):
    data = {
        "L_min": L_min.tolist(),
        "P_opt": P_opt.tolist(),
        "C_opt": C_opt.tolist(),
    }
    
    file_path = directory / "pareto_frontier" / "frontier_samples.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def save_pareto_frontier_fits(directory, comp_param_fit, loss_comp_fit):
    file_path_c_p = directory / "pareto_frontier" / "compute_params_fit.json"
    file_path_l_c = directory / "pareto_frontier" / "loss_compute_fit.json"
    os.makedirs(os.path.dirname(file_path_c_p), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_l_c), exist_ok=True)
    with open(file_path_c_p, 'w', encoding='utf-8') as file:
        json.dump(comp_param_fit, file, indent=4)
    with open(file_path_l_c, 'w', encoding='utf-8') as file:
        json.dump(loss_comp_fit, file, indent=4)

def save_compute_optimal_points(directory, parameters, comp_param_fit, losses, compute_costs):
    min_loss, opt_compute = get_curve_optimal_points(parameters, comp_param_fit, losses, compute_costs)

    compute_optimal_points = {
        "parameters": np.asarray(parameters).tolist(),
        "opt_compute": opt_compute.tolist(),
        "min_loss": min_loss.tolist()
    }

    file_path = directory / "pareto_frontier" / "compute_optimal_points.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(compute_optimal_points, file, indent=4)

def rescale_loss_curves(file_path, compute_costs, losses, L0):
    comp_opt_path = file_path / "pareto_frontier" / "compute_optimal_points.json"
    with comp_opt_path.open("r", encoding="utf-8") as file:
        compute_opt_points = json.load(file)
        l_opt, c_opt = compute_opt_points.get("min_loss"), compute_opt_points.get("opt_compute")

    scaled_compute = []
    scaled_losses = []
    for loss, compute, curve_l_opt, curve_c_opt in zip(losses, compute_costs, l_opt, c_opt):
        print(f"{compute[1]:.2e}, {curve_c_opt:.2e}")
        scaled_c = compute / curve_c_opt
        scaled_l = (loss - L0) / (curve_l_opt - L0)
        scaled_compute.append(scaled_c.tolist())
        scaled_losses.append(scaled_l.tolist())

    rescaled_curves = {
        "scaled_compute": scaled_compute,
        "scaled_losses": scaled_losses
    }

    rescaled_path = file_path / "pareto_frontier" / "rescaled_loss_curves.json"
    with open(rescaled_path, 'w', encoding='utf-8') as file:
        json.dump(rescaled_curves, file, indent=4)

def plot_param_compute_pareto_fit(file_path, parameters, compute_costs, params, compute_pred):
    fig, ax = plt.subplots()
    ax.scatter(parameters.flatten(), compute_costs.flatten(), label="Optimal Frontier Samples", color='blue')
    ax.plot(params, compute_pred, color='red', label="Fitted Power Law", linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Model Size (Number of Parameters)')
    plt.ylabel('Compute Cost (FLOPs)')
    plt.title('Loss vs Compute Cost for Different Model Sizes')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path / "pareto_frontier" / "frontier_fit.png")

def plot_rescaled_loss_curves(file_path):
    path = file_path / "pareto_frontier" / "rescaled_loss_curves.json"

    with path.open("r", encoding="utf-8") as file:
        rescaled_curevs = json.load(file)
        losses, computes = rescaled_curevs["scaled_losses"], rescaled_curevs["scaled_compute"]


    def _prepare_color_map(count: int, cmap: Any) -> Iterable[Any]:
        steps = max(count, 1)
        return cmap(np.linspace(0, 1, steps))

    cmap = _prepare_color_map(len(losses), plt.cm.winter)

    fig, ax = plt.subplots()
    for i, (loss, compute) in enumerate(zip(losses, computes)):
        compute = np.array(compute)
        print(len(compute[compute <= 1.0]))
        loss = np.array(loss)
        compute = np.array(compute)
        ymax = interp1d(compute, loss, bounds_error=False, fill_value=np.inf)(0.1)
        ax.plot(compute, loss, color=cmap[i], label="Optimal Points")
        ax.set_ylim(1.0, ymax)
        # ax.set_xscale("log")
        plt.xlabel("Normalized Compute", fontsize=30)
        plt.ylabel("Normalized Loss", fontsize=30)
        plt.xlim(0.0, 1.0)
        plt.tight_layout()
        plt.grid()
        plt.savefig(file_path / "pareto_frontier" / "scaled_loss_curves.png")



def main(log_path, c_min, c_max, n_points):
    losses, compute, params = collect_losses_and_compute(Path(log_path))
    l_min, p_opt, c_opt = get_pareto_frontier(losses, compute, params, c_min, c_max, n_points)

    mask = (p_opt > p_opt.min()) & (p_opt < p_opt.max())
    p_opt = p_opt[mask]
    c_opt = c_opt[mask] 
    l_min = l_min[mask]

    # print(f"compute[2][:3]: {compute[2][:3]:.2e}")

    compute_params_power_law = fit_power_law(p_opt, c_opt)
    per_curve_opt_loss, per_curve_opt_compute = get_curve_optimal_points(params, compute_params_power_law, losses, compute)
    finite_mask = np.isfinite(per_curve_opt_loss) & np.isfinite(per_curve_opt_compute)
    loss_compute_power_law = fit_loss_compute_power_law(
        per_curve_opt_compute[finite_mask],
        per_curve_opt_loss[finite_mask],
        num_inits=20,
    )

    save_pareto_frontier_samples(Path(log_path), l_min, p_opt, c_opt)
    save_pareto_frontier_fits(Path(log_path), compute_params_power_law, loss_compute_power_law)
    save_compute_optimal_points(Path(log_path), params, compute_params_power_law, losses, compute)
    rescale_loss_curves(Path(log_path), compute, losses, loss_compute_power_law["L0"])
    plot_param_compute_pareto_fit(Path(log_path), p_opt, c_opt, compute_params_power_law['fit']['params'], compute_params_power_law['fit']['opt_compute'])
    plot_rescaled_loss_curves(Path(log_path))

if __name__ == "__main__":
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-13274968"
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-13294144"
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-13275993"
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-13365173"
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-combined"
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/job-13423237"
    # log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/test/logs/avg_job-13423237_job-13423464"
    log_path = "/project/theorie/m/M.Rautenberg/scaling_collapse/experiments/mup/logs/job-13656168"
    c_min = 4e10
    c_max = 1e13
    n_points = 1000
    main(log_path, c_min, c_max, n_points)
    
