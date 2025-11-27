# save as make_teacher_dataset.py
import os, yaml
import jax, jax.numpy as jnp
import numpy as np
import argparse
from pathlib import Path

def load_config(path):
    with open(Path("datasets").joinpath(path), "r") as f:
        return yaml.safe_load(f)

def sorted_dense_keys(d):
    return sorted(d.keys(), key=lambda k: int(k.split("_")[1]))

def init_params(input_dim, nodes_per_layer, key):
    keys = sorted_dense_keys(nodes_per_layer)
    sizes = [input_dim] + [nodes_per_layer[k] for k in keys]
    params, subkeys = {}, jax.random.split(key, num=len(keys))
    for i, k in enumerate(keys):
        fan_in, fan_out = sizes[i], sizes[i+1]
        W = jax.random.normal(subkeys[i], (fan_in, fan_out)) / jnp.sqrt(fan_in)
        b = jnp.zeros((fan_out,))
        params[k] = {"W": W, "b": b}
    return params

def forward(x, params, activations_per_layer):
    for k in sorted_dense_keys(params):
        p = params[k]
        x = x @ p["W"] + p["b"]
        act = activations_per_layer[k].lower()
        if act == "relu":
            x = jax.nn.relu(x)
        elif act == "identity":
            pass
        else:
            pass  # keep it simple: only relu/identity as per config
    return x

def make_teacher_data(cfg_path, output_file):
    name = "data.npz" if output_file is None else output_file
    cfg = load_config(cfg_path)
    net_cfg = cfg["simulation_parameters"]["network"]
    samp_cfg = cfg["simulation_parameters"]["data"]

    acts = net_cfg["activations_per_layer"]
    nodes = net_cfg["nodes_per_layer"]
    teacher_dir = samp_cfg["teacher_data_dir"]
    n_samples = int(samp_cfg["samples"])
    in_dim = int(samp_cfg["input_dim"])
    out_dim = int(samp_cfg["output_dim"])

    layer_dims = "x".join(str(nodes[k]) for k in sorted_dense_keys(nodes))
    save_dir = os.path.join(teacher_dir, layer_dims)
    os.makedirs(save_dir, exist_ok=True)

    rng_source = np.random.default_rng()
    init_seed = int(rng_source.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    key = jax.random.key(init_seed)
    params = init_params(in_dim, nodes, key)

    x_key = jax.random.split(key, 2)[1]
    X = jax.random.normal(x_key, (n_samples, in_dim))
    Y = forward(X, params, acts)

    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(Y, dtype=np.float32)
    np.savez_compressed(os.path.join(save_dir, name), inputs=X_np, outputs=Y_np)

    # Build overview (weights-only param counts to match your example)
    params_per_layer = {k: int(params[k]["W"].size) for k in sorted_dense_keys(params)}
    overview = {
        "network": {
            "num_layers_total": len(nodes),
            "num_hidden_layers": max(0, len(nodes) - 1),
            "activations_per_layer": {k: acts[k] for k in sorted_dense_keys(acts)},
            "nodes_per_layer": {k: int(nodes[k]) for k in sorted_dense_keys(nodes)},
            "params_per_layer": params_per_layer,
            "total_params": int(sum(params_per_layer.values())),
        },
        "training": {
            "data_samples": n_samples,
            "input_dimension": in_dim,
            "output_dimension": out_dim,
            "noise_std": samp_cfg["noise_std"]
        },
    }
    with open(os.path.join(save_dir, "dataset_overview.yaml"), "w") as f:
        yaml.safe_dump(overview, f, sort_keys=False)

    print(f"Saved dataset to {os.path.join(save_dir, name)}")
    print(f"Saved overview to {os.path.join(save_dir,'dataset_overview.yaml')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single MLP experiment defined in config.yaml.")
    parser.add_argument("--config", type=Path, default="teacher_config.yaml",help="Path to YAML configuration file.")
    parser.add_argument("--output_file", type=Path, default=None, help="Directory to store run outputs.")
    args = parser.parse_args()

    print(f"Using config file: {args.config} for teacher data generation.")

    make_teacher_data(args.config, args.output_file)

if __name__ == "__main__":
    main()

