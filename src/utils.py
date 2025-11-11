# utils.py
from typing import Dict, Any, Tuple, List
from collections import OrderedDict
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import yaml
import os 

from mlp import MLP
from param_schemes import SCHEMES # pyright: ignore[reportMissingImports]


# ----------------------------
# Formatting
# ----------------------------

def load_yaml_as_dict(file_path):
    """Load YAML data from a file and return it as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    # Ensure it returns a dictionary (use empty dict if None or not a dict)
    if not isinstance(data, dict):
        data = {} if data is None else {"value": data}
    
    return data

def extract_after_char(s, start, stop):
    return s[s.rfind(start)+1:s.rfind(stop)]


def format_widths(widths: Tuple[int, ...]) -> str:
    return "x".join(str(int(w)) for w in widths)


def parameter_summary(params: Dict[str, Any]) -> Tuple[Dict[str, int], int]:
    per_layer: Dict[str, int] = {}
    total = 0
    for layer_name, layer_params in params.items():
        leaves = jax.tree_util.tree_leaves(layer_params)
        layer_total = 0
        for leaf in leaves:
            arr = np.asarray(leaf)
            layer_total += int(arr.size)
        per_layer[layer_name] = layer_total
        total += layer_total
    return per_layer, total

def tree_to_python(tree: Any) -> Any:
    """Convert a pytree of arrays into nested Python lists/numbers for JSON dumping."""
    def convert(x):
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        if isinstance(x, np.generic):
            return x.item()
        if hasattr(x, "tolist"):
            return x.tolist()
        return x
    return jax.tree_util.tree_map(convert, tree)


def _sorted_nodes(nodes_map: Dict[str, int]) -> OrderedDict:
    return OrderedDict(sorted(((k, int(v)) for k, v in nodes_map.items()), key=lambda kv: int(kv[0].split("_")[1])))

# ----------------------------
# Data
# ----------------------------

def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches

def load_teacher_dataset(cfg) -> Tuple[np.ndarray, np.ndarray]:
    base_dir = Path("datasets").joinpath(cfg.task)

    # expected_folder = f"scheme-{cfg.param_scheme}_widths-{format_widths(cfg.widths)}"
    if base_dir.is_file():
        data_path = base_dir
    else:
        data_path = base_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Teacher dataset not found at {data_path}. Generate it with generate_teacher_data.py."
        )
    with np.load(data_path) as data:
        if "inputs" in data and "outputs" in data:
            inputs = data["inputs"].astype(np.float32)
            outputs = data["outputs"].astype(np.float32)
        elif all(key in data for key in ("x_train", "y_train", "x_test", "y_test")):
            x_train = data["x_train"].astype(np.float32)
            y_train = data["y_train"].astype(np.float32)
            x_test = data["x_test"].astype(np.float32)
            y_test = data["y_test"].astype(np.float32)
            inputs = np.concatenate([x_train, x_test], axis=0)
            outputs = np.concatenate([y_train, y_test], axis=0)
        else:
            raise ValueError(
                "Unsupported teacher dataset format. Expected 'inputs'/'outputs' arrays "
                "or legacy 'x_train'/'y_train'/'x_test'/'y_test'."
            )
    return inputs, outputs

def make_regression_data(n_train=2048, n_test=1024, phase=0.0, target="sin_mix"):
    rng = np.random.default_rng()
    x_tr = rng.uniform(-np.pi, np.pi, size=(n_train, 1)).astype(np.float32)
    x_te = rng.uniform(np.pi, 2*np.pi, size=(n_test, 1)).astype(np.float32)

    def f(x):
        if target == "sin":
            return np.sin(x + phase)
        elif target == "sin_mix":
            return np.sin(x) + 0.3*np.sin(3*x + phase)
        elif target == "sin_sin":
            return np.sin(np.sin(x))
        elif target == "cos_sin":
            return np.cos(2*x) * np.sin(0.5*x)
        else:
            raise ValueError(f"unknown target: {target}")

    y_tr = f(x_tr).astype(np.float32)
    y_te = f(x_te).astype(np.float32)
    return (x_tr, y_tr), (x_te, y_te)

# ----------------------------
# Loss
# ----------------------------

def mse_loss(params, apply_fn, xb, yb, return_layer_act=False):
    if return_layer_act:
        preds, acts = apply_fn({"params": params}, xb, capture_layer_acts=True)
        return jnp.mean((preds - yb) ** 2), jnp.mean(acts, axis=0)
    else:
        preds = apply_fn({"params": params}, xb)
        return jnp.mean((preds - yb) ** 2)

# ----------------------------
# Grad norms (optional instrumentation)
# Currently unused
# ----------------------------

def grad_norms_by_layer(grads):
    flat = jax.tree_util.tree_flatten_with_path(grads)
    norms = {}
    for path, leaf in flat[0]:
        if leaf is None:
            continue
        key = "/".join([str(p) for p in path])
        norms[key] = jnp.linalg.norm(leaf)
    return norms

# ----------------------------
# Optimizer utilities
# ----------------------------
class TrainState(train_state.TrainState):
    lr_mults: Any  # pytree of LR multipliers (same structure as variables)


def make_optimizer(base_lr: float, lr_mults_pytree):
    # A transform that multiplies grads by the LR multipliers
    def scale_by_lr_mults(lr_mults):
        def init_fn(_): return optax.EmptyState()
        def update_fn(updates, state, params=None):
            scaled = jax.tree.map(lambda g, m: g * m, updates, lr_mults)
            return scaled, state
        return optax.GradientTransformation(init_fn, update_fn)

    tx = optax.chain(
        scale_by_lr_mults(lr_mults_pytree),
        optax.sgd(learning_rate=base_lr, momentum=0.0, nesterov=False),
    )
    return tx

def identity(x):
    return x

ACTS = {
    "identity": identity,
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
}


def create_state(
    rng,
    widths,
    activations,
    param_scheme_name: str,
    lr: float,
    sample_x,
    kernel_dims,
    nodes_per_layer,
    base_layer_widths, 
    base_kernel_dims,
):
    
    acts = [ACTS[act] for act in activations] 
    scheme_template = SCHEMES[param_scheme_name]
    scheme = (
        scheme_template
              .with_layer_widths(nodes_per_layer)
              .with_base_layer_widths(base_layer_widths)
              .with_base_kernel_dims(base_kernel_dims)
              )
              
    model = MLP(widths=tuple(widths), acts=acts, param_scheme=scheme, kernel_dims=kernel_dims, base_kernel_dims=base_kernel_dims)
    variables = model.init(rng, sample_x)
    params = variables["params"]
    lr_mults_vars = scheme.lr_multiplier_pytree(kernel_dims)
    tx = make_optimizer(lr, lr_mults_vars["params"])  # apply multipliers to param leaves
    state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        lr_mults=lr_mults_vars,
    )
    return state, model
