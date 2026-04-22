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
import pickle

from mlp import MLP
from param_schemes import SCHEMES # pyright: ignore[reportMissingImports]


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"


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
    return OrderedDict(
        sorted(
            ((k, int(v)) for k, v in nodes_map.items()),
            key=lambda kv: int(kv[0].split("_")[1])
        )
    )

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

def resolve_dataset_path(task_path: str | Path) -> Path:
    candidate = Path(task_path)
    candidates: List[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                candidate,
                REPO_ROOT / candidate,
                DATASETS_DIR / candidate,
            ]
        )

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Could not resolve dataset path from {task_path!r}.")

def getImageData(filename):
    with open(filename, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
        X = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        y = np.array(batch['labels'])
    return X, y


def load_all_cifar10_data(dataset_path: str | Path = "cifar-10-batches-py"):
    """Loads all CIFAR-10 batches and concatenates them into train/test datasets."""
    base_dir = resolve_dataset_path(dataset_path)
    train_paths = [base_dir / f"data_batch_{batch_idx}" for batch_idx in range(1, 6)]
    test_path = base_dir / "test_batch"

    x_train_list, y_train_list = [], []

    # Load all training batches
    for path in train_paths:
        X, y = getImageData(path)
        x_train_list.append(X)
        y_train_list.append(y)

    # Concatenate all training data
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Load test data
    x_test, y_test = getImageData(test_path)

    # Convert to float and scale to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0

    # Channel-wise z-score using TRAIN stats only
    # shapes: mean/std -> (1,1,1,3)
    mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    std = x_train.std(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(std, 1e-7)  # avoid divide-by-zero
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    x_train = x_train.reshape(x_train.shape[0], 3072)
    x_test = x_test.reshape(x_test.shape[0], 3072)

    return x_train, y_train, x_test, y_test


def load_winequality(
    path: str | Path = "winequality",
    *,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the wine quality classification dataset with train-split standardization."""
    base_path = resolve_dataset_path(path)

    if base_path.is_file():
        csv_paths = [base_path]
    else:
        csv_paths = [
            csv_path
            for csv_path in (
                base_path / "winequality-red.csv",
                base_path / "winequality-white.csv",
            )
            if csv_path.is_file()
        ]
        if not csv_paths:
            raise FileNotFoundError(
                f"Could not find winequality CSV files under {base_path}."
            )

    feature_blocks: List[np.ndarray] = []
    label_blocks: List[np.ndarray] = []
    include_wine_type_feature = len(csv_paths) > 1

    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=";", skiprows=1, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"Expected tabular data with at least one feature and one target column in {csv_path}."
            )

        features = data[:, :-1].astype(np.float32)
        labels = data[:, -1].astype(np.int32)

        if include_wine_type_feature:
            stem = csv_path.stem.lower()
            if "red" in stem:
                wine_type_value = 1.0
            elif "white" in stem:
                wine_type_value = 0.0
            else:
                raise ValueError(
                    f"Could not infer wine type from {csv_path.name}. Expected 'red' or 'white' in the filename."
                )
            wine_type_feature = np.full((features.shape[0], 1), wine_type_value, dtype=np.float32)
            features = np.concatenate([features, wine_type_feature], axis=1)

        feature_blocks.append(features)
        label_blocks.append(labels)

    features_all = np.concatenate(feature_blocks, axis=0)
    labels_all = np.concatenate(label_blocks, axis=0)

    if features_all.shape[0] <= 1:
        raise ValueError("Need at least two winequality samples to construct a train/test split.")

    n_test = int(round(features_all.shape[0] * float(test_split)))
    n_test = max(1, min(features_all.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(features_all.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train = features_all[train_idx]
    x_test = features_all[test_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - mean) / std).reshape(x_test.shape[0], -1).astype(np.float32)

    class_values = np.unique(labels_all)
    label_to_index = {int(label): idx for idx, label in enumerate(class_values.tolist())}
    labels_encoded = np.asarray([label_to_index[int(label)] for label in labels_all], dtype=np.int32)
    y_train_idx = labels_encoded[train_idx]
    y_test_idx = labels_encoded[test_idx]

    if one_hot:
        eye = np.eye(len(class_values), dtype=np.float32)
        y_train = eye[y_train_idx]
        y_test = eye[y_test_idx]
    else:
        y_train = y_train_idx.astype(np.int32)
        y_test = y_test_idx.astype(np.int32)

    return x_train, y_train, x_test, y_test


def load_drybean(
    path: str | Path = "drybean",
    *,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the dry bean classification dataset with train-split standardization."""
    base_path = resolve_dataset_path(path)
    arff_path = base_path if base_path.is_file() else base_path / "Dry_Bean_Dataset.arff"

    if not arff_path.is_file():
        raise FileNotFoundError(f"Could not find Dry_Bean_Dataset.arff under {base_path}.")

    feature_rows: List[List[float]] = []
    label_rows: List[str] = []
    class_names: List[str] | None = None
    in_data = False

    with arff_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()
            if not in_data:
                if lower_line.startswith("@attribute class"):
                    class_block = line[line.index("{") + 1:line.rindex("}")]
                    class_names = [name.strip() for name in class_block.split(",")]
                elif lower_line == "@data":
                    in_data = True
                continue

            values = [value.strip() for value in line.split(",")]
            feature_rows.append([float(value) for value in values[:-1]])
            label_rows.append(values[-1])

    features_all = np.asarray(feature_rows, dtype=np.float32)
    if features_all.shape[0] <= 1:
        raise ValueError("Need at least two drybean samples to construct a train/test split.")

    if class_names is None:
        class_names = sorted(set(label_rows))
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    labels_encoded = np.asarray([label_to_index[label] for label in label_rows], dtype=np.int32)

    n_test = int(round(features_all.shape[0] * float(test_split)))
    n_test = max(1, min(features_all.shape[0] - 1, n_test))

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(features_all.shape[0])
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]

    x_train = features_all[train_idx]
    x_test = features_all[test_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-7)
    x_train = ((x_train - mean) / std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - mean) / std).reshape(x_test.shape[0], -1).astype(np.float32)

    y_train_idx = labels_encoded[train_idx]
    y_test_idx = labels_encoded[test_idx]

    if one_hot:
        eye = np.eye(len(class_names), dtype=np.float32)
        y_train = eye[y_train_idx]
        y_test = eye[y_test_idx]
    else:
        y_train = y_train_idx.astype(np.int32)
        y_test = y_test_idx.astype(np.int32)

    return x_train, y_train, x_test, y_test


def load_yearpredictionmsd(path: str | Path = "yearpredictionmsd") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load YearPredictionMSD with the fixed publication train/test split."""
    base_path = resolve_dataset_path(path)
    data_path = base_path if base_path.is_file() else base_path / "YearPredictionMSD.txt"

    if not data_path.is_file():
        raise FileNotFoundError(f"Could not find YearPredictionMSD.txt under {base_path}.")

    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 91:
        raise ValueError(
            "Expected YearPredictionMSD data with shape (num_examples, 91), "
            f"got {data.shape} from {data_path}."
        )

    n_train = 463_715
    n_test = 51_630
    total_expected = n_train + n_test
    if data.shape[0] != total_expected:
        raise ValueError(
            f"Expected {total_expected} YearPredictionMSD rows, found {data.shape[0]} in {data_path}."
        )

    x_train = data[:n_train, 1:]
    y_train = data[:n_train, :1]
    x_test = data[n_train:, 1:]
    y_test = data[n_train:, :1]

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.maximum(x_std, 1e-7)
    x_train = ((x_train - x_mean) / x_std).reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = ((x_test - x_mean) / x_std).reshape(x_test.shape[0], -1).astype(np.float32)

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std = np.maximum(y_std, 1e-7)
    y_train = ((y_train - y_mean) / y_std).astype(np.float32)
    y_test = ((y_test - y_mean) / y_std).astype(np.float32)

    return x_train, y_train, x_test, y_test


def load_teacher_dataset(path) -> Tuple[np.ndarray, np.ndarray]:
    base_dir = resolve_dataset_path(path)

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

# ----------------------------
# Loss
# ----------------------------

def first_layer_term_matrix(params, first_layer_activation_matrix):
    if "Dense_1" not in params or "kernel" not in params["Dense_1"]:
        raise ValueError("Could not find Dense_1/kernel needed to compute first-layer term matrix.")

    second_layer_kernel = params["Dense_1"]["kernel"]
    if second_layer_kernel.ndim != 2:
        raise ValueError(
            "Expected Dense_1/kernel to have shape (width_0, output_dim) so the saved term matrix can be computed. "
            f"Got shape {second_layer_kernel.shape}."
        )

    if second_layer_kernel.shape[1] == 1:
        return first_layer_activation_matrix * second_layer_kernel[:, 0].astype(jnp.float32)

    return (
        first_layer_activation_matrix[:, :, None]
        * second_layer_kernel.astype(jnp.float32)[None, :, :]
    )

def mse_loss(params, apply_fn, xb, yb, return_layer_act=False):
    if return_layer_act:
        preds, acts, first_layer_activation_matrix = apply_fn({"params": params}, xb, capture_layer_acts=True)
        return (
            jnp.mean((preds - yb) ** 2),
            jnp.mean(acts, axis=0),
            first_layer_term_matrix(params, first_layer_activation_matrix),
        )
    else:
        preds = apply_fn({"params": params}, xb)
        return jnp.mean((preds - yb) ** 2)
    
# def cross_entropy_loss(params, apply_fn, xb, yb, return_layer_act=False):
#     # x: (N, C), y: (N,)
#     if return_layer_act:
#         preds, acts = apply_fn({"params": params}, xb, capture_layer_acts=True)
#     else:
#         preds = apply_fn({"params": params}, xb)
    
#     log_sum_exp = jnp.log(jnp.sum(jnp.exp(preds), axis=1))
#     rows = jnp.arange(preds.shape[0])
#     loss = -(preds[rows, yb.astype(jnp.int32)] - log_sum_exp)

#     if return_layer_act:
#         return jnp.mean(loss, axis=0), jnp.mean(acts, axis=0)
#     else: 
#         return jnp.mean(loss, axis=0)

def cross_entropy_loss(params, apply_fn, xb, yb, return_layer_act=False):
    def prepare_classification_labels(logits, labels):
        if logits.ndim != 2:
            raise ValueError(
                "Cross-entropy loss expects model outputs with shape (batch_size, num_classes). "
                f"Got shape {logits.shape}."
            )

        num_classes = int(logits.shape[-1])
        labels_array = jnp.asarray(labels)

        if labels_array.ndim == 1:
            return jax.nn.one_hot(labels_array.astype(jnp.int32), num_classes, dtype=jnp.float32)
        if labels_array.ndim == 2 and labels_array.shape[-1] == 1:
            flat_labels = labels_array[:, 0].astype(jnp.int32)
            return jax.nn.one_hot(flat_labels, num_classes, dtype=jnp.float32)
        if labels_array.ndim == 2 and labels_array.shape[-1] == num_classes:
            return labels_array.astype(jnp.float32)

        raise ValueError(
            "Unsupported classification target shape. Expected integer class indices with "
            f"shape (batch_size,) or one-hot targets with shape (batch_size, {num_classes}), "
            f"but got {labels_array.shape}."
        )

    if return_layer_act:
        preds, acts, first_layer_activation_matrix = apply_fn({"params": params}, xb, capture_layer_acts=True)
        labels_onehot = prepare_classification_labels(preds, yb)
        return (
            jnp.mean(optax.safe_softmax_cross_entropy(preds, labels=labels_onehot), axis=0),
            jnp.mean(acts, axis=0),
            first_layer_term_matrix(params, first_layer_activation_matrix),
        )
    else:
        preds = apply_fn({"params": params}, xb)
        labels_onehot = prepare_classification_labels(preds, yb)
        return jnp.mean(optax.safe_softmax_cross_entropy(preds, labels=labels_onehot), axis=0)


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

def decay_mask(params):
    # True where we want weight decay, False where we don't
    return jax.tree_util.tree_map(lambda p: p.ndim > 1, params)

def make_optimizer(base_lr: float, wd: float, lr_mults_pytree):
    # A transform that multiplies grads by the LR multipliers
    def scale_by_lr_mults(lr_mults):
        def init_fn(_): return optax.EmptyState()
        def update_fn(updates, state, params=None):
            scaled = jax.tree.map(lambda g, m: g * m, updates, lr_mults)
            return scaled, state
        return optax.GradientTransformation(init_fn, update_fn)

    tx = optax.chain(
        optax.add_decayed_weights(wd, mask=decay_mask),
        scale_by_lr_mults(lr_mults_pytree),
        optax.sgd(learning_rate=base_lr, momentum=None, nesterov=False),
    )
    return tx

def identity(x):
    return x

ACTS = {
    "identity": identity,
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "sigmoid" : jax.nn.sigmoid
}


def create_state(
    rng,
    widths,
    activations,
    param_scheme_name: str,
    lr: float,
    wd: float,
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
    tx = make_optimizer(lr, wd, lr_mults_vars["params"])  # apply multipliers to param leaves
    state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        lr_mults=lr_mults_vars
    )
    return state, model
