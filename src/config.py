from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import yaml
from pathlib import Path


@dataclass
class Config:
    activations_per_layer: Dict[str, str]
    nodes_per_layer: Dict[str, int]
    base_layer_widths: Dict[str, int]
    param_scheme: str
    lr: float
    epochs: int
    batch_size: int
    task: str
    test_split: float
    save_loss_frequency: Any
    loss_type: str

    @property
    def widths(self) -> Tuple[int, ...]:
        sorted_nodes = sorted(
            ((str(k), int(v)) for k, v in self.nodes_per_layer.items()),
            key=lambda kv: _layer_index(kv[0])
        )
        if len(sorted_nodes) < 2:
            raise ValueError("nodes_per_layer must provide at least one hidden layer and one output layer.")
        return [val for _, val in sorted_nodes]

    @property
    def activations(self) -> str:
        return tuple(val for _, val in self.activations_per_layer.items())

    @property
    def kernel_dimensions(self) -> dict:
        num_nodes_per_layer = list(self.nodes_per_layer.values())
        labels = list(self.nodes_per_layer.keys())
        kernel_dims = {"in": {"fan_in": self.num_input_features,
                              "fan_out": num_nodes_per_layer[0]}}
        
        for i in range(len(num_nodes_per_layer) - 2):
            kernel_dims[labels[i]] = {"fan_in": num_nodes_per_layer[i],
                                      "fan_out": num_nodes_per_layer[i+1]}

        kernel_dims["out"] = {"fan_in": num_nodes_per_layer[-2],
                               "fan_out": num_nodes_per_layer[-1]}
        
        return kernel_dims
    
    @property
    def base_kernel_dimensions(self) -> dict:
        nodes_per_base_layer = list(self.base_layer_widths.values())
        labels = list(self.base_layer_widths.keys())
        base_kernel_dims = {"in": {"fan_in": self.num_input_features,
                              "fan_out": nodes_per_base_layer[0]}}
        
        for i in range(len(nodes_per_base_layer) - 2):
            base_kernel_dims[labels[i]] = {"fan_in": nodes_per_base_layer[i],
                                      "fan_out": nodes_per_base_layer[i+1]}

        base_kernel_dims["out"] = {"fan_in": nodes_per_base_layer[-2],
                               "fan_out": nodes_per_base_layer[-1]}
        
        return base_kernel_dims
    

REQUIRED_FIELDS = set(Config.__annotations__.keys())


def _layer_index(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _ensure_nodes_map(nodes_map: Dict[str, Any]) -> Dict[str, int]:
    cleaned = OrderedDict()
    for name, value in (nodes_map or {}).items():
        cleaned[str(name)] = int(value)
    if len(cleaned) < 2:
        raise ValueError("nodes_per_layer must include hidden layers and an output layer.")
    for val in cleaned.values():
        if val <= 0:
            raise ValueError("nodes_per_layer entries must be positive integers.")
    return dict(sorted(cleaned.items(), key=lambda kv: _layer_index(kv[0])))


def _ensure_activation_map(activations_map: Dict[str, str], nodes_map: Dict[str, int]) -> Dict[str, str]:
    cleaned = dict(activations_map)
    sorted_nodes = sorted(nodes_map.keys(), key=_layer_index)
    default_act = "tanh"
    for key in sorted_nodes[:-1]:
        cleaned.setdefault(key, default_act)
    cleaned.setdefault(sorted_nodes[-1], "linear")
    return cleaned


def load_simulation_parameters(input) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(input, Path):
        with open(input, "r") as f:
            raw = yaml.safe_load(f) or {}
    
    else:
        raw = input

    sim = raw.get("simulation_parameters")
    if not isinstance(sim, dict):
        raise ValueError("config.yaml must contain a 'simulation_parameters' mapping.")

    network_cfg = sim.get("network")
    training_cfg = sim.get("training")

    nodes_map = _ensure_nodes_map(network_cfg.get("nodes_per_layer"))
    activations_map = _ensure_activation_map(network_cfg.get("activations_per_layer"), nodes_map)
    param_scheme = str(network_cfg.get("param_scheme"))

    lr = float(training_cfg.get("lr"))
    epochs = int(training_cfg.get("epochs"))
    batch_size = int(training_cfg.get("batch_size"))
    task = str(training_cfg.get("task"))
    test_split = float(training_cfg.get("test_split"))
    save_loss_frequency = training_cfg.get("save_loss_frequency")
    base_layer_widths = network_cfg.get("base_layer_widths")
    loss_type = training_cfg.get("loss_type")

    base_values: Dict[str, Any] = {
        "activations_per_layer": activations_map,
        "nodes_per_layer": nodes_map,
        "param_scheme": param_scheme,
        "base_layer_widths": base_layer_widths,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "task": task,
        "test_split": test_split,
        "save_loss_frequency": save_loss_frequency,
        "loss_type": loss_type
    }

    return base_values, {"network": network_cfg, "training": training_cfg}


def ensure_required_fields(values: Dict[str, Any]) -> None:
    missing = sorted(name for name in REQUIRED_FIELDS if name not in values or values[name] is None)
    if missing:
        raise ValueError(
            "Missing configuration values: "
            + ", ".join(missing)
            + ". Provide them in config.yaml."
        )


def create_config(values: Dict[str, Any]) -> Config:
    ensure_required_fields(values)
    filtered = {k: values[k] for k in REQUIRED_FIELDS}
    return Config(**filtered)
