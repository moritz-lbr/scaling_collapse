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
    wd: float
    epochs: int
    batch_size: int
    task: str
    test_split: float
    save_loss_frequency: Any
    weight_monitoring: bool
    loss_type: str

    @property
    def widths(self) -> Tuple[int, ...]:
        sorted_nodes = sorted(
            ((str(k), int(v)) for k, v in self.nodes_per_layer.items()))
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


def load_simulation_parameters(input) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(input, Path):
        with open(input, "r") as f:
            raw = yaml.safe_load(f) or {}
    
    else:
        raw = input

    sim = raw.get("simulation_parameters")
    if not isinstance(sim, dict):
        raise ValueError("config.yaml must contain a 'simulation_parameters' mapping.")

    base_values: Dict[str, Any] = sim.get("network") | sim.get("training")

    return base_values


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
