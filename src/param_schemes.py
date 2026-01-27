from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Tuple
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import unflatten_dict

# ----------------------------
# Parametrization scheme hook
# ----------------------------


@dataclass
class ParamScheme:
    name: str
    input_layer: List[float]
    output_layer: List[float]
    hidden_layer: List[float]
    n_var: List[str]
    n_mult: List[str]
    n_lr: List[str]
    layer_widths: Dict[str, int] = field(default_factory=dict)
    params_init: Dict[str, float] = field(init=False)
    base_layer_widths: Dict[str, int] = field(default_factory=dict)   # per layer base "width" (usually out_dim)
    base_kernel_dims: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        # store (a, b) for each role
        self.params_init = {
            "in":     (self.input_layer[0],  self.input_layer[1]),
            "out":    (self.output_layer[0], self.output_layer[1]),
            "hidden": (self.hidden_layer[0], self.hidden_layer[1]),
        }
        # store c for each role
        self.params_lr = {
            "in":     (self.input_layer[2]),
            "out":    (self.output_layer[2]),
            "hidden": (self.hidden_layer[2]),
        }

    def with_base_layer_widths(self, base_widths: Dict[str, int]) -> "ParamScheme":
        sanitized = {str(k): int(v) for k, v in base_widths.items()}
        return replace(self, base_layer_widths=sanitized)

    def with_base_kernel_dims(self, base_kernel_dims: Dict[str, Dict[str, int]]) -> "ParamScheme":
        # Expect keys like: base_kernel_dims[layer]["fan_in"], ["fan_out"]
        # Values are ints coming from the BASE model's parameter shapes
        return replace(self, base_kernel_dims=base_kernel_dims)


    def with_layer_widths(self, layer_widths: Dict[str, int]) -> "ParamScheme":
        """Return a copy of this scheme carrying per-layer width metadata."""
        sanitized = {str(k): int(v) for k, v in (layer_widths).items()}
        return replace(self, layer_widths=sanitized)

    def scale_init(self, init_fn, scale):
        return lambda key, shape, dtype=jnp.float32: scale * init_fn(key, shape, dtype)


    def kernel_inits(self, role, kernel, base_kernel=None):
        if role == "in" or role == "out":
            n_var_type = role
            n_mult_type = role 
            a, b = self.params_init[role]
        else:
            n_var_type = "hidden"
            n_mult_type = "hidden" 
            a, b = self.params_init["hidden"]
        
        var_scaling = self.n_var[n_var_type]
        mult_scaling = self.n_mult[n_mult_type]
        # Current dims
        na_cur = 1.0 if mult_scaling == "one" else float(kernel[role][mult_scaling])
        nb_cur = 1.0 if var_scaling == "one" else float(kernel[role][var_scaling])
        # base dims 
        na_base = 1.0 if mult_scaling == "one" else float(base_kernel[role][mult_scaling])
        nb_base = 1.0 if var_scaling  == "one" else float(base_kernel[role][var_scaling])

        if self.name == "standard":
            var_ratio = nb_cur
            mult_ratio = na_cur
        else:
            if role == "out":
                var_ratio  = (nb_cur**2 / nb_base)
            else:
                var_ratio = nb_cur

            if role == "in" or role == "out":
                mult_ratio = (na_cur / na_base)
            else:
                mult_ratio = na_cur

        
        kernel_inits = self.scale_init(nn.initializers.normal(stddev=float(var_ratio) ** (-b)), scale=float(mult_ratio**(-a)))
        return kernel_inits

    
    def lr_multiplier_pytree(self, kernel_dims: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a pytree of scalar multipliers that matches variables['params'].
        Example rule: per-layer multiplier ~ width^(-c), where width = out_dim.
        """
        flat_mults: Dict[str, jnp.ndarray] = {}
        # print(self.base_kernel_dims.items())

        for i, layer in enumerate(kernel_dims.items()):
            role, vals = layer
            if role == "in" or role == "out":
                c = self.params_lr[role]
                mult_lr = self.n_lr[role] 
                n_cur = 1.0 if mult_lr == "one" else float(vals[mult_lr])
                n_base = 1.0 if mult_lr == "one" else float(self.base_kernel_dims[role][mult_lr])
            else: 
                c = self.params_lr["hidden"]
                mult_lr = self.n_lr["hidden"]
                n_cur = 1.0 if mult_lr == "one" else float(vals[mult_lr])
                n_base = 1.0 if mult_lr == "one" else float(self.base_kernel_dims[role][mult_lr])
            if self.name == "standard":
                n = n_cur
            else: 
                n = n_cur / n_base
            m = n ** (-c)
            flat_mults[f"Dense_{i}/kernel"] = jnp.array(m, dtype=jnp.float32)
        
        mults = unflatten_dict({tuple(p.split("/")): v for p, v in flat_mults.items()})

        return {"params": mults}
    


STANDARD = ParamScheme(
    name="standard",
    input_layer = [0.0, 0.5, 0.0],
    output_layer = [0.0, 0.5, 0.0],
    hidden_layer = [0.0, 0.5, 0.0],
    n_var = {"in": "fan_in", 
             "out": "fan_in", 
             "hidden": "fan_in"},
    n_mult = {"in": "one", 
              "out": "one", 
              "hidden": "one"},
    n_lr = {"in": "one", 
             "out": "one", 
             "hidden": "one"}
)


MUP = ParamScheme(
    name="muP",
    input_layer = [0.0, 0.5, -1.0],
    output_layer = [0.0, 1.0, 1.0],
    hidden_layer = [0.0, 0.5, 0.0],
    n_var = {"in": "fan_in", 
             "out": "fan_in", 
             "hidden": "fan_in"},
    n_mult = {"in": "one", 
              "out": "one", 
              "hidden": "one"},
    n_lr = {"in": "fan_out", 
             "out": "fan_in", 
             "hidden": "one"}
)

# MUP = ParamScheme(
#     name="muP",
#     input_layer = [0.0, 0.5, -1.0],
#     output_layer = [0.0, 0.5, 1.0],
#     hidden_layer = [0.0, 0.5, 0.0],
#     n_var = {"in": "fan_in", 
#              "out": "fan_in", 
#              "hidden": "fan_in"},
#     n_mult = {"in": "one", 
#               "out": "one", 
#               "hidden": "one"},
#     n_lr = {"in": "fan_out", 
#              "out": "fan_in", 
#              "hidden": "one"}
# )


SCHEMES = {
    "standard": STANDARD,
    "muP": MUP,
}

