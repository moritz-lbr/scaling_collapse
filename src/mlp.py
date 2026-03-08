from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", False)  # make sure you don't accidentally use FP64

from param_schemes import ParamScheme


# ----------------------------
# MLP with role-aware inits + manual AMP (no jmp)
# ----------------------------
class MLP(nn.Module):
    widths: Tuple[int, ...]
    acts: Callable
    param_scheme: ParamScheme
    kernel_dims: Any
    base_kernel_dims: Any

    # AMP controls
    compute_dtype: Any = jnp.bfloat16   # BF16 compute on A40
    param_dtype: Any = jnp.float32      # keep parameters in FP32
    output_dtype: Any = jnp.float32     # return FP32 for loss/metrics

    @nn.compact
    def __call__(self, x, capture_layer_acts: bool = False):
        layer_acts = []

        # Cast inputs to compute dtype (BF16)
        x = x.astype(self.compute_dtype)

        # Input layer
        kinit = self.param_scheme.kernel_inits("in", self.kernel_dims, self.base_kernel_dims)
        x = nn.Dense(
            self.widths[0],
            kernel_init=kinit,
            use_bias=False,
            name="Dense_0",
            dtype=self.compute_dtype,      # computation in BF16
            param_dtype=self.param_dtype,  # params stored in FP32
        )(x)

        if capture_layer_acts:
            layer_acts.append(jnp.abs(x).astype(jnp.float32).mean(axis=1))
        x = self.acts[0](x)

        # Hidden layers
        for i, w in enumerate(self.widths[1:-1], start=1):
            kinit = self.param_scheme.kernel_inits(f"Dense_{i-1}", self.kernel_dims, self.base_kernel_dims)
            x = nn.Dense(
                w,
                kernel_init=kinit,
                use_bias=False,
                name=f"Dense_{i}",
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )(x)

            if capture_layer_acts:
                layer_acts.append(jnp.abs(x).astype(jnp.float32).mean(axis=1))
            x = self.acts[i](x)

        # Output layer
        kinit = self.param_scheme.kernel_inits("out", self.kernel_dims, self.base_kernel_dims)
        y = nn.Dense(
            self.widths[-1],
            kernel_init=kinit,
            use_bias=False,
            name=f"Dense_{len(self.widths)-1}",
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)

        if capture_layer_acts:
            layer_acts.append(jnp.abs(y).astype(jnp.float32).mean(axis=1))
            stacked_act = jnp.stack(layer_acts, axis=0).T

        # Output activation (still BF16 compute), then cast to FP32 for downstream loss/metrics
        y = self.acts[-1](y).astype(self.output_dtype)

        if capture_layer_acts:
            return y, stacked_act
        return y

