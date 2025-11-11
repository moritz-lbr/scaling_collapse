# train.py
from typing import Dict, Any, Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from utils import mse_loss, create_state, load_teacher_dataset


@jax.jit
def train_step(state, xb, yb):
    def loss_fn(p):
        return mse_loss(p, state.apply_fn, xb, yb)
    grads = jax.grad(loss_fn)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(params=new_params, opt_state=new_opt_state, step=state.step + 1)

def save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, epoch, total_epochs, progress):
    train_loss = mse_loss(state.params, state.apply_fn, xtr_full, ytr_full)
    test_loss, layer_activations = mse_loss(state.params, state.apply_fn, xte_full, yte_full, return_layer_act=True)
    
    history["train_loss"].append(float(train_loss))
    history["test_loss"].append(float(test_loss))
    history["layer_activations"].append(layer_activations.tolist())
    if progress is not None:
        progress(
            epoch=epoch + 1,
            total=total_epochs,
            train_loss=float(train_loss),
            test_loss=float(test_loss),
        )


def run_once(
    cfg,
    progress: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    
    test_split = float(cfg.test_split)

    def compute_split_counts(total: int) -> tuple[int, int]:
        if total <= 1:
            return total, 0
        n_test = int(round(total * test_split))
        n_test = max(1, min(total - 1, n_test))
        n_train = total - n_test
        return n_train, n_test

    rng_source = np.random.default_rng()

    inputs, targets = load_teacher_dataset(cfg)
    total_samples = int(inputs.shape[0])
    n_train, n_test = compute_split_counts(total_samples)
    permutation = rng_source.permutation(total_samples)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    if train_idx.size == 0:
        train_idx = permutation[:-1]
        test_idx = permutation[-1:]
    xtr_np = inputs[train_idx]
    ytr_np = targets[train_idx]
    xte = inputs[test_idx]
    yte = targets[test_idx]
    if set(train_idx) & set(test_idx):
        raise ValueError("Train and test sets overlap.")

    xtr_np = np.asarray(xtr_np, dtype=np.float32)
    ytr_np = np.asarray(ytr_np, dtype=np.float32)
    xte = np.asarray(xte, dtype=np.float32)
    yte = np.asarray(yte, dtype=np.float32)

    xtr_full = jnp.asarray(xtr_np)
    ytr_full = jnp.asarray(ytr_np)
    xte_full = jnp.asarray(xte)
    yte_full = jnp.asarray(yte)

    init_seed = int(rng_source.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    rng = jax.random.key(init_seed)
    sample_x = xtr_full[: min(8, xtr_full.shape[0])]
    state, _ = create_state(
        rng,
        cfg.widths,
        cfg.activations,
        cfg.param_scheme,
        cfg.lr,
        sample_x,
        cfg.kernel_dimensions,
        nodes_per_layer=cfg.nodes_per_layer,
        base_layer_widths=cfg.base_layer_widths,
        base_kernel_dims=cfg.base_kernel_dimensions
    )

    history = {"train_loss": [], "test_loss": [], "layer_activations": []}
    total_epochs = cfg.epochs
    batch_rng = rng_source
    batch_size = cfg.batch_size if cfg.batch_size and cfg.batch_size > 0 else len(xtr_np)

    for epoch in range(cfg.epochs):
        indices = batch_rng.permutation(len(xtr_np))
        if batch_size >= len(xtr_np):
            print("Using full-batch training.")
            xb = jnp.asarray(xtr_np[indices])
            yb = jnp.asarray(ytr_np[indices])
            state = train_step(state, xb, yb)
            save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, epoch, total_epochs, progress)
        else:
            for i, start in enumerate(range(0, len(indices), batch_size)):
                batch_idx = indices[start:start + batch_size]
                xb = jnp.asarray(xtr_np[batch_idx])
                yb = jnp.asarray(ytr_np[batch_idx])
                state = train_step(state, xb, yb)
                if cfg.save_loss_frequency == "epoch":
                    continue
                else:
                    if isinstance(cfg.save_loss_frequency, int):
                        if i % cfg.save_loss_frequency == 0:
                            save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, epoch, total_epochs, progress)
                    else: 
                        raise ValueError("Invalid save_loss_frequency value. Please provide an integer or 'epoch'.")
               
        if cfg.save_loss_frequency == "epoch":
            save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, epoch, total_epochs, progress)

    final_params = jax.device_get(state.params)

    dataset_info = {
        "n_train": int(xtr_np.shape[0]),
        "n_test": int(xte_full.shape[0]),
        "total": int(total_samples),
        "test_split": float(
            xte_full.shape[0] / max(1, total_samples)
        ),
        "save_loss_frequency": cfg.save_loss_frequency,
    }

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "history": history,
        "final_params": final_params,
        "dataset_info": dataset_info,
    }
