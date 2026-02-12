# train.py
from typing import Dict, Any, Callable, Optional
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict
import zarr

from utils import mse_loss, cross_entropy_loss, create_state, load_teacher_dataset, load_all_cifar10_data


def make_train_step(loss_fn):
    @jax.jit
    def train_step(state, xb, yb):
        grads = jax.grad(loss_fn)(state.params, state.apply_fn, xb, yb)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(params=new_params, opt_state=new_opt_state, step=state.step + 1)
    return train_step

def make_loss_saver(loss_fn):
    def save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, train_step, total_train_steps, progress=None):
        train_loss = loss_fn(state.params, state.apply_fn, xtr_full, ytr_full)
        test_loss, layer_activations = loss_fn(state.params, state.apply_fn, xte_full, yte_full, return_layer_act=True)
        
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        history["layer_activations"].append(layer_activations.tolist())

        if progress is not None:
            progress(
                epoch=int(train_step),
                total=int(total_train_steps),
                train_loss=float(train_loss),
                test_loss=float(test_loss),
            )
    return save_losses

def init_weight_store(zarr_path, params, total_steps, sep="/", dtype=None, time_chunk=64):
    root = zarr.open_group(zarr_path, mode="w")
    flat = flatten_dict(params, sep=sep)

    root.attrs["total_steps"] = int(total_steps)
    root.attrs["param_keys"] = list(flat.keys())

    for k, v in flat.items():
        v0 = np.array(jax.device_get(v))
        dt = dtype if dtype is not None else v0.dtype

        root.create_dataset(
            name=k,
            shape=(total_steps,) + v0.shape,
            chunks=(min(time_chunk, total_steps),) + v0.shape,  # chunk along time
            dtype=dt,
        )
    return root

def append_params(root, params, step, sep="/", cast_dtype=None):
    flat = flatten_dict(params, sep=sep)

    for k, v in flat.items():
        arr = np.array(jax.device_get(v))
        if cast_dtype is not None:
            arr = arr.astype(cast_dtype)
        root[k][int(step), ...] = arr


def run_once(
    cfg,
    run_dir: Path,
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

    if "teacher_data" in cfg.task or "regression_data" in cfg.task:
        inputs, targets = load_teacher_dataset(cfg.task)
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
    else: 
        xtr_np, ytr_np, xte, yte = load_all_cifar10_data()
        total_samples = int(xtr_np.shape[0] + xte.shape[0])
  

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
        cfg.wd,
        sample_x,
        cfg.kernel_dimensions,
        nodes_per_layer=cfg.nodes_per_layer,
        base_layer_widths=cfg.base_layer_widths,
        base_kernel_dims=cfg.base_kernel_dimensions
    )

    history = {"train_loss": [], "test_loss": [], "layer_activations": []}
    batch_rng = rng_source
    batch_size = cfg.batch_size if cfg.batch_size and cfg.batch_size > 0 else len(xtr_np)

    if cfg.loss_type == "mse":
        loss_function = mse_loss
    elif cfg.loss_type == "cross_entropy":
        loss_function = cross_entropy_loss
    else:
        raise ValueError("The loss type specified in the config could not be found")
    
    train_step = make_train_step(loss_function)
    save_losses = make_loss_saver(loss_function)

    total_steps = cfg.epochs * int(len(xtr_full) / cfg.batch_size)
    if isinstance(cfg.save_loss_frequency, str):
        total_steps_saved = cfg.epochs
    else:
        total_steps_saved = int(total_steps / cfg.save_loss_frequency)

    root = init_weight_store(run_dir / "weights.zarr", state.params, total_steps_saved + 1)

    # Save initial weights if weight_monitoring is activated
    save_step_counter = 0
    append_params(root, state.params, save_step_counter, cast_dtype=np.float16) 

    for epoch in range(cfg.epochs):
        indices = batch_rng.permutation(len(xtr_np))

        if batch_size >= len(xtr_np):
            print("Using full-batch training.")
            xb = jnp.asarray(xtr_np[indices])
            yb = jnp.asarray(ytr_np[indices])
            state = train_step(state, xb, yb)
            should_save = True if cfg.save_loss_frequency == "epoch" else (
                isinstance(cfg.save_loss_frequency, int)
                and state.step % cfg.save_loss_frequency == 0
            )
            if should_save:
                save_step_counter += 1
                save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, save_step_counter, total_steps_saved, progress)
                append_params(root, state.params, save_step_counter, cast_dtype=np.float16) 

        else:
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                xb = jnp.asarray(xtr_np[batch_idx])
                yb = jnp.asarray(ytr_np[batch_idx])
                state = train_step(state, xb, yb)

                if cfg.save_loss_frequency == "epoch":
                    continue
                elif isinstance(cfg.save_loss_frequency, int):
                    if state.step % cfg.save_loss_frequency == 0:
                        save_step_counter += 1
                        save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, save_step_counter, total_steps_saved, progress)
                        append_params(root, state.params, save_step_counter, cast_dtype=np.float16) 
 
                else: 
                    raise ValueError("Invalid save_loss_frequency value. Please provide an integer or 'epoch'.")
               
        if cfg.save_loss_frequency == "epoch":
            save_step_counter += 1
            save_losses(history, state, xtr_full, ytr_full, xte_full, yte_full, save_step_counter, total_steps_saved, progress)
            append_params(root, state.params, save_step_counter, cast_dtype=np.float16) 

    final_params = jax.device_get(state.params)

    dataset_info = {
        "total": int(total_samples),
        "n_train": int(xtr_np.shape[0]),
        "n_test": int(xte_full.shape[0]),
        "test_split": float(test_split),
    }

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "history": history,
        "final_params": final_params,
        "dataset_info": dataset_info,
    }
