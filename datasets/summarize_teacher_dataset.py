import argparse
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np


def _load_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(dataset_path) as data:
        if "inputs" in data and "outputs" in data:
            inputs = data["inputs"].astype(np.float32)
            outputs = data["outputs"].astype(np.float32)
        elif all(key in data for key in ("x_train", "y_train", "x_test", "y_test")):
            inputs = np.concatenate(
                [data["x_train"].astype(np.float32), data["x_test"].astype(np.float32)],
                axis=0,
            )
            outputs = np.concatenate(
                [data["y_train"].astype(np.float32), data["y_test"].astype(np.float32)],
                axis=0,
            )
        else:
            raise ValueError(
                "Unsupported dataset format. Expected 'inputs'/'outputs' arrays "
                "or legacy 'x_train'/'y_train'/'x_test'/'y_test'."
            )
    return inputs, outputs


def _format_row(values: Iterable[float]) -> str:
    return "  ".join(f"{v:10.4f}" for v in values)


def _summarize_matrix(name: str, matrix: np.ndarray) -> None:
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    n_samples, n_features = matrix.shape
    print(f"\n{name}: {n_samples} samples • {n_features} features")
    if n_samples == 0:
        print("  (no samples)")
        return

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    mins = matrix.min(axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    medians = np.median(matrix, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    maxs = matrix.max(axis=0)

    header = "feature    mean        std      median        min           max"
    print(header)
    for idx in range(n_features):
        row_values = (
            float(idx),
            float(means[idx]),
            float(stds[idx]),
            float(medians[idx]),
            float(mins[idx]),
            float(maxs[idx]),
        )
        print(f"{idx:7d}  {_format_row(row_values[1:])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single MLP experiment defined in config.yaml.")
    parser.add_argument("--dataset", type=Path,help="Path to directory with configuration .npz data file.")
    args = parser.parse_args()

    inputs, outputs = _load_dataset(args.dataset)

    _summarize_matrix("Inputs", inputs)
    _summarize_matrix("Outputs", outputs)

if __name__ == "__main__":
    main()
