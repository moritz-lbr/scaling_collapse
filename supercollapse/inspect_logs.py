from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

REPO_API_URL = "https://api.github.com/repos/shikaiqiu/supercollapse/contents/logs"
USER_AGENT = "supercollapse-log-inspector"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def request_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_remote_pickles() -> list[dict[str, Any]]:
    entries = request_json(REPO_API_URL)
    pickles = [entry for entry in entries if entry["type"] == "file" and entry["name"].endswith(".pkl")]
    return sorted(pickles, key=lambda entry: entry["name"])


def resolve_remote_target(entries: list[dict[str, Any]], target: str) -> dict[str, Any]:
    by_name = {entry["name"]: entry for entry in entries}
    if target in by_name:
        return by_name[target]
    if not target.endswith(".pkl") and f"{target}.pkl" in by_name:
        return by_name[f"{target}.pkl"]
    if target.isdigit():
        index = int(target)
        if 1 <= index <= len(entries):
            return entries[index - 1]

    matches = [entry for entry in entries if entry["name"].startswith(target)]
    if len(matches) == 1:
        return matches[0]

    available = ", ".join(entry["name"] for entry in entries)
    raise ValueError(f"Could not resolve '{target}'. Available files: {available}")


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def ensure_cached(entry: dict[str, Any], cache_dir: Path) -> Path:
    cache_path = cache_dir / entry["name"]
    expected_size = int(entry["size"])
    if cache_path.exists() and cache_path.stat().st_size == expected_size:
        return cache_path
    return download_file(entry["download_url"], cache_path)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def unique_preview(series: pd.Series, limit: int = 8) -> str:
    preview = list(series.drop_duplicates().head(limit + 1))
    rendered = ", ".join(format_value(value) for value in preview[:limit])
    if len(preview) > limit:
        rendered = f"{rendered}, ..."
    return rendered


def summarize_metadata(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    metadata = df.drop(columns=["history"], errors="ignore")
    if metadata.empty:
        return lines

    for column in metadata.columns:
        series = metadata[column]
        unique_count = int(series.nunique(dropna=False))
        if is_numeric_dtype(series):
            if unique_count <= 8:
                lines.append(f"- {column}: {unique_count} unique [{unique_preview(series)}]")
            else:
                lines.append(
                    f"- {column}: min={format_value(series.min())}, max={format_value(series.max())}, "
                    f"unique={unique_count}"
                )
        else:
            lines.append(f"- {column}: {unique_count} unique [{unique_preview(series)}]")
    return lines


def summarize_history_column(df: pd.DataFrame) -> list[str]:
    if "history" not in df.columns:
        return []

    histories = df["history"]
    frames = [history for history in histories if isinstance(history, pd.DataFrame)]
    if not frames:
        return ["No nested history DataFrames were found in the 'history' column."]
    if len(frames) != len(histories):
        return [f"'history' contains {len(frames)} DataFrames out of {len(histories)} rows."]

    lengths = [len(frame) for frame in frames]
    schema_set = {tuple(frame.columns) for frame in frames}
    lines = [
        f"History rows per run: min={min(lengths)}, median={statistics.median(lengths):.1f}, max={max(lengths)}",
    ]
    if len(schema_set) == 1:
        columns = ", ".join(frames[0].columns.astype(str))
        lines.append(f"History columns: {columns}")
    else:
        lines.append(f"History schemas vary across runs ({len(schema_set)} variants).")

    last_row = frames[0].tail(1)
    if not last_row.empty:
        summary = ", ".join(
            f"{column}={format_value(last_row.iloc[0][column])}"
            for column in last_row.columns
            if column in {"step", "compute", "test_loss", "lr", "tau", "scaled_L", "scaled_C", "scaled_tau"}
        )
        if summary:
            lines.append(f"First run final history row: {summary}")
    return lines


def inspect_pickle(path: Path, preview_runs: int, preview_history_rows: int) -> None:
    print(f"Local file: {path}")
    print(f"File size: {human_size(path.stat().st_size)}")
    print("Note: pickle loading executes pickle deserialization. Use only for trusted files.")

    obj = pd.read_pickle(path)
    print(f"Loaded object: {type(obj).__name__}")

    if not isinstance(obj, pd.DataFrame):
        print(repr(obj))
        return

    print(f"Outer table: {obj.shape[0]} rows x {obj.shape[1]} columns")
    print("Columns:")
    for column, dtype in obj.dtypes.items():
        print(f"- {column}: {dtype}")

    metadata_lines = summarize_metadata(obj)
    if metadata_lines:
        print("\nMetadata overview:")
        for line in metadata_lines:
            print(line)

    history_lines = summarize_history_column(obj)
    if history_lines:
        print("\nHistory overview:")
        for line in history_lines:
            print(f"- {line}")

    metadata = obj.drop(columns=["history"], errors="ignore")
    if not metadata.empty:
        print(f"\nSample runs (first {min(preview_runs, len(metadata))}):")
        print(metadata.head(preview_runs).to_string(index=False))

    if "history" in obj.columns and isinstance(obj.iloc[0]["history"], pd.DataFrame):
        sample_history = obj.iloc[0]["history"].head(preview_history_rows)
        print(f"\nSample history from first run (first {len(sample_history)} rows):")
        print(sample_history.to_string(index=False))


def command_list(_: argparse.Namespace) -> int:
    entries = fetch_remote_pickles()
    print(f"Found {len(entries)} remote pickle files in shikaiqiu/supercollapse/logs:\n")
    for index, entry in enumerate(entries, start=1):
        print(f"{index:>2}. {entry['name']:<18} {human_size(int(entry['size'])):>8}  {entry['download_url']}")
    return 0


def command_show(args: argparse.Namespace) -> int:
    target_path = Path(args.target).expanduser()
    if target_path.exists():
        inspect_pickle(target_path, args.preview_runs, args.preview_history_rows)
        return 0

    entries = fetch_remote_pickles()
    entry = resolve_remote_target(entries, args.target)
    local_path = ensure_cached(entry, args.cache_dir)
    print(f"Remote file: {entry['name']}")
    print(f"Raw URL: {entry['download_url']}")
    inspect_pickle(local_path, args.preview_runs, args.preview_history_rows)
    return 0


def command_download(args: argparse.Namespace) -> int:
    entries = fetch_remote_pickles()
    entry = resolve_remote_target(entries, args.target)
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / entry["name"]
    download_file(entry["download_url"], output_path)
    print(f"Downloaded {entry['name']} to {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List, download, and inspect remote training-log pickles from shikaiqiu/supercollapse.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available remote .pkl files.")
    list_parser.set_defaults(func=command_list)

    show_parser = subparsers.add_parser(
        "show",
        help="Inspect one remote .pkl file by name or 1-based index. Local paths also work.",
    )
    show_parser.add_argument("target", help="Remote file name, unique prefix, 1-based index, or local pickle path.")
    show_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Where downloaded files are cached.")
    show_parser.add_argument("--preview-runs", type=int, default=5, help="How many outer rows to print.")
    show_parser.add_argument(
        "--preview-history-rows",
        type=int,
        default=5,
        help="How many rows of the first nested history DataFrame to print.",
    )
    show_parser.set_defaults(func=command_show)

    download_parser = subparsers.add_parser("download", help="Download one remote .pkl file without cloning the repo.")
    download_parser.add_argument("target", help="Remote file name, unique prefix, or 1-based index.")
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory where the selected pickle is saved.",
    )
    download_parser.set_defaults(func=command_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else (["list"] if len(sys.argv) == 1 else None))
    try:
        return args.func(args)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"Network error while talking to GitHub: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
