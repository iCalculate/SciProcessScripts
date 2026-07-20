from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_export_subset(
    source: Path,
    output: Path,
    *,
    directions: tuple[str, ...],
) -> Path:
    source_root = source.expanduser().resolve()
    output_root = output.expanduser().resolve()
    matrix_path = source_root / "aligned_curves.npz"
    metadata_path = source_root / "curves.csv"
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            "Source dataset directory must contain aligned_curves.npz and curves.csv"
        )

    frame = pd.read_csv(metadata_path)
    if "curve_id" not in frame.columns or "direction" not in frame.columns:
        raise ValueError("Source curves.csv must contain curve_id and direction columns")
    selected = frame.loc[frame["direction"].isin(directions)].copy()
    if selected.empty:
        raise ValueError("No curves matched the requested direction filter")

    with np.load(matrix_path, allow_pickle=True) as payload:
        if "curve_id" not in payload.files:
            raise ValueError("aligned_curves.npz is missing curve_id")
        curve_ids = np.asarray(payload["curve_id"]).astype(str)
        curve_index = pd.Index(curve_ids)
        positions = curve_index.get_indexer(selected["curve_id"].astype(str))
        if np.any(positions < 0):
            missing = selected.loc[positions < 0, "curve_id"].astype(str).tolist()
            raise ValueError(
                f"Selected metadata contains {len(missing)} missing curve IDs in the matrix payload"
            )
        arrays: dict[str, np.ndarray] = {}
        for name in payload.files:
            array = np.asarray(payload[name])
            arrays[name] = array[positions] if array.shape[:1] == (curve_ids.shape[0],) else array

    output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_root / "curves.csv", index=False)
    np.savez_compressed(output_root / "aligned_curves.npz", **arrays)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a direction-filtered subset from an exported neural dataset."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--direction",
        dest="directions",
        action="append",
        choices=("forward", "reverse"),
        required=True,
        help="Sweep direction to keep; pass more than once to keep multiple directions.",
    )
    args = parser.parse_args()
    subset = build_export_subset(
        args.source,
        args.output,
        directions=tuple(args.directions),
    )
    print(subset)


if __name__ == "__main__":
    main()
