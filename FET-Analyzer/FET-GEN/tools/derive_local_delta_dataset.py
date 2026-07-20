from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from devicecurvegen.neural import (
    _physics_baseline,
    _threshold_local_envelope_from_conditions,
    _training_target,
    load_exported_neural_dataset,
)


def derive_local_delta_dataset(
    source: Path,
    base_checkpoint: Path,
    output: Path,
    *,
    window_scale: float = 1.8,
    min_window_v: float = 0.12,
    floor: float = 0.03,
) -> Path:
    dataset = load_exported_neural_dataset(source.expanduser().resolve())
    physics = _physics_baseline(dataset)
    target, channels = _training_target(dataset, physics)
    with np.load(base_checkpoint.expanduser().resolve(), allow_pickle=True) as payload:
        mean = np.asarray(payload["mean"], dtype=np.float32)
        components = np.asarray(payload["components"], dtype=np.float32)
    if mean.shape[0] != target.shape[1]:
        raise ValueError("Base checkpoint dimensionality is incompatible with the dataset")
    centered = target - mean[None, :]
    projection = mean[None, :] + (centered @ components.T) @ components
    delta = target - projection
    points = dataset.grid.size
    drain_envelope = _threshold_local_envelope_from_conditions(
        grid=dataset.grid,
        conditions=dataset.conditions,
        window_scale=window_scale,
        min_window_v=min_window_v,
        floor=floor,
    )
    drain_delta = delta[:, :points] * drain_envelope
    if channels == 2:
        gate_delta = delta[:, points : 2 * points]
        gate_delta = gate_delta - np.median(gate_delta, axis=1, keepdims=True)
    else:
        gate_delta = None

    curves_csv = source.expanduser().resolve() / "curves.csv"
    curves = pd.read_csv(curves_csv)
    output_root = output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_root / "curves.csv", index=False)
    payload = {
        "curve_id": curves["curve_id"].astype(str).to_numpy(dtype=str),
        "x_norm": dataset.grid.astype(np.float32),
        "log10_abs_id": (physics + drain_delta).astype(np.float32),
    }
    if gate_delta is not None:
        payload["log10_abs_ig"] = gate_delta.astype(np.float32)
    np.savez_compressed(output_root / "aligned_curves.npz", **payload)
    manifest = {
        "source": str(source.expanduser().resolve()),
        "base_checkpoint": str(base_checkpoint.expanduser().resolve()),
        "curve_count": int(dataset.log_current.shape[0]),
        "channels": channels,
        "window_scale": window_scale,
        "min_window_v": min_window_v,
        "floor": floor,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a local threshold-delta training export relative to a stable PCA base checkpoint."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-scale", type=float, default=1.8)
    parser.add_argument("--min-window-v", type=float, default=0.12)
    parser.add_argument("--floor", type=float, default=0.03)
    args = parser.parse_args()
    print(
        derive_local_delta_dataset(
            args.source,
            args.base,
            args.output,
            window_scale=args.window_scale,
            min_window_v=args.min_window_v,
            floor=args.floor,
        )
    )


if __name__ == "__main__":
    main()
