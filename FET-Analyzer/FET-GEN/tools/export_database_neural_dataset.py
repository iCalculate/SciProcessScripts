from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from devicecurvegen.database import (
    aligned_gate_points,
    aligned_points,
    create_database_engine,
    create_schema,
    curves,
    source_files,
)


def export_database_neural_dataset(
    output: Path,
    *,
    database_url: str,
    directions: tuple[str, ...] | None = None,
) -> Path:
    engine = create_database_engine(database_url)
    create_schema(engine)
    metadata_query = (
        select(
            curves.c.curve_id,
            source_files.c.source_path,
            curves.c.direction,
            curves.c.voltage_min_v,
            curves.c.voltage_max_v,
            curves.c.ion.label("feature_ion"),
            curves.c.ioff.label("feature_ioff"),
            curves.c.polarity.label("feature_polarity"),
            curves.c.vth.label("feature_vth"),
            curves.c.ss_mv_dec.label("feature_ss_mv_dec"),
            curves.c.gm_max.label("feature_gm_max"),
            curves.c.hysteresis_v.label("feature_hysteresis_v"),
            curves.c.noise_log_sigma.label("feature_noise_log_sigma"),
            curves.c.leakage_level.label("feature_leakage_level"),
            curves.c.has_gate_current,
        )
        .select_from(curves.join(source_files, curves.c.source_file_id == source_files.c.id))
        .where(
            curves.c.vth.is_not(None),
            curves.c.ss_mv_dec.is_not(None),
            curves.c.polarity.in_(["n-type", "p-type"]),
        )
        .order_by(curves.c.curve_id)
    )
    if directions:
        metadata_query = metadata_query.where(curves.c.direction.in_(list(directions)))
    with engine.connect() as connection:
        rows = connection.execute(metadata_query).mappings().all()
        if not rows:
            raise ValueError("Database contains no trainable transfer curves for this filter")
        frame = pd.DataFrame(rows)
        curve_ids = frame["curve_id"].astype(str).to_numpy(dtype=str)
        point_count = int(
            connection.scalar(
                select(aligned_points.c.point_index)
                .order_by(aligned_points.c.point_index.desc())
                .limit(1)
            )
            or 0
        ) + 1
        if point_count < 16:
            raise ValueError("Database aligned curves have too few points")
        first_curve_id = curve_ids[0]
        grid = np.asarray(
            connection.scalars(
                select(aligned_points.c.x_norm)
                .where(aligned_points.c.curve_id == first_curve_id)
                .order_by(aligned_points.c.point_index)
            ).all(),
            dtype=np.float32,
        )
        curve_index = {curve_id: index for index, curve_id in enumerate(curve_ids)}
        log_current = np.full((len(curve_ids), point_count), np.nan, dtype=np.float32)
        log_gate_current = np.full_like(log_current, np.nan, dtype=np.float32)

        point_query = select(
            aligned_points.c.curve_id,
            aligned_points.c.point_index,
            aligned_points.c.log10_abs_id,
        ).order_by(aligned_points.c.curve_id, aligned_points.c.point_index)
        for partition in connection.execution_options(stream_results=True).execute(
            point_query
        ).partitions(20_000):
            for curve_id, point_index, log_id in partition:
                row_index = curve_index.get(str(curve_id))
                if row_index is None or point_index >= point_count:
                    continue
                log_current[row_index, point_index] = float(log_id)

        gate_query = select(
            aligned_gate_points.c.curve_id,
            aligned_gate_points.c.point_index,
            aligned_gate_points.c.log10_abs_ig,
        ).order_by(
            aligned_gate_points.c.curve_id,
            aligned_gate_points.c.point_index,
        )
        for partition in connection.execution_options(stream_results=True).execute(
            gate_query
        ).partitions(20_000):
            for curve_id, point_index, log_ig in partition:
                row_index = curve_index.get(str(curve_id))
                if row_index is None or point_index >= point_count:
                    continue
                log_gate_current[row_index, point_index] = float(log_ig)

    output_root = output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "curves.csv", index=False)
    np.savez_compressed(
        output_root / "aligned_curves.npz",
        curve_id=curve_ids,
        x_norm=grid,
        log10_abs_id=log_current,
        log10_abs_ig=log_gate_current,
    )
    manifest = {
        "database_url": database_url,
        "curve_count": int(len(frame)),
        "directions": sorted(frame["direction"].astype(str).unique().tolist()),
        "direction_counts": frame["direction"].value_counts().to_dict(),
        "polarity_counts": frame["feature_polarity"].value_counts().to_dict(),
        "gate_complete_curves": int(np.count_nonzero(np.all(np.isfinite(log_gate_current), axis=1))),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trainable neural dataset directly from the project database."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--database-url",
        default="mysql+pymysql://root@127.0.0.1:3307/devicecurvegen",
    )
    parser.add_argument(
        "--direction",
        dest="directions",
        action="append",
        choices=("forward", "reverse", "single"),
        help="Optional direction filter; repeat to include multiple directions.",
    )
    args = parser.parse_args()
    exported = export_database_neural_dataset(
        args.output,
        database_url=args.database_url,
        directions=tuple(args.directions) if args.directions else None,
    )
    print(exported)


if __name__ == "__main__":
    main()
