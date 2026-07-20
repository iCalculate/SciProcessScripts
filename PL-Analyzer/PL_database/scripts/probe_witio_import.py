from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.witio_import import DatasetSample, probe_witio_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a WITec .wip file with the Python witio reader.")
    parser.add_argument("wip_path", type=Path, help="Path to the .wip file to inspect.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / ".tmp" / "witio_probe",
        help="Directory where the report JSON and sampled CSV traces will be written.",
    )
    parser.add_argument("--max-point-datasets", type=int, default=2)
    parser.add_argument("--max-line-datasets", type=int, default=1)
    parser.add_argument("--max-area-datasets", type=int, default=1)
    parser.add_argument("--max-series-datasets", type=int, default=1)
    parser.add_argument("--max-traces-per-point", type=int, default=1)
    parser.add_argument("--max-traces-per-line", type=int, default=5)
    parser.add_argument("--max-traces-per-area", type=int, default=4)
    parser.add_argument("--max-traces-per-series", type=int, default=5)
    return parser


def write_dataset_samples(output_dir: Path, dataset_samples: list[DatasetSample]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset in dataset_samples:
        trace_records: list[dict[str, object]] = []
        for trace in dataset.traces:
            csv_path = output_dir / trace.file_name
            csv_path.write_text(
                "x_axis,intensity\n"
                + "\n".join(
                    f"{float(x_value):.10f},{float(y_value):.10f}"
                    for x_value, y_value in zip(trace.x_axis, trace.intensity, strict=True)
                )
                + "\n",
                encoding="utf-8",
            )
            trace_records.append(
                {
                    "csv_path": str(csv_path),
                    "trace_index": trace.metadata.get("trace_index"),
                    "grid_x": trace.metadata.get("grid_x"),
                    "grid_y": trace.metadata.get("grid_y"),
                    "secondary_axis_kind": trace.metadata.get("secondary_axis_kind"),
                    "secondary_axis_unit": trace.metadata.get("secondary_axis_unit"),
                    "secondary_axis_value": trace.metadata.get("secondary_axis_value"),
                    "source_tree_path": trace.metadata.get("source_tree_path"),
                    "n_points": int(len(trace.x_axis)),
                    "mean_intensity": float(trace.intensity.mean()),
                    "max_intensity": float(trace.intensity.max()),
                }
            )
        records.append(
            {
                **dataset.summary,
                "trace_exports": trace_records,
            }
        )
    return records


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report, dataset_samples = probe_witio_file(
        args.wip_path,
        max_point_datasets=args.max_point_datasets,
        max_line_datasets=args.max_line_datasets,
        max_area_datasets=args.max_area_datasets,
        max_series_datasets=args.max_series_datasets,
        max_traces_per_point=args.max_traces_per_point,
        max_traces_per_line=args.max_traces_per_line,
        max_traces_per_area=args.max_traces_per_area,
        max_traces_per_series=args.max_traces_per_series,
    )
    report["selected_datasets"] = write_dataset_samples(output_dir, dataset_samples)

    report_path = output_dir / "witio_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Report written to: {report_path}")
    print(json.dumps(report["inventory_by_mode"], ensure_ascii=False))
    for dataset in report["selected_datasets"]:
        print(
            f"{dataset['acquisition_mode']}: {dataset['caption']} "
            f"(sampled {dataset['sampled_trace_count']}/{dataset['trace_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
