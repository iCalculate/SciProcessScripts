from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from .b1500_dataset import build_b1500_dataset
from .database import (
    backfill_b1500_gate_points,
    create_database_engine,
    create_schema,
    database_status,
    import_b1500_to_mysql,
)
from .harmonize import inspect_measurement
from .neural import NeuralTrainingConfig, train_neural_checkpoint
from .physics import generate_curves
from .residual import ResidualEngine
from .schemas import GenerationCondition
from .training import train_residual_checkpoint

app = typer.Typer(no_args_is_help=True, help="DeviceCurveGen command-line interface")
SUPPORTED_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise typer.BadParameter(f"File does not exist: {path}")
    return path


def _require_directory(path: Path) -> Path:
    if not path.is_dir():
        raise typer.BadParameter(f"Directory does not exist: {path}")
    return path


@app.command()
def generate(
    output: Annotated[Path, typer.Option(help="Output CSV path")] = Path("generated.csv"),
    seed: int = 12345,
    variants: int = 1,
    ion: float = 1e-5,
    ioff: float = 1e-15,
    vth: float = 0.0,
    ss: float = 230.0,
    ss_region_width: float = 0.5,
    ai_strength: float = 0.0,
    physical_strictness: float = 0.0,
    diversity: float = 1.0,
    hysteresis: float = 1.5,
    noise: float = 1e-13,
    noise_floor: float = 1e-13,
    quantization_step: float = 1e-15,
    output_noise_gain: float = 4.0,
    gate_leakage: float = 1e-14,
    gate_leakage_v_char: float = 0.70,
    gate_leakage_exponent: float = 0.80,
    ion_sigma_fraction: float = 0.08,
    ioff_sigma_fraction: float = 0.15,
    vth_sigma: float = 0.20,
    ss_sigma_fraction: float = 0.10,
    hysteresis_sigma: float = 0.10,
    mobility: float = 20.0,
    mobility_sigma_fraction: float = 0.10,
    contact_resistance: float = 1e4,
    contact_resistance_sigma_fraction: float = 0.15,
    polarity: str = "n-type",
    voltage_min: float = -20.0,
    voltage_max: float = 20.0,
    points: int = 601,
) -> None:
    if polarity not in {"n-type", "p-type"}:
        raise typer.BadParameter("polarity must be n-type or p-type")
    result = generate_curves(
        GenerationCondition(
            seed=seed,
            variants=variants,
            target_ion=ion,
            target_ioff=ioff,
            target_vth=vth,
            target_ss_mv_dec=ss,
            ss_region_width_v=ss_region_width,
            ai_residual_strength=ai_strength,
            physical_strictness=physical_strictness,
            diversity=diversity,
            hysteresis_v=hysteresis,
            noise_sigma_a=noise,
            noise_floor_a=noise_floor,
            quantization_step_a=quantization_step,
            output_noise_gain=output_noise_gain,
            gate_leakage_a=gate_leakage,
            gate_leakage_v_char=gate_leakage_v_char,
            gate_leakage_exponent=gate_leakage_exponent,
            ion_sigma_fraction=ion_sigma_fraction,
            ioff_sigma_fraction=ioff_sigma_fraction,
            vth_sigma_v=vth_sigma,
            ss_sigma_fraction=ss_sigma_fraction,
            hysteresis_sigma_v=hysteresis_sigma,
            mobility_cm2_vs=mobility,
            mobility_sigma_fraction=mobility_sigma_fraction,
            contact_resistance_ohm=contact_resistance,
            contact_resistance_sigma_fraction=contact_resistance_sigma_fraction,
            polarity=polarity,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            points=points,
        ),
        ResidualEngine(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate",
                "seed",
                "Vg",
                "Id_forward",
                "Id_reverse",
                "Ig_forward",
                "Ig_reverse",
                "Id_physics_forward",
                "Id_physics_reverse",
            ]
        )
        for candidate in result.candidates:
            for values in zip(
                candidate.voltage,
                candidate.forward_current,
                candidate.reverse_current,
                candidate.gate_forward_current,
                candidate.gate_reverse_current,
                candidate.physics_forward_current,
                candidate.physics_reverse_current,
                strict=True,
            ):
                writer.writerow([candidate.candidate_id, candidate.seed, *values])
    typer.echo(f"Wrote {len(result.candidates)} candidate(s) to {output}")


@app.command()
def inspect(
    path: Path,
    voltage_column: str | None = None,
    current_column: str | None = None,
    output: Path | None = None,
) -> None:
    path = _require_file(path)
    try:
        result = inspect_measurement(
            path.name,
            path.read_bytes(),
            voltage_column=voltage_column,
            current_column=current_column,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    rendered = json.dumps(result.model_dump(mode="json"), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"Wrote inspection to {output}")
    else:
        typer.echo(rendered)


@app.command()
def extract(
    path: Path,
    voltage_column: str | None = None,
    current_column: str | None = None,
) -> None:
    path = _require_file(path)
    try:
        result = inspect_measurement(
            path.name,
            path.read_bytes(),
            voltage_column=voltage_column,
            current_column=current_column,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    payload = [
        {
            "direction": segment.direction,
            "rows": segment.rows,
            "features": segment.features.model_dump(mode="json") if segment.features else None,
        }
        for segment in result.segments
    ]
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def ingest(
    directory: Path,
    output: Annotated[Path, typer.Option(help="Dataset JSON path")] = Path("data/dataset.json"),
) -> None:
    directory = _require_directory(directory)
    files = sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    records: list[dict] = []
    errors: list[str] = []
    for path in files:
        try:
            records.append(
                inspect_measurement(path.name, path.read_bytes()).model_dump(mode="json")
            )
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": str(directory.resolve()),
                "files_discovered": len(files),
                "datasets": records,
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(records)} dataset(s) to {output}; {len(errors)} file(s) skipped")


@app.command()
def train(
    directory: Path,
    output: Annotated[Path, typer.Option(help="Residual checkpoint path")] = Path(
        "models/residual-pca.npz"
    ),
    components: int = 8,
) -> None:
    directory = _require_directory(directory)
    files = sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    try:
        result = train_residual_checkpoint(files, output, components=components)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@app.command("train-neural")
def train_neural(
    dataset: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Exported dataset directory containing aligned_curves.npz and curves.csv; "
                "omit to train directly from MySQL"
            )
        ),
    ] = None,
    output: Annotated[Path, typer.Option(help="Conditional VAE checkpoint path")] = Path(
        "models/residual-cvae.npz"
    ),
    database_url: Annotated[
        str | None,
        typer.Option(help="MySQL SQLAlchemy URL; defaults to DEVICEGEN_DATABASE_URL"),
    ] = None,
    latent_dim: int = 12,
    hidden_dim: int = 96,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    beta: float = 0.005,
    validation_fraction: float = 0.1,
    patience: int = 7,
    seed: int = 12345,
    max_curves: int | None = None,
    low_current_weight: float = 1.5,
    subthreshold_weight: float = 2.5,
    slope_weight: float = 0.10,
    feature_eval_limit: int = 512,
) -> None:
    if dataset is not None and not dataset.is_dir():
        raise typer.BadParameter(f"Dataset directory does not exist: {dataset}")
    config = NeuralTrainingConfig(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        beta=beta,
        validation_fraction=validation_fraction,
        patience=patience,
        seed=seed,
        max_curves=max_curves,
        low_current_weight=low_current_weight,
        subthreshold_weight=subthreshold_weight,
        slope_weight=slope_weight,
        feature_eval_limit=feature_eval_limit,
    )

    def report(metrics: dict[str, float | int | None]) -> None:
        def render(value: float | int | None) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        weighted_rmse = metrics.get("validation_weighted_rmse_decades")
        low_rmse = metrics.get("validation_low_current_rmse_decades")
        subthreshold_rmse = metrics.get("validation_subthreshold_rmse_decades")
        typer.echo(
            "epoch "
            f"{int(metrics['epoch']):03d} "
            f"train={float(metrics['train_loss']):.5f} "
            f"val={float(metrics['validation_loss']):.5f} "
            f"rmse={float(metrics['validation_rmse_decades']):.4f} decades "
            f"weighted={render(weighted_rmse)} "
            f"low={render(low_rmse)} "
            f"sub={render(subthreshold_rmse)}"
        )

    try:
        result = train_neural_checkpoint(
            output,
            dataset_path=dataset,
            database_url=database_url,
            config=config,
            progress=report,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))


@app.command("build-b1500")
def build_b1500(
    source: Annotated[Path, typer.Argument(help="B1500 export root directory")],
    output: Annotated[Path, typer.Option(help="Output dataset directory")] = Path(
        "data/b1500_test_dataset"
    ),
    max_xml_mb: Annotated[
        float,
        typer.Option(help="Skip direct XML/XTR files larger than this size in MB"),
    ] = 128.0,
    suffix: Annotated[
        list[str] | None,
        typer.Option("--suffix", help="Only include this suffix; repeat for multiple suffixes"),
    ] = None,
) -> None:
    source = _require_directory(source)
    summary = build_b1500_dataset(
        source,
        output,
        max_xml_mb=max_xml_mb,
        suffixes=set(suffix) if suffix else None,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("init-db")
def init_db(
    database_url: Annotated[
        str | None,
        typer.Option(help="MySQL SQLAlchemy URL; defaults to DEVICEGEN_DATABASE_URL"),
    ] = None,
) -> None:
    engine = create_database_engine(database_url)
    create_schema(engine)
    typer.echo("Database schema is ready")


@app.command("import-b1500-mysql")
def import_b1500_mysql(
    source: Annotated[Path, typer.Argument(help="B1500 export root directory")],
    database_url: Annotated[
        str | None,
        typer.Option(help="MySQL SQLAlchemy URL; defaults to DEVICEGEN_DATABASE_URL"),
    ] = None,
    replace: Annotated[
        bool,
        typer.Option(help="Delete existing DeviceCurveGen database rows before import"),
    ] = False,
    suffix: Annotated[
        list[str] | None,
        typer.Option("--suffix", help="Only include this suffix; repeat for multiple suffixes"),
    ] = None,
    max_xml_mb: Annotated[
        float,
        typer.Option(help="Skip direct XML/XTR files larger than this size in MB"),
    ] = 128.0,
    hash_files: Annotated[
        bool,
        typer.Option(help="Compute SHA1 for source files during import"),
    ] = False,
) -> None:
    source = _require_directory(source)
    summary = import_b1500_to_mysql(
        source,
        database_url,
        replace=replace,
        suffixes=set(suffix) if suffix else None,
        max_xml_mb=max_xml_mb,
        hash_files=hash_files,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("backfill-b1500-ig-mysql")
def backfill_b1500_ig_mysql(
    source: Annotated[Path, typer.Argument(help="B1500 export root directory")],
    database_url: Annotated[
        str | None,
        typer.Option(help="MySQL SQLAlchemy URL; defaults to DEVICEGEN_DATABASE_URL"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Parse source files and report counts without writing Ig points"),
    ] = False,
    replace: Annotated[
        bool,
        typer.Option(help="Rewrite existing raw/aligned Ig rows as well as missing rows"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(help="Limit candidate curves for testing a backfill run"),
    ] = None,
    max_xml_mb: Annotated[
        float,
        typer.Option(help="Skip direct XML/XTR files larger than this size in MB"),
    ] = 128.0,
    shard_index: Annotated[
        int,
        typer.Option(help="Zero-based source-file shard to process"),
    ] = 0,
    shard_count: Annotated[
        int,
        typer.Option(help="Number of stable source-file shards"),
    ] = 1,
) -> None:
    source = _require_directory(source)
    summary = backfill_b1500_gate_points(
        source,
        database_url,
        dry_run=dry_run,
        replace=replace,
        limit=limit,
        max_xml_mb=max_xml_mb,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("db-status")
def db_status(
    database_url: Annotated[
        str | None,
        typer.Option(help="MySQL SQLAlchemy URL; defaults to DEVICEGEN_DATABASE_URL"),
    ] = None,
) -> None:
    typer.echo(json.dumps(database_status(database_url), indent=2))


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8010,
    reload: bool = False,
) -> None:
    uvicorn.run("devicecurvegen.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
