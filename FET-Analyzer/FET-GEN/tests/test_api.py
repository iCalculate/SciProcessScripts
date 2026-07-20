import os
import json
from zipfile import ZipFile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import devicecurvegen.api as api_module
from devicecurvegen.api import app
from devicecurvegen.residual import ResidualEngine

client = TestClient(app)


@pytest.fixture(autouse=True)
def procedural_api_engine(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "residual_engine",
        ResidualEngine(discover_default=False),
    )
    monkeypatch.setattr(api_module, "model_load_error", None)


def test_health_and_generation() -> None:
    assert client.get("/health").status_code == 200
    response = client.post(
        "/api/generate",
        json={
            "target_ion": 1e-5,
            "target_ioff": 1e-11,
            "target_vth": 5,
            "target_ss_mv_dec": 120,
            "variants": 2,
            "points": 101,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["candidates"]) == 2
    assert len(payload["candidates"][0]["voltage"]) == 101
    assert payload["candidates"][0]["latent_code"]

    export = client.get(
        "/api/export",
        params={
            "seed": payload["candidates"][0]["seed"],
            "condition": json.dumps(payload["condition"]),
        },
    )
    assert export.status_code == 200
    assert export.headers["content-disposition"].startswith("attachment")
    assert export.text.startswith("Vg,Id_forward")


def test_extract_endpoint() -> None:
    voltage = np.linspace(-5, 10, 101)
    current = 1e-11 + 1e-5 / (1 + np.exp(-(voltage - 2) / 0.35))
    response = client.post(
        "/api/extract",
        json={"voltage": voltage.tolist(), "current": current.tolist()},
    )
    assert response.status_code == 200
    assert response.json()["polarity"] == "n-type"


def test_matrix_synthesis_endpoint_generates_site_outputs() -> None:
    response = client.post(
        "/api/database/matrix-synthesize",
        json={
            "rows": 1,
            "cols": 2,
            "mode": "generate",
            "duplicate_mode": "allow",
            "parameters": [
                {"key": "target_vth", "values": [[0.1, 0.6]]},
                {"key": "target_ion", "values": [[1e-6, 2e-6]]},
            ],
            "filters": {},
            "generation_condition": {
                "target_ion": 1e-6,
                "target_ioff": 1e-12,
                "target_vth": 0.0,
                "target_ss_mv_dec": 150,
                "points": 101,
                "variants": 1,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert [item["site"] for item in payload["assignments"]] == ["A1", "B1"]
    assert payload["generated_count"] == 2
    assert len(payload["assignments"][0]["generated"]["voltage"]) == 101

    export_response = client.post(
        "/api/database/matrix-export",
        json={
            "rows": 1,
            "cols": 1,
            "mode": "generate",
            "duplicate_mode": "allow",
            "parameters": [{"key": "target_vth", "values": [[0.1]]}],
            "filters": {},
            "generation_condition": {
                "target_ion": 1e-6,
                "target_ioff": 1e-12,
                "target_vth": 0.0,
                "target_ss_mv_dec": 150,
                "points": 101,
                "variants": 1,
            },
        },
    )
    assert export_response.status_code == 200
    assert export_response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    with ZipFile(BytesIO(export_response.content)) as workbook:
        names = set(workbook.namelist())
        curves_sheet = workbook.read("xl/worksheets/sheet3.xml").decode("utf-8")
    assert "xl/worksheets/sheet1.xml" in names
    assert "xl/worksheets/sheet2.xml" in names
    assert "xl/worksheets/sheet3.xml" in names
    assert "A1_seed" in curves_sheet
    assert "_id_forward_x" in curves_sheet
    assert "_id_forward_y" in curves_sheet
    assert "voltage_v" not in curves_sheet


def test_database_export_includes_xyxy_wide_csv(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "export_curve_rows",
        lambda curve_ids=None, filters=None: {
            "curves": [{"curve_id": "curveA"}],
            "raw_points": [
                {"curve_id": "curveA", "point_index": 0, "voltage_v": 0.0, "current_a": 1e-12},
                {"curve_id": "curveA", "point_index": 1, "voltage_v": 1.0, "current_a": 1e-9},
            ],
            "gate_points": [
                {"curve_id": "curveA", "point_index": 0, "voltage_v": 0.0, "current_a": 2e-13},
            ],
            "aligned_gate_points": [],
            "analysis": {"count": 1},
        },
    )
    response = client.post("/api/database/export", json={"curve_ids": ["curveA"], "filters": {}})
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        xyxy = archive.read("xyxy_curves.csv").decode("utf-8")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert "xyxy_curves.csv" in names
    assert manifest["files"][0] == "xyxy_curves.csv"
    assert xyxy.startswith("point_index,curveA_raw_id_x,curveA_raw_id_y")
    assert "curveA_raw_ig_x,curveA_raw_ig_y" in xyxy


def test_database_export_options_can_exclude_ig(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "export_curve_rows",
        lambda curve_ids=None, filters=None: {
            "curves": [{"curve_id": "curveA"}],
            "raw_points": [
                {"curve_id": "curveA", "point_index": 0, "voltage_v": 0.0, "current_a": 1e-12},
            ],
            "gate_points": [
                {"curve_id": "curveA", "point_index": 0, "voltage_v": 0.0, "current_a": 2e-13},
            ],
            "aligned_gate_points": [
                {"curve_id": "curveA", "point_index": 0, "voltage_v": 0.0, "abs_ig_a": 2e-13},
            ],
            "analysis": {"count": 1},
        },
    )
    response = client.post(
        "/api/database/export",
        json={
            "curve_ids": ["curveA"],
            "filters": {},
            "export_options": {"include_ig": False},
        },
    )
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        xyxy = archive.read("xyxy_curves.csv").decode("utf-8")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert "raw_ig_points.csv" not in names
    assert "aligned_ig_points.csv" not in names
    assert "_raw_ig_" not in xyxy
    assert manifest["export_options"]["include_ig"] is False
    assert "raw_ig_points.csv" not in manifest["files"]


def test_neural_training_status_exposes_log_space_parameters() -> None:
    response = client.get("/api/neural-training/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"idle", "completed", "failed", "running"}
    assert payload["config"]["low_current_weight"] == 1.5
    assert payload["config"]["subthreshold_weight"] == 2.5
    assert payload["config"]["slope_weight"] == 0.1
    assert payload["config"]["rare_curve_weight"] == 1.35


def test_model_compare_endpoint_returns_profiles() -> None:
    response = client.post(
        "/api/model/compare",
        json={
            "condition": {
                "target_ion": 1e-5,
                "target_ioff": 1e-11,
                "target_vth": 1.5,
                "target_ss_mv_dec": 140,
                "variants": 3,
                "points": 101,
                "ai_residual_strength": 0.65,
                "gate_ai_residual_strength": 0.65,
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["condition"]["variants"] == 1
    keys = [item["key"] for item in payload["items"]]
    assert "physics_only" in keys
    assert "active_model" in keys
    assert "procedural_prior" in keys
    assert payload["items"][0]["ai_residual_strength"] == 0.0
    assert payload["items"][1]["candidate"]["quality_score"] >= 0.0


def test_model_compare_endpoint_includes_leaderboard_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "residual-pca.npz"
    np.savez_compressed(
        checkpoint,
        model_type=np.asarray("learned_pca"),
        format_version=np.asarray(2, dtype=np.int64),
        grid=np.linspace(-1.0, 1.0, 17, dtype=np.float32),
        mean=np.zeros(17, dtype=np.float32),
        components=np.zeros((2, 17), dtype=np.float32),
        scales=np.ones(2, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "architecture": "latent_pca",
                    "method": "latent_pca",
                    "channels": ["Ids"],
                }
            )
        ),
    )

    leaderboard = api_module.ExperimentLeaderboardResponse(
        entries=[
            api_module.ExperimentLeaderboardEntry(
                name="best_jump",
                method="conditional_pca",
                architecture="conditional_pca",
                experiment_path=str(tmp_path),
                checkpoint_path=str(checkpoint),
                jump_p95_decades=0.53,
                generated_vth_mae_v=0.95,
                generated_ss_mae_mv_dec=291.0,
            ),
            api_module.ExperimentLeaderboardEntry(
                name="best_weighted",
                method="latent_pca",
                architecture="latent_pca",
                experiment_path=str(tmp_path),
                checkpoint_path=str(checkpoint),
                validation_weighted_rmse_decades=0.24,
            ),
        ],
        report_path=None,
    )
    monkeypatch.setattr(api_module, "_load_experiment_leaderboard", lambda limit=24: leaderboard)

    response = client.post(
        "/api/model/compare",
        json={"condition": {"target_ion": 1e-5, "target_ioff": 1e-11, "target_vth": 1.0, "target_ss_mv_dec": 140, "points": 101}},
    )
    assert response.status_code == 200
    keys = [item["key"] for item in response.json()["items"]]
    assert "best_jump_model" in keys


def test_model_leaderboard_endpoint_reads_experiment_summaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_dir = tmp_path / "example-run"
    experiment_dir.mkdir()
    (experiment_dir / "canonical-model-comparison.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        encoding="utf-8",
    )
    (tmp_path / "model-selection-report-20260627.md").write_text(
        "# report",
        encoding="utf-8",
    )
    (experiment_dir / "summary.json").write_text(
        json.dumps(
            [
                {
                    "name": "hybrid_best",
                    "description": "Hybrid threshold-guided sweep winner.",
                    "method": "hybrid_threshold_pca",
                    "seconds": 12.3,
                    "jump_p95_decades": 0.53,
                    "jump_spike_rate": 0.0007,
                    "generated_vth_mae_v": 0.96,
                    "generated_ss_mae_mv_dec": 291.6,
                    "checkpoint_path": "models/residual-hybrid-threshold-pca.npz",
                },
                {
                    "name": "pca_baseline",
                    "description": "Stable PCA baseline.",
                    "result": {
                        "method": "latent_pca",
                        "validation_weighted_rmse_decades": 0.2415,
                        "feature_vth_mae_v": 0.8082,
                        "feature_ss_mae_mv_dec": 382.7,
                        "output": "experiments/pca.npz",
                    },
                    "jump_metrics": {
                        "jump_p95_decades": 0.7103,
                        "jump_spike_rate": 0.0016,
                        "generated_vth_mae_v": 1.4344,
                        "generated_ss_mae_mv_dec": 206.3,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "experiments_root", tmp_path)

    response = client.get("/api/model/leaderboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["entries"][0]["name"] == "hybrid_best"
    assert payload["entries"][0]["method"] == "hybrid_threshold_pca"
    assert payload["entries"][1]["method"] == "latent_pca"
    assert payload["report_path"].endswith("model-selection-report-20260627.md")
    assert payload["comparison_artifact_url"].startswith(
        "/api/model/leaderboard-artifact/latest"
    )


def test_model_leaderboard_artifact_endpoint_serves_latest_svg(
    tmp_path: Path,
    monkeypatch,
) -> None:
    older_dir = tmp_path / "older"
    newer_dir = tmp_path / "newer"
    older_dir.mkdir()
    newer_dir.mkdir()
    older_svg = older_dir / "canonical-model-comparison.svg"
    newer_svg = newer_dir / "canonical-model-comparison.svg"
    older_svg.write_text("<svg>older</svg>", encoding="utf-8")
    newer_svg.write_text("<svg>newer</svg>", encoding="utf-8")
    os.utime(older_svg, (1_700_000_000, 1_700_000_000))
    os.utime(newer_svg, (1_800_000_000, 1_800_000_000))
    monkeypatch.setattr(api_module, "experiments_root", tmp_path)

    response = client.get("/api/model/leaderboard-artifact/latest")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "newer" in response.text


def test_example_file_is_served_without_path_traversal() -> None:
    response = client.get("/api/examples/sample_transfer.csv")
    assert response.status_code == 200
    assert response.text.startswith("Vg,Id")
    traversal = client.get("/api/examples/%2e%2e/pyproject.toml")
    assert traversal.status_code == 404


def test_frontend_shell_and_assets_disable_cache() -> None:
    index_response = client.get("/")
    if index_response.status_code == 404:
        pytest.skip("frontend dist is not available in this environment")
    assert index_response.status_code == 200
    assert index_response.headers["cache-control"] == "no-store"

    match = next(
        (
            line.split('"')[1]
            for line in index_response.text.splitlines()
            if "/assets/" in line and ('src="' in line or 'href="' in line)
        ),
        None,
    )
    assert match is not None

    asset_response = client.get(match)
    assert asset_response.status_code == 200
    assert asset_response.headers["cache-control"] == "no-store"


def test_inspect_override_and_file_limit(monkeypatch) -> None:
    response = client.post(
        "/api/inspect",
        files={"file": ("curve.csv", b"A,B\n0,1e-12\n1,1e-11\n2,1e-10\n3,1e-9")},
        data={"voltage_column": "A", "current_column": "B"},
    )
    assert response.status_code == 200
    assert response.json()["mapping"]["confidence"] == 1.0

    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 20)
    too_large = client.post(
        "/api/inspect",
        files={"file": ("curve.csv", b"Vg,Id\n" + b"0,1e-12\n" * 10)},
    )
    assert too_large.status_code == 413


def test_training_endpoint_activates_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEVICEGEN_MODEL_OUTPUT", str(tmp_path / "model.npz"))
    monkeypatch.setattr(api_module, "residual_engine", ResidualEngine())
    files = []
    for index in range(3):
        voltage = np.linspace(-5, 10, 81)
        current = 1e-11 + (1e-5 * (1 + 0.05 * index)) / (1 + np.exp(-(voltage - 2) / 0.4))
        content = "Vg,Id\n" + "\n".join(
            f"{vg},{ids}" for vg, ids in zip(voltage, current, strict=True)
        )
        files.append(("files", (f"curve-{index}.csv", content.encode(), "text/csv")))
    response = client.post("/api/train", files=files, data={"components": "4"})
    assert response.status_code == 200
    assert response.json()["curves"] == 3
    assert client.get("/api/model").json()["residual_mode"] == "learned_pca"
