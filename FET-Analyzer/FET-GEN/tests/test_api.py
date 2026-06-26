import json

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


def test_neural_training_status_exposes_log_space_parameters() -> None:
    response = client.get("/api/neural-training/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"idle", "completed", "failed", "running"}
    assert payload["config"]["low_current_weight"] == 1.5
    assert payload["config"]["subthreshold_weight"] == 2.5
    assert payload["config"]["slope_weight"] == 0.1


def test_example_file_is_served_without_path_traversal() -> None:
    response = client.get("/api/examples/sample_transfer.csv")
    assert response.status_code == 200
    assert response.text.startswith("Vg,Id")
    traversal = client.get("/api/examples/%2e%2e/pyproject.toml")
    assert traversal.status_code == 404


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
