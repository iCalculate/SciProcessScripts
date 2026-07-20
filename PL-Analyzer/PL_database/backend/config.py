from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_CONFIG_TEMPLATE_PATH = PROJECT_ROOT / "config.example.yaml"


class ImporterSettings(BaseModel):
    backend: str = "witio"
    mock_mode: bool = False


MatlabSettings = ImporterSettings


class WitecSettings(BaseModel):
    extraction_mode: str = "auto"
    manual_x_paths: list[str] = Field(default_factory=list)
    manual_y_paths: list[str] = Field(default_factory=list)
    skip_keywords: list[str] = Field(
        default_factory=lambda: [
            "image",
            "map",
            "mapping",
            "topography",
            "video",
            "camera",
            "hyperspectral",
        ]
    )
    spectrum_keywords: list[str] = Field(
        default_factory=lambda: [
            "spectrum",
            "graph",
            "raman",
            "pl",
            "photoluminescence",
            "intensity",
        ]
    )


class DatabaseSettings(BaseModel):
    sqlite_path: str = "data/database/pl_spectra.sqlite3"
    hdf5_path: str = "data/hdf5/pl_spectra.h5"
    export_root: str = "data/exported"
    log_root: str = "data/logs"


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8110
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
    )


class AppSettings(BaseModel):
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)
    config_path: Path = Field(default_factory=lambda: DEFAULT_CONFIG_PATH)
    importer: ImporterSettings = Field(default_factory=ImporterSettings)
    witec: WitecSettings = Field(default_factory=WitecSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    @property
    def matlab(self) -> ImporterSettings:
        # Backward-compatible alias while configuration migrates away from MATLAB naming.
        return self.importer

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def sqlite_path(self) -> Path:
        return self.resolve_path(self.database.sqlite_path)

    @property
    def hdf5_path(self) -> Path:
        return self.resolve_path(self.database.hdf5_path)

    @property
    def export_root(self) -> Path:
        return self.resolve_path(self.database.export_root)

    @property
    def log_root(self) -> Path:
        return self.resolve_path(self.database.log_root)

    @property
    def frontend_dist(self) -> Path:
        return self.project_root / "frontend" / "dist"

    def ensure_runtime_dirs(self) -> None:
        for path in [
            self.sqlite_path.parent,
            self.hdf5_path.parent,
            self.export_root,
            self.log_root,
            self.project_root / "data" / "raw_wip",
        ]:
            path.mkdir(parents=True, exist_ok=True)


def ensure_local_config_exists(
    config_path: str | Path | None = None,
    template_path: str | Path | None = None,
) -> bool:
    target_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    source_path = (
        Path(template_path)
        if template_path is not None
        else DEFAULT_CONFIG_TEMPLATE_PATH
    )

    if not source_path.exists() or target_path.exists():
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    env_config_path = os.getenv("PLDB_CONFIG")
    if config_path is None and not env_config_path:
        ensure_local_config_exists()

    chosen_path = (
        Path(config_path)
        if config_path is not None
        else Path(env_config_path or DEFAULT_CONFIG_PATH)
    )
    chosen_path = chosen_path.expanduser()
    if not chosen_path.is_absolute():
        chosen_path = (PROJECT_ROOT / chosen_path).resolve()

    payload: dict = {}
    if chosen_path.exists():
        with chosen_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a mapping: {chosen_path}")
            payload = loaded

    if "importer" not in payload and isinstance(payload.get("matlab"), dict):
        legacy = payload.get("matlab") or {}
        payload["importer"] = {
            "backend": "witio",
            "mock_mode": bool(legacy.get("mock_mode", False)),
        }
    payload.pop("matlab", None)

    settings = AppSettings(
        project_root=PROJECT_ROOT,
        config_path=chosen_path,
        **payload,
    )
    settings.ensure_runtime_dirs()
    return settings
