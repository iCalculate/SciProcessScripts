from __future__ import annotations

from pathlib import Path

from backend import config as config_module


def test_load_settings_seeds_local_config_from_template(tmp_path: Path, monkeypatch) -> None:
    default_config = tmp_path / "config.yaml"
    config_template = tmp_path / "config.example.yaml"
    config_template.write_text(
        "matlab:\n"
        "  mock_mode: true\n"
        "database:\n"
        "  sqlite_path: data/database/seeded.sqlite3\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", default_config)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_TEMPLATE_PATH", config_template)
    monkeypatch.delenv("PLDB_CONFIG", raising=False)

    settings = config_module.load_settings()

    assert default_config.exists()
    assert settings.config_path == default_config
    assert settings.matlab.mock_mode is True
    assert settings.sqlite_path == tmp_path / "data" / "database" / "seeded.sqlite3"
