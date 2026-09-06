from pathlib import Path

import pytest

from nommd_arena.glee import GleeConfigurationError, load_glee_api_key, load_glee_api_key_file


def test_loads_key_from_private_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLEE_API_KEY", raising=False)
    private = tmp_path / "private"
    private.mkdir()
    (private / "glee.env").write_text("# DeepRMM-01\nGLEE_API_KEY=glee_file_key\n", encoding="utf-8")
    assert load_glee_api_key(tmp_path) == "glee_file_key"


def test_environment_key_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLEE_API_KEY", "glee_environment_key")
    assert load_glee_api_key(tmp_path) == "glee_environment_key"


def test_missing_key_reports_configuration_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLEE_API_KEY", raising=False)
    with pytest.raises(GleeConfigurationError, match="GLEE_API_KEY is absent"):
        load_glee_api_key(tmp_path)


def test_rejects_malformed_key_without_echoing_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLEE_API_KEY", "wrong secret value")
    with pytest.raises(GleeConfigurationError) as error:
        load_glee_api_key(tmp_path)
    assert "wrong secret value" not in str(error.value)


def test_explicit_key_file_ignores_process_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    credential = tmp_path / "collector.env"
    credential.write_text("GLEE_API_KEY=glee_collector_key\n", encoding="utf-8")
    monkeypatch.setenv("GLEE_API_KEY", "glee_cloud_key")
    assert load_glee_api_key_file(credential) == "glee_collector_key"
