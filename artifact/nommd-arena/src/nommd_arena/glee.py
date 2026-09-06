"""GLEE Competition connection helpers."""

from __future__ import annotations

import os
from pathlib import Path

from glee_sdk import GleeClient


class GleeConfigurationError(RuntimeError):
    """Report a missing or malformed GLEE connection without exposing its key."""


def _validate_api_key(value: str) -> str:
    key = value.strip()
    if not key.startswith("glee_") or len(key) <= len("glee_"):
        raise GleeConfigurationError("GLEE_API_KEY must contain a GLEE agent key beginning with 'glee_'")
    if any(character.isspace() for character in key):
        raise GleeConfigurationError("GLEE_API_KEY must not contain whitespace")
    return key


def _read_env_value(path: Path, variable: str) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == variable:
            return value.strip().strip("'\"")
    return None


def load_glee_api_key(project_root: Path, env_file: Path | None = None) -> str:
    """Load the agent key from the process environment or the local private file."""
    environment_key = os.environ.get("GLEE_API_KEY")
    if environment_key:
        return _validate_api_key(environment_key)
    path = env_file or project_root / "private" / "glee.env"
    file_key = _read_env_value(path, "GLEE_API_KEY")
    if file_key:
        return _validate_api_key(file_key)
    raise GleeConfigurationError(f"GLEE_API_KEY is absent; set it in the environment or {path}")


def load_glee_api_key_file(env_file: Path) -> str:
    """Load one explicitly selected agent key without consulting process-wide state."""
    file_key = _read_env_value(env_file, "GLEE_API_KEY")
    if file_key:
        return _validate_api_key(file_key)
    raise GleeConfigurationError(f"GLEE_API_KEY is absent from the explicit credential file {env_file}")


def glee_stats(project_root: Path, env_file: Path | None = None) -> dict:
    """Authenticate as the configured GLEE agent and return read-only account statistics."""
    client = GleeClient(api_key=load_glee_api_key(project_root, env_file))
    return client.stats()
