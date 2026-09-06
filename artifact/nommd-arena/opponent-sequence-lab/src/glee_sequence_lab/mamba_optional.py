"""Pinned official Mamba loaders isolated from the live GLEE environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.machinery
import os
import sys
import sysconfig
import types
from pathlib import Path
from typing import Any


MAMBA_SOURCE_REPOSITORY = "https://github.com/state-spaces/mamba.git"
MAMBA_SOURCE_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
_MAMBA2: type[Any] | None = None
_MAMBA3: type[Any] | None = None


def ensure_python_headers() -> str | None:
    configured = Path(sysconfig.get_path("include"))
    if (configured / "Python.h").is_file():
        return str(configured)
    pattern = f"cpython-{sys.version_info.major}.{sys.version_info.minor}*-linux-x86_64-gnu/include/python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = sorted((Path.home() / ".local" / "share" / "uv" / "python").glob(pattern))
    selected = next((candidate for candidate in candidates if (candidate / "Python.h").is_file()), None)
    if selected is None:
        return None
    current = os.environ.get("C_INCLUDE_PATH", "")
    values = [str(selected), *(value for value in current.split(os.pathsep) if value)]
    os.environ["C_INCLUDE_PATH"] = os.pathsep.join(dict.fromkeys(values))
    return str(selected)


def _package_without_eager_initializer() -> None:
    """Load official submodules while bypassing the package initializer's eager Mamba-3 import."""
    for name in tuple(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]
    distribution = importlib.metadata.distribution("mamba-ssm")
    package_path = distribution.locate_file("mamba_ssm")
    package = types.ModuleType("mamba_ssm")
    package.__path__ = [str(package_path)]
    package.__package__ = "mamba_ssm"
    package.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None, is_package=True)
    sys.modules["mamba_ssm"] = package


def load_mamba2() -> type[Any]:
    global _MAMBA2

    if _MAMBA2 is not None:
        return _MAMBA2
    ensure_python_headers()
    _package_without_eager_initializer()
    _MAMBA2 = importlib.import_module("mamba_ssm.modules.mamba2").Mamba2
    return _MAMBA2


def load_mamba3() -> type[Any]:
    global _MAMBA3

    if _MAMBA3 is not None:
        return _MAMBA3
    ensure_python_headers()
    _package_without_eager_initializer()
    unavailable_mimo = types.ModuleType("mamba_ssm.ops.tilelang.mamba3.mamba3_mimo")
    sys.modules[unavailable_mimo.__name__] = unavailable_mimo
    _MAMBA3 = importlib.import_module("mamba_ssm.modules.mamba3").Mamba3
    return _MAMBA3
