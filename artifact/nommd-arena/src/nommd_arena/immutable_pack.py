"""Deterministic immutable JSON-object packs addressed by SHA-256."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PACK_KIND = "gzip-jsonl-content-addressed-v1"


def canonical_json(value: object) -> str:
    """Return the repository's stable JSON representation."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def object_sha256(value: object) -> str:
    """Hash one object by its canonical UTF-8 JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one file without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError as error:
        raise ValueError(f"immutable pack must remain under its reference root: {resolved_path}") from error


def _resolve_relative(root: Path, relative: object) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / str(relative)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"immutable pack reference escapes its root: {relative}") from error
    return candidate


def seal_json_objects(*, root: Path, directory: Path, prefix: str, objects: Iterable[dict[str, Any]]) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    """Seal unique canonical objects in deterministic hash order and return one pack reference."""
    unique: dict[str, dict[str, Any]] = {}
    ordered_hashes: list[str] = []
    for value in objects:
        digest = object_sha256(value)
        previous = unique.setdefault(digest, value)
        if canonical_json(previous) != canonical_json(value):
            raise RuntimeError(f"SHA-256 collision while sealing immutable JSON object {digest}")
        ordered_hashes.append(digest)
    if not unique:
        raise ValueError("cannot seal an empty immutable object pack")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{prefix}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream:
                for digest in sorted(unique):
                    stream.write(canonical_json(unique[digest]).encode("utf-8") + b"\n")
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        pack_sha256 = file_sha256(temporary)
        path = directory / f"{prefix}-{pack_sha256[:20]}.jsonl.gz"
        if path.is_file():
            if file_sha256(path) != pack_sha256:
                raise RuntimeError(f"immutable pack path collision: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
        reference = {
            "kind": PACK_KIND,
            "path": _safe_relative(root, path),
            "pack_sha256": pack_sha256,
            "object_count": len(unique),
            "ordered_object_sha256": ordered_hashes,
        }
        return reference, unique
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=32)
def _read_pack_cached(path_text: str, expected_sha256: str) -> dict[str, dict[str, Any]]:
    path = Path(path_text)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"immutable pack failed SHA-256 verification: {path}")
    objects: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"immutable pack contains a non-object at {path}:{line_number}")
            digest = object_sha256(value)
            previous = objects.setdefault(digest, value)
            if canonical_json(previous) != canonical_json(value):
                raise RuntimeError(f"immutable pack contains a SHA-256 collision at {path}:{line_number}")
    return objects


def load_json_object(*, root: Path, reference: dict[str, Any], object_sha256_value: str) -> dict[str, Any]:
    """Load and verify one object from an immutable pack reference."""
    if reference.get("kind") != PACK_KIND:
        raise ValueError(f"unsupported immutable pack kind: {reference.get('kind')!r}")
    path = _resolve_relative(root, reference.get("path"))
    expected_pack_sha256 = str(reference.get("pack_sha256") or "")
    if len(expected_pack_sha256) != 64:
        raise ValueError("immutable pack reference has no valid SHA-256")
    objects = _read_pack_cached(str(path), expected_pack_sha256)
    try:
        value = objects[object_sha256_value]
    except KeyError as error:
        raise RuntimeError(f"immutable object {object_sha256_value} is absent from {path}") from error
    return json.loads(canonical_json(value))


def load_ordered_json_objects(*, root: Path, reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize the ordered object sequence named by a pack reference."""
    hashes = reference.get("ordered_object_sha256")
    if not isinstance(hashes, list) or any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ValueError("immutable pack reference has no valid ordered object hashes")
    expected_count = int(reference.get("object_count") or 0)
    if expected_count < 1 or expected_count != len(set(hashes)):
        raise ValueError("immutable pack reference has an invalid object count")
    return [load_json_object(root=root, reference=reference, object_sha256_value=digest) for digest in hashes]
