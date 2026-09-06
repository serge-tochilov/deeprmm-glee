"""Immutable compressed text blobs used by compact model-call receipts."""

from __future__ import annotations

import copy
import gzip
import hashlib
import os
import threading
import uuid
from pathlib import Path
from typing import Any


BLOB_KIND = "gzip-text-sha256-v1"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_text_blob(*, root: Path, value: str) -> dict[str, object]:
    """Store one UTF-8 string once under its uncompressed content hash."""
    root = root.resolve()
    digest = _sha_text(value)
    path = root / "llm-objects" / "sha256" / digest[:2] / f"{digest}.txt.gz"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as raw_stream:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream:
                    stream.write(value.encode("utf-8"))
                raw_stream.flush()
                os.fsync(raw_stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        if _sha_text(stream.read()) != digest:
            raise RuntimeError(f"immutable text blob failed content verification: {path}")
    return {
        "kind": BLOB_KIND,
        "path": str(path.relative_to(root)),
        "sha256": digest,
        "stored_sha256": _sha_file(path),
        "chars": len(value),
    }


def load_text_blob(*, root: Path, reference: dict[str, Any]) -> str:
    """Load one compressed text blob after path, stored-file, and content verification."""
    if reference.get("kind") != BLOB_KIND:
        raise ValueError(f"unsupported immutable text blob kind: {reference.get('kind')!r}")
    root = root.resolve()
    path = (root / str(reference.get("path") or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"immutable text blob reference escapes its root: {path}") from error
    if _sha_file(path) != reference.get("stored_sha256"):
        raise RuntimeError(f"immutable text blob failed stored SHA-256 verification: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = stream.read()
    if _sha_text(value) != reference.get("sha256") or len(value) != int(reference.get("chars") or -1):
        raise RuntimeError(f"immutable text blob failed content verification: {path}")
    return value


def reference_text_blob(*, root: Path, receipt_dir: Path, value: str) -> dict[str, object]:
    """Store text under one shared root and make its root portable from a receipt directory."""
    reference = store_text_blob(root=root, value=value)
    relative_root = os.path.relpath(root.resolve(), receipt_dir.resolve())
    reference["root_from_receipt"] = relative_root
    return reference


def load_referenced_text_blob(*, receipt_dir: Path, reference: dict[str, Any]) -> str:
    """Resolve a portable receipt-relative blob root and verify the referenced text."""
    relative_root = reference.get("root_from_receipt", ".")
    if not isinstance(relative_root, str) or Path(relative_root).is_absolute():
        raise ValueError("immutable text blob has an invalid receipt-relative root")
    root = (receipt_dir.resolve() / relative_root).resolve()
    return load_text_blob(root=root, reference=reference)


def externalize_call_request(record: dict[str, Any], *, log_path: Path, blob_root: Path | None = None) -> dict[str, Any]:
    """Replace exact request text with immutable refs while retaining stable request hashes."""
    compact = copy.deepcopy(record)
    request = compact.get("request")
    if not isinstance(request, dict):
        return compact
    root = blob_root.resolve() if blob_root is not None else log_path.parent.resolve()
    for name in ("system", "user"):
        value = request.pop(name, None)
        if isinstance(value, str):
            reference = reference_text_blob(root=root, receipt_dir=log_path.parent, value=value)
            expected = request.get(f"{name}_sha256")
            if expected is not None and expected != reference["sha256"]:
                raise RuntimeError(f"model-call {name} hash disagrees with its exact text")
            request[f"{name}_sha256"] = reference["sha256"]
            request[f"{name}_ref"] = reference
    return compact


def hydrate_call_request(record: dict[str, Any], *, log_path: Path) -> dict[str, Any]:
    """Materialize exact request text from refs while accepting legacy inline records."""
    hydrated = copy.deepcopy(record)
    request = hydrated.get("request")
    if not isinstance(request, dict):
        return hydrated
    for name in ("system", "user"):
        if isinstance(request.get(name), str):
            continue
        reference = request.get(f"{name}_ref")
        if isinstance(reference, dict):
            value = load_referenced_text_blob(receipt_dir=log_path.parent, reference=reference)
            if _sha_text(value) != request.get(f"{name}_sha256"):
                raise RuntimeError(f"model-call {name} reference disagrees with its receipt")
            request[name] = value
    return hydrated
