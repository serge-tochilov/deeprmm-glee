"""Hash-bound pure-PyTorch CPU runtime for the frozen public self-mirror."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import stat
import threading
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .corpus import file_sha256
from .cpu_reference import _CpuGatedRMSNorm, configure_cpu_threads, install_cpu_reference_kernels
from .self_mirror_live import PublicSelfMirrorAdapter, _UnixServer, _prepare_socket
from .self_mirror_release import PublicSelfMirrorRelease


SELF_MIRROR_CPU_RUNTIME_CONTRACT = "glee-public-self-mirror-cpu-reference-runtime-v1"
SELF_MIRROR_CPU_RUNTIME_STATUS = "promoted-live-cpu-reference"
SELF_MIRROR_CPU_EQUIVALENCE_CONTRACT = "glee-public-self-mirror-cpu-equivalence-v1"
SELF_MIRROR_CPU_EQUIVALENCE_STATUS = "passed-full-frozen-corpus"
SELF_MIRROR_CPU_BACKEND = "pure-pytorch-mamba2-reference"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def enable_self_mirror_cpu_reference(release: PublicSelfMirrorRelease, mamba2_type: type[nn.Module]) -> int:
    """Switch every frozen self-mirror Mamba-2 block to the reference CPU path."""
    if release.device.type != "cpu":
        raise ValueError("the self-mirror CPU reference path requires a CPU-loaded release")
    count = 0
    for _seed, model in release.models:
        for module in model.modules():
            if isinstance(module, mamba2_type):
                module.use_mem_eff_path = False
                if module.rmsnorm:
                    module.norm = _CpuGatedRMSNorm(module.norm)
                count += 1
    if count < 1:
        raise RuntimeError("the frozen public self-mirror contains no Mamba-2 blocks")
    return count


def build_self_mirror_cpu_release(release_dir: Path, *, intraop_threads: int = 1, interop_threads: int = 1) -> tuple[PublicSelfMirrorRelease, dict[str, object]]:
    """Load unchanged public self-mirror weights through the CPU reference execution path."""
    configure_cpu_threads(intraop_threads=intraop_threads, interop_threads=interop_threads)
    mamba2_type = install_cpu_reference_kernels()
    release = PublicSelfMirrorRelease(release_dir, device="cpu")
    blocks = enable_self_mirror_cpu_reference(release, mamba2_type)
    return release, {
        "contract": SELF_MIRROR_CPU_RUNTIME_CONTRACT,
        "backend": SELF_MIRROR_CPU_BACKEND,
        "device": "cpu",
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "mamba2_blocks": blocks,
        "torch_version": torch.__version__,
    }


def runtime_implementation_receipt() -> dict[str, str]:
    """Bind one promoted runtime to its exact implementation and environment lock."""
    root = Path(__file__).resolve().parents[2]
    return {
        "self_mirror_cpu_reference_sha256": file_sha256(Path(__file__)),
        "cpu_reference_sha256": file_sha256(Path(__file__).with_name("cpu_reference.py")),
        "self_mirror_live_sha256": file_sha256(Path(__file__).with_name("self_mirror_live.py")),
        "self_mirror_release_sha256": file_sha256(Path(__file__).with_name("self_mirror_release.py")),
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
    }


def validate_self_mirror_cpu_runtime(*, runtime_manifest_path: Path, release_dir: Path, require_transport_evidence: bool = True) -> dict[str, object]:
    """Verify one CPU runtime against its model, source, dependencies, and evidence."""
    runtime_path = runtime_manifest_path.resolve()
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    allowed = {SELF_MIRROR_CPU_RUNTIME_STATUS} if require_transport_evidence else {SELF_MIRROR_CPU_RUNTIME_STATUS, "candidate-awaiting-transport-smoke"}
    if runtime.get("contract") != SELF_MIRROR_CPU_RUNTIME_CONTRACT or runtime.get("status") not in allowed or runtime.get("backend") != SELF_MIRROR_CPU_BACKEND:
        raise ValueError("unsupported or unpromoted public self-mirror CPU runtime")
    release_path = release_dir.resolve() / "manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    model = _mapping(runtime.get("model_release"), name="self-mirror CPU model release")
    if model.get("release_id") != release.get("release_id") or model.get("manifest_sha256") != file_sha256(release_path):
        raise ValueError("self-mirror CPU runtime identifies a different frozen model")
    if dict(_mapping(runtime.get("implementation"), name="self-mirror CPU implementation")) != runtime_implementation_receipt():
        raise ValueError("self-mirror CPU runtime implementation hash mismatch")
    dependencies = {name: importlib.metadata.version(name) for name in ("torch", "mamba-ssm", "causal-conv1d", "triton")}
    if dict(_mapping(runtime.get("dependencies"), name="self-mirror CPU dependencies")) != dependencies:
        raise ValueError("self-mirror CPU runtime dependency versions changed")
    if _mapping(runtime.get("threads"), name="self-mirror CPU threads") != {"intraop": 1, "interop": 1}:
        raise ValueError("self-mirror CPU runtime is promoted only for one intra-operation and one inter-operation thread")
    evidence_receipt = _mapping(runtime.get("equivalence"), name="self-mirror CPU equivalence")
    evidence_path = (runtime_path.parent / str(evidence_receipt.get("path") or "")).resolve()
    evidence_path.relative_to(runtime_path.parent)
    if file_sha256(evidence_path) != evidence_receipt.get("sha256"):
        raise ValueError("self-mirror CPU equivalence hash mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("contract") != SELF_MIRROR_CPU_EQUIVALENCE_CONTRACT or evidence.get("status") != SELF_MIRROR_CPU_EQUIVALENCE_STATUS:
        raise ValueError("self-mirror CPU runtime lacks full-corpus equivalence evidence")
    if _mapping(evidence.get("model_release"), name="self-mirror equivalence model").get("manifest_sha256") != file_sha256(release_path):
        raise ValueError("self-mirror CPU equivalence covers a different model")
    observed_error = evidence.get("maximum_selector_absolute_error")
    allowed_error = runtime.get("maximum_selector_absolute_error")
    if not isinstance(observed_error, (int, float)) or isinstance(observed_error, bool) or not isinstance(allowed_error, (int, float)) or isinstance(allowed_error, bool) or float(observed_error) > float(allowed_error) or float(allowed_error) > 0.00001:
        raise ValueError("self-mirror CPU equivalence error exceeds its promoted boundary")
    observed_diagnostic = evidence.get("maximum_diagnostic_absolute_error")
    allowed_diagnostic = runtime.get("maximum_diagnostic_absolute_error")
    if not isinstance(observed_diagnostic, (int, float)) or isinstance(observed_diagnostic, bool) or not isinstance(allowed_diagnostic, (int, float)) or isinstance(allowed_diagnostic, bool) or float(observed_diagnostic) > float(allowed_diagnostic) or float(allowed_diagnostic) > 0.00005:
        raise ValueError("self-mirror CPU diagnostic-head equivalence error exceeds its promoted boundary")
    semantic = _mapping(evidence.get("semantic_equivalence"), name="self-mirror CPU semantic equivalence")
    if semantic.get("gate_passed") is not True or semantic.get("component_argmax_mismatches") != 0 or semantic.get("ensemble_top_choice_mismatch") != 0 or semantic.get("robust_pairwise_reversals") != 0 or semantic.get("robust_canonical_top_choice_reversal") != 0:
        raise ValueError("self-mirror CPU semantic equivalence gate did not pass")
    receipt = {
        "runtime_id": runtime.get("runtime_id"),
        "runtime_manifest_sha256": file_sha256(runtime_path),
        "equivalence_sha256": file_sha256(evidence_path),
        "backend": SELF_MIRROR_CPU_BACKEND,
        "device": "cpu",
        "intraop_threads": 1,
        "interop_threads": 1,
    }
    if not require_transport_evidence:
        return {**receipt, "transport_validation": "deferred-for-isolated-smoke-only"}
    transport_receipt = _mapping(runtime.get("transport_smoke"), name="self-mirror CPU transport smoke")
    transport_path = (runtime_path.parent / str(transport_receipt.get("path") or "")).resolve()
    transport_path.relative_to(runtime_path.parent)
    if file_sha256(transport_path) != transport_receipt.get("sha256"):
        raise ValueError("self-mirror CPU transport-smoke hash mismatch")
    transport = json.loads(transport_path.read_text(encoding="utf-8"))
    if transport.get("contract") != "glee-public-self-mirror-socket-smoke-v1" or transport.get("status") != "passed" or transport.get("release_id") != release.get("release_id") or transport.get("failure_count") != 0:
        raise ValueError("self-mirror CPU runtime lacks passing isolated transport evidence")
    transport_runtime = _mapping(transport.get("runtime"), name="self-mirror transport runtime")
    if transport_runtime.get("runtime_id") != runtime.get("runtime_id") or transport_runtime.get("equivalence_sha256") != file_sha256(evidence_path):
        raise ValueError("self-mirror transport evidence covers a different runtime")
    return {**receipt, "transport_smoke_sha256": file_sha256(transport_path)}


class CpuPublicSelfMirrorAdapter(PublicSelfMirrorAdapter):
    """Public self-mirror adapter with a validated 1-thread CPU reference backend."""

    def __init__(self, *, release_dir: Path, registry_path: Path, runtime_manifest_path: Path, require_transport_evidence: bool = True) -> None:
        promoted = validate_self_mirror_cpu_runtime(runtime_manifest_path=runtime_manifest_path, release_dir=release_dir, require_transport_evidence=require_transport_evidence)
        configure_cpu_threads(intraop_threads=1, interop_threads=1)
        mamba2_type = install_cpu_reference_kernels()
        super().__init__(release_dir=release_dir, registry_path=registry_path, device="cpu")
        blocks = enable_self_mirror_cpu_reference(self.release, mamba2_type)
        self.execution_runtime = {**promoted, "mamba2_blocks": blocks, "torch_version": torch.__version__}

    def status(self) -> dict[str, object]:
        return {**super().status(), "execution_runtime": dict(self.execution_runtime)}


def serve_self_mirror_cpu_reference(*, release_dir: Path, runtime_manifest_path: Path, registry_path: Path, socket_path: Path, status_path: Path, require_transport_evidence: bool = True) -> None:
    """Serve a verified self-mirror CPU runtime over its credential-free Unix socket."""
    resolved_socket = socket_path.resolve()
    _prepare_socket(resolved_socket)
    adapter = CpuPublicSelfMirrorAdapter(release_dir=release_dir, registry_path=registry_path, runtime_manifest_path=runtime_manifest_path, require_transport_evidence=require_transport_evidence)
    server = _UnixServer(str(resolved_socket), adapter, status_path.resolve())
    os.chmod(resolved_socket, 0o600)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def stop(_signal: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.publish_status()
        server.serve_forever(poll_interval=0.25)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        server.server_close()
        adapter.close()
        if resolved_socket.exists() and stat.S_ISSOCK(resolved_socket.stat().st_mode):
            resolved_socket.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--allow-candidate-runtime", action="store_true")
    arguments = parser.parse_args()
    serve_self_mirror_cpu_reference(release_dir=arguments.release_dir, runtime_manifest_path=arguments.runtime_manifest, registry_path=arguments.registry, socket_path=arguments.socket, status_path=arguments.status, require_transport_evidence=not arguments.allow_candidate_runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
