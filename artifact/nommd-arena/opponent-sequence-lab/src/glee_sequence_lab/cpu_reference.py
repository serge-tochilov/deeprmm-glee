"""Hash-validated pure-PyTorch CPU runtime for the frozen conditional twin."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .conditional_release import ConditionalTwinRelease
from .corpus import file_sha256
from .mamba_optional import load_mamba2


CPU_REFERENCE_RUNTIME_CONTRACT = "glee-post-planner-conditional-cpu-reference-runtime-v1"
CPU_REFERENCE_RUNTIME_STATUS = "promoted-live-cpu-reference"
CPU_REFERENCE_EQUIVALENCE_CONTRACT = "glee-post-planner-conditional-cpu-equivalence-v1"
CPU_REFERENCE_EQUIVALENCE_STATUS = "passed-full-frozen-corpus"
BUYER_CONTINUATION_EQUIVALENCE_CONTRACT = "glee-persuasion-buyer-continuation-cpu-equivalence-v1"
BUYER_CONTINUATION_EQUIVALENCE_STATUS = "passed-all-deeprmm-held-out-rows"
CPU_REFERENCE_BACKEND = "pure-pytorch-mamba2-reference"
_REFERENCE_INSTALLED = False
_MAMBA2_TYPE: type[nn.Module] | None = None


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


class _CpuGatedRMSNorm(nn.Module):
    """Pure-PyTorch equivalent of the fused gated RMS normalization used by Mamba-2."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.weight = source.weight
        self.eps = float(source.eps)
        self.group_size = source.group_size
        self.norm_before_gate = bool(source.norm_before_gate)

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        group_size = int(self.group_size or value.shape[-1])
        if value.shape[-1] % group_size:
            raise ValueError("RMSNorm group size does not divide the hidden width")
        shape = value.shape
        grouped = value.reshape(*shape[:-1], shape[-1] // group_size, group_size)
        grouped = grouped * torch.rsqrt(grouped.float().square().mean(dim=-1, keepdim=True) + self.eps).to(grouped.dtype)
        return grouped.reshape(shape) * self.weight

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_before_gate:
            normalized = self._normalize(x)
            return normalized if z is None else normalized * F.silu(z)
        gated = x if z is None else x * F.silu(z)
        return self._normalize(gated)


def install_cpu_reference_kernels() -> type[nn.Module]:
    """Install process-local CPU operations for the official Mamba-2 unfused path."""
    global _MAMBA2_TYPE, _REFERENCE_INSTALLED

    if _REFERENCE_INSTALLED and _MAMBA2_TYPE is not None:
        return _MAMBA2_TYPE
    mamba2_type = load_mamba2()
    import mamba_ssm.modules.mamba2 as mamba2_module
    from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref

    def reference_scan(
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        dt_softplus: bool = False,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        return_final_states: bool = False,
        return_varlen_states: bool = False,
        state_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del state_dtype
        if initial_states is not None or seq_idx is not None or cu_seqlens is not None or return_final_states or return_varlen_states:
            raise ValueError("the conditional CPU runtime does not support cached or variable-length Mamba state")
        if dt_limit != (0.0, float("inf")):
            raise ValueError("the conditional CPU runtime does not support a finite dt limit")
        sequence_length = x.shape[1]
        padding = (-sequence_length) % chunk_size
        if padding:
            x = F.pad(x, (0, 0, 0, 0, 0, padding))
            dt = F.pad(dt, (0, 0, 0, padding))
            B = F.pad(B, (0, 0, 0, 0, 0, padding))
            C = F.pad(C, (0, 0, 0, 0, 0, padding))
            if z is not None:
                z = F.pad(z, (0, 0, 0, 0, 0, padding))
        result = ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus)
        return result[:, :sequence_length]

    mamba2_module.causal_conv1d_fn = None
    mamba2_module.mamba_chunk_scan_combined = reference_scan
    _MAMBA2_TYPE = mamba2_type
    _REFERENCE_INSTALLED = True
    return mamba2_type


def configure_cpu_threads(*, intraop_threads: int, interop_threads: int) -> None:
    """Pin the compact serial service to explicit PyTorch CPU thread counts."""
    if intraop_threads < 1 or interop_threads < 1:
        raise ValueError("CPU thread counts must be positive")
    torch.set_num_threads(intraop_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        if torch.get_num_interop_threads() != interop_threads:
            raise


def enable_cpu_reference_path(release: ConditionalTwinRelease, mamba2_type: type[nn.Module]) -> int:
    """Switch every loaded Mamba-2 block to the process-local pure-PyTorch path."""
    if release.device.type != "cpu":
        raise ValueError("the CPU reference path requires a CPU-loaded conditional release")
    count = 0
    for _seed, model in release.sequence.models:
        for module in model.modules():
            if isinstance(module, mamba2_type):
                module.use_mem_eff_path = False
                if module.rmsnorm:
                    module.norm = _CpuGatedRMSNorm(module.norm)
                count += 1
    if count < 1:
        raise RuntimeError("the frozen conditional release contains no Mamba-2 blocks")
    return count


def build_cpu_reference_release(release_dir: Path, *, intraop_threads: int = 1, interop_threads: int = 1) -> tuple[ConditionalTwinRelease, dict[str, object]]:
    """Load unchanged frozen weights through the CPU reference execution path."""
    configure_cpu_threads(intraop_threads=intraop_threads, interop_threads=interop_threads)
    mamba2_type = install_cpu_reference_kernels()
    release = ConditionalTwinRelease(release_dir, device="cpu")
    mamba_blocks = enable_cpu_reference_path(release, mamba2_type)
    return release, {
        "contract": CPU_REFERENCE_RUNTIME_CONTRACT,
        "backend": CPU_REFERENCE_BACKEND,
        "device": str(release.device),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "mamba2_blocks": mamba_blocks,
        "torch_version": torch.__version__,
    }


def _runtime_implementation_receipt(*, live_conditional_path: Path, buyer_continuation_live_path: Path, cli_path: Path, uv_lock_path: Path) -> dict[str, str]:
    return {
        "cpu_reference_sha256": file_sha256(Path(__file__)),
        "live_conditional_sha256": file_sha256(live_conditional_path),
        "buyer_continuation_live_sha256": file_sha256(buyer_continuation_live_path),
        "cli_sha256": file_sha256(cli_path),
        "mamba_optional_sha256": file_sha256(Path(__file__).with_name("mamba_optional.py")),
        "uv_lock_sha256": file_sha256(uv_lock_path),
    }


def validate_cpu_runtime_manifest(*, runtime_manifest_path: Path, release_dir: Path, buyer_continuation_release_dir: Path, live_conditional_path: Path, buyer_continuation_live_path: Path, cli_path: Path, uv_lock_path: Path, require_transport_evidence: bool = True) -> dict[str, object]:
    """Validate the promoted runtime against the exact model, source, dependencies, and equivalence evidence."""
    manifest_path = runtime_manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_statuses = {CPU_REFERENCE_RUNTIME_STATUS} if require_transport_evidence else {CPU_REFERENCE_RUNTIME_STATUS, "candidate-awaiting-transport-smoke"}
    if manifest.get("contract") != CPU_REFERENCE_RUNTIME_CONTRACT or manifest.get("status") not in allowed_statuses:
        raise ValueError("unsupported or unpromoted conditional CPU runtime")
    if manifest.get("backend") != CPU_REFERENCE_BACKEND:
        raise ValueError("conditional CPU runtime selects an unsupported backend")
    release_manifest_path = release_dir.resolve() / "manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    model = _require_mapping(manifest.get("model_release"), name="CPU runtime model release")
    if model.get("candidate_id") != release_manifest.get("candidate_id") or model.get("manifest_sha256") != file_sha256(release_manifest_path):
        raise ValueError("conditional CPU runtime identifies a different frozen model release")
    continuation_manifest_path = buyer_continuation_release_dir.resolve() / "manifest.json"
    continuation_manifest = json.loads(continuation_manifest_path.read_text(encoding="utf-8"))
    continuation = _require_mapping(manifest.get("buyer_continuation_release"), name="CPU runtime buyer-continuation release")
    if continuation.get("release_id") != continuation_manifest.get("release_id") or continuation.get("manifest_sha256") != file_sha256(continuation_manifest_path):
        raise ValueError("conditional CPU runtime identifies a different buyer-continuation release")
    implementation = _require_mapping(manifest.get("implementation"), name="CPU runtime implementation")
    observed_implementation = _runtime_implementation_receipt(live_conditional_path=live_conditional_path, buyer_continuation_live_path=buyer_continuation_live_path, cli_path=cli_path, uv_lock_path=uv_lock_path)
    if dict(implementation) != observed_implementation:
        raise ValueError("conditional CPU runtime implementation hash mismatch")
    dependencies = _require_mapping(manifest.get("dependencies"), name="CPU runtime dependencies")
    observed_dependencies = {name: importlib.metadata.version(name) for name in ("torch", "mamba-ssm", "causal-conv1d", "triton")}
    if dict(dependencies) != observed_dependencies:
        raise ValueError("conditional CPU runtime dependency versions changed")
    threads = _require_mapping(manifest.get("threads"), name="CPU runtime threads")
    if threads != {"intraop": 1, "interop": 1}:
        raise ValueError("conditional CPU runtime is promoted only for one intra-operation and one inter-operation thread")
    evidence_receipt = _require_mapping(manifest.get("equivalence"), name="CPU runtime equivalence evidence")
    evidence_path = (manifest_path.parent / str(evidence_receipt.get("path") or "")).resolve()
    evidence_path.relative_to(manifest_path.parent)
    if file_sha256(evidence_path) != evidence_receipt.get("sha256"):
        raise ValueError("conditional CPU runtime equivalence evidence hash mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("contract") != CPU_REFERENCE_EQUIVALENCE_CONTRACT or evidence.get("status") != CPU_REFERENCE_EQUIVALENCE_STATUS:
        raise ValueError("conditional CPU runtime lacks passing full-corpus equivalence evidence")
    if _require_mapping(evidence.get("model_release"), name="equivalence model release").get("manifest_sha256") != file_sha256(release_manifest_path):
        raise ValueError("conditional CPU equivalence evidence covers a different model release")
    release_corpus = _require_mapping(release_manifest.get("corpus"), name="conditional release corpus")
    evidence_corpus = _require_mapping(evidence.get("corpus"), name="equivalence corpus")
    if evidence_corpus.get("manifest_sha256") != release_corpus.get("manifest_sha256") or evidence_corpus.get("rows") != _require_mapping(evidence.get("coverage"), name="equivalence coverage").get("rows"):
        raise ValueError("conditional CPU equivalence evidence does not cover the complete frozen release corpus")
    observed_error = evidence.get("maximum_absolute_probability_error")
    promoted_error = manifest.get("maximum_probability_error")
    if not isinstance(observed_error, (int, float)) or isinstance(observed_error, bool) or not isinstance(promoted_error, (int, float)) or isinstance(promoted_error, bool):
        raise ValueError("conditional CPU runtime probability-error bounds are invalid")
    if float(observed_error) > float(promoted_error) or float(promoted_error) > 2e-6:
        raise ValueError("conditional CPU runtime equivalence error exceeds its promoted boundary")
    continuation_evidence_receipt = _require_mapping(manifest.get("buyer_continuation_equivalence"), name="CPU runtime buyer-continuation equivalence evidence")
    continuation_evidence_path = (manifest_path.parent / str(continuation_evidence_receipt.get("path") or "")).resolve()
    continuation_evidence_path.relative_to(manifest_path.parent)
    if file_sha256(continuation_evidence_path) != continuation_evidence_receipt.get("sha256"):
        raise ValueError("buyer-continuation CPU equivalence evidence hash mismatch")
    continuation_evidence = json.loads(continuation_evidence_path.read_text(encoding="utf-8"))
    if continuation_evidence.get("contract") != BUYER_CONTINUATION_EQUIVALENCE_CONTRACT or continuation_evidence.get("status") != BUYER_CONTINUATION_EQUIVALENCE_STATUS:
        raise ValueError("conditional CPU runtime lacks passing buyer-continuation equivalence evidence")
    continuation_model = _require_mapping(continuation_evidence.get("conditional_release"), name="buyer-continuation equivalence conditional release")
    continuation_head = _require_mapping(continuation_evidence.get("buyer_continuation_release"), name="buyer-continuation equivalence release")
    if continuation_model.get("manifest_sha256") != file_sha256(release_manifest_path) or continuation_head.get("manifest_sha256") != file_sha256(continuation_manifest_path):
        raise ValueError("buyer-continuation CPU equivalence evidence covers different frozen releases")
    continuation_error = continuation_evidence.get("maximum_absolute_probability_error")
    continuation_promoted_error = manifest.get("buyer_continuation_maximum_probability_error")
    if not isinstance(continuation_error, (int, float)) or isinstance(continuation_error, bool) or not isinstance(continuation_promoted_error, (int, float)) or isinstance(continuation_promoted_error, bool):
        raise ValueError("buyer-continuation CPU runtime probability-error bounds are invalid")
    if float(continuation_error) > float(continuation_promoted_error) or float(continuation_promoted_error) > 2e-6:
        raise ValueError("buyer-continuation CPU runtime equivalence error exceeds its promoted boundary")
    result = {
        "runtime_id": manifest.get("runtime_id"),
        "runtime_manifest_sha256": file_sha256(manifest_path),
        "equivalence_sha256": file_sha256(evidence_path),
        "buyer_continuation_equivalence_sha256": file_sha256(continuation_evidence_path),
        "backend": CPU_REFERENCE_BACKEND,
        "device": "cpu",
        "intraop_threads": 1,
        "interop_threads": 1,
    }
    if not require_transport_evidence:
        return {**result, "transport_validation": "deferred-for-isolated-smoke-only"}
    transport_receipt = _require_mapping(manifest.get("transport_smoke"), name="CPU runtime transport smoke")
    transport_path = (manifest_path.parent / str(transport_receipt.get("path") or "")).resolve()
    transport_path.relative_to(manifest_path.parent)
    if file_sha256(transport_path) != transport_receipt.get("sha256"):
        raise ValueError("conditional CPU runtime transport-smoke hash mismatch")
    transport = json.loads(transport_path.read_text(encoding="utf-8"))
    if transport.get("contract") != "glee-post-planner-conditional-cpu-transport-smoke-v1" or transport.get("status") != "passed" or transport.get("candidate_id") != release_manifest.get("candidate_id") or transport.get("buyer_continuation_release_id") != continuation_manifest.get("release_id") or transport.get("failure_count_increase") != 0:
        raise ValueError("conditional CPU runtime lacks passing isolated transport evidence")
    transport_runtime = _require_mapping(transport.get("runtime"), name="transport runtime")
    if transport_runtime.get("runtime_id") != manifest.get("runtime_id") or transport_runtime.get("equivalence_sha256") != file_sha256(evidence_path) or transport_runtime.get("buyer_continuation_equivalence_sha256") != file_sha256(continuation_evidence_path):
        raise ValueError("conditional CPU transport evidence covers a different runtime")
    return {**result, "transport_smoke_sha256": file_sha256(transport_path)}


def load_validated_cpu_reference_release(*, release_dir: Path, buyer_continuation_release_dir: Path, runtime_manifest_path: Path, live_conditional_path: Path, buyer_continuation_live_path: Path, cli_path: Path, uv_lock_path: Path, require_transport_evidence: bool = True) -> tuple[ConditionalTwinRelease, dict[str, object]]:
    """Validate a promoted runtime and load its unchanged frozen conditional model on CPU."""
    promoted = validate_cpu_runtime_manifest(runtime_manifest_path=runtime_manifest_path, release_dir=release_dir, buyer_continuation_release_dir=buyer_continuation_release_dir, live_conditional_path=live_conditional_path, buyer_continuation_live_path=buyer_continuation_live_path, cli_path=cli_path, uv_lock_path=uv_lock_path, require_transport_evidence=require_transport_evidence)
    release, execution = build_cpu_reference_release(release_dir, intraop_threads=1, interop_threads=1)
    return release, {**execution, **promoted}
