"""Command-line entry point for the isolated sequence-twin lab."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from .analysis import compare_account_ablation, compare_experiments
from .conditional_experiment import ConditionalTrainingConfig, run_conditional_experiment
from .conditional_release import ConditionalReleaseBuilder, run_conditional_smoke
from .corpus import SequenceCorpusBuilder
from .experiment import TrainingConfig, run_experiment
from .live_conditional import serve_live_conditional
from .live_shadow import serve_live_shadow
from .meta_controller_replay_v15 import MetaControllerV15Replay
from .model import ModelConfig
from .pre_terra_conditional_v2 import ConditionalCorpusBuilder
from .pre_terra_v3 import PreTerraV3CorpusBuilder
from .post_planner_audit import run_wording_sensitivity_audit
from .persuasion_buyer_continuation import BuyerContinuationTrainingConfig, PersuasionBuyerContinuationCorpusBuilder, freeze_persuasion_buyer_continuation_release, run_persuasion_buyer_continuation_experiment
from .shadow import ShadowCandidateBuilder, run_shadow_smoke
from .self_mirror import PublicSelfMirrorCorpusBuilder
from .self_mirror_live import run_public_self_mirror_smoke, serve_public_self_mirror
from .self_mirror_release import PublicSelfMirrorReleaseBuilder
from .selector_training import train_local_selector
from .selector_evaluation import evaluate_local_selector
from .synthetic import SyntheticPolicyZooBuilder
from .v3_fusion import V3TrainingConfig, run_v3_fusion_suite


LAB_ROOT = Path(__file__).resolve().parents[2]
NOMMD_ARENA_ROOT = LAB_ROOT.parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glee-sequence-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("build-corpus", help="freeze a verified GLEE frontier into normalized game, event, and target tables")
    corpus.add_argument("--analytics-root", type=Path, default=NOMMD_ARENA_ROOT / "runs" / "glee-analytics-online-v1")
    corpus.add_argument("--archive-root", type=Path, default=NOMMD_ARENA_ROOT / "runs")
    corpus.add_argument("--account-groups", type=Path, default=NOMMD_ARENA_ROOT / "reports" / "glee-account-linkage-v1-20260814" / "groups.json")
    corpus.add_argument("--output-dir", type=Path, required=True)
    corpus.add_argument("--train-fraction", type=float, default=0.70)
    corpus.add_argument("--validation-fraction", type=float, default=0.15)
    self_mirror = subparsers.add_parser("build-public-self-mirror-corpus", help="derive DeepRMM-01 self-action targets with strictly opponent-observable inputs")
    self_mirror.add_argument("--source-corpus", type=Path, required=True)
    self_mirror.add_argument("--output-dir", type=Path, required=True)
    freeze_self_mirror = subparsers.add_parser("freeze-public-self-mirror", help="seal independently seeded public self-mirror arms into one portable ensemble")
    freeze_self_mirror.add_argument("--experiment-dir", type=Path, action="append", required=True)
    freeze_self_mirror.add_argument("--output-dir", type=Path, required=True)
    freeze_self_mirror.add_argument("--release-id", required=True)
    self_mirror_smoke = subparsers.add_parser("public-self-mirror-smoke", help="verify a public self-mirror release across categorical and proposal targets")
    self_mirror_smoke.add_argument("--release-dir", type=Path, required=True)
    self_mirror_smoke.add_argument("--corpus-dir", type=Path, required=True)
    self_mirror_smoke.add_argument("--output", type=Path, required=True)
    self_mirror_smoke.add_argument("--device")
    self_mirror_serve = subparsers.add_parser("public-self-mirror-serve", help="serve public self-expectedness scoring over a credential-free Unix socket")
    self_mirror_serve.add_argument("--release-dir", type=Path, required=True)
    self_mirror_serve.add_argument("--registry", type=Path, required=True)
    self_mirror_serve.add_argument("--socket", type=Path, required=True)
    self_mirror_serve.add_argument("--status", type=Path, required=True)
    self_mirror_serve.add_argument("--device")
    train = subparsers.add_parser("train", help="train one frozen population or hierarchical experiment arm")
    train.add_argument("--corpus-dir", type=Path, action="append", required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--arm", choices=("population-gru", "hierarchical-gru", "population-transformer", "hierarchical-transformer", "population-mamba2", "hierarchical-mamba2", "population-mamba3-siso", "hierarchical-mamba3-siso"), required=True)
    train.add_argument("--label")
    train.add_argument("--seed", type=int, default=1729)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--evaluation-batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=8e-4)
    train.add_argument("--account-dropout", type=float, default=0.35)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--model-dim", type=int, default=96)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--event-streams", choices=("shared", "separate-head-gated"), default="shared")
    train.add_argument("--message-model-dim", type=int, default=32)
    train.add_argument("--message-hidden-dim", type=int, default=48)
    train.add_argument("--message-gate-max", type=float, default=0.5)
    train.add_argument("--delay-message-mode", choices=("gated", "base-only"), default="gated")
    train.add_argument("--no-mixed-precision", action="store_true")
    train.add_argument("--train-source-type", choices=("real", "synthetic"), action="append")
    train.add_argument("--initial-checkpoint", type=Path)
    train.add_argument("--account-disjoint-fold", type=int, choices=range(5))
    train.add_argument("--history-window", type=int)
    train.add_argument("--mask-message-inputs", action="store_true")
    train.add_argument("--equal-family-weighting", action="store_true", help="give each family equal total optimizer-step weight despite unequal target counts")
    train.add_argument("--selection-objective", choices=("action", "joint-self-mirror"), default="action")
    train.add_argument("--cuda-memory-fraction", type=float)
    synthetic = subparsers.add_parser("build-synthetic", help="build a train-only mechanics-respecting synthetic policy zoo")
    synthetic.add_argument("--output-dir", type=Path, required=True)
    synthetic.add_argument("--games-per-family", type=int, default=1_500)
    synthetic.add_argument("--seed", type=int, default=271_828)
    compare = subparsers.add_parser("compare", help="run paired complete-game bootstrap comparisons across completed arms")
    compare.add_argument("--baseline-dir", type=Path, required=True)
    compare.add_argument("--arm-dir", type=Path, action="append", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--bootstrap-replicates", type=int, default=2_000)
    compare.add_argument("--seed", type=int, default=81_077)
    account_ablation = subparsers.add_parser("compare-account-ablation", help="compare a hierarchical checkpoint with its forced-population inference path")
    account_ablation.add_argument("--experiment-dir", type=Path, required=True)
    account_ablation.add_argument("--output-dir", type=Path, required=True)
    account_ablation.add_argument("--bootstrap-replicates", type=int, default=2_000)
    account_ablation.add_argument("--seed", type=int, default=81_077)
    freeze_shadow = subparsers.add_parser("freeze-shadow-candidate", help="seal matched completed arms as one portable action-only shadow ensemble")
    freeze_shadow.add_argument("--experiment-dir", type=Path, action="append", required=True)
    freeze_shadow.add_argument("--output-dir", type=Path, required=True)
    freeze_shadow.add_argument("--candidate-id", required=True)
    shadow_smoke = subparsers.add_parser("shadow-smoke", help="verify frozen shadow loading and single-prefix inference without registering historical predictions")
    shadow_smoke.add_argument("--release-dir", type=Path, required=True)
    shadow_smoke.add_argument("--corpus-dir", type=Path, required=True)
    shadow_smoke.add_argument("--output", type=Path, required=True)
    shadow_smoke.add_argument("--warmup", type=int, default=3)
    shadow_smoke.add_argument("--repetitions", type=int, default=20)
    shadow_serve = subparsers.add_parser("shadow-serve", help="serve pre-Terra visible-prefix shadow inference over a credential-free Unix socket")
    shadow_serve.add_argument("--release-dir", type=Path, required=True)
    shadow_serve.add_argument("--registry", type=Path, required=True)
    shadow_serve.add_argument("--socket", type=Path, required=True)
    shadow_serve.add_argument("--status", type=Path, required=True)
    shadow_serve.add_argument("--device")
    conditional_serve = subparsers.add_parser("conditional-serve", help="serve activated post-planner candidate-response inference over a credential-free Unix socket")
    conditional_serve.add_argument("--release-dir", type=Path, required=True)
    conditional_serve.add_argument("--activation", type=Path, required=True)
    conditional_serve.add_argument("--socket", type=Path, required=True)
    conditional_serve.add_argument("--status", type=Path, required=True)
    conditional_serve.add_argument("--device")
    conditional_serve.add_argument("--runtime-manifest", type=Path)
    conditional_serve.add_argument("--buyer-continuation-release", type=Path)
    conditional_serve.add_argument("--buyer-continuation-registry", type=Path)
    pre_terra_v3 = subparsers.add_parser("build-pre-terra-v3-corpus", help="freeze exact archived pre-Terra feature snapshots and optionally attach aligned sequence forecasts")
    pre_terra_v3.add_argument("--source-run", type=Path, action="append", required=True)
    pre_terra_v3.add_argument("--sequence-release", type=Path, help="aligned pre-Terra sequence release; omit when building the core training corpus")
    pre_terra_v3.add_argument("--reference-core-corpus", type=Path, help="core corpus whose artifact hashes the sequence-augmented rebuild must reproduce")
    pre_terra_v3.add_argument("--account-groups", type=Path, default=NOMMD_ARENA_ROOT / "reports" / "glee-account-linkage-v1-20260814" / "groups.json")
    pre_terra_v3.add_argument("--output-dir", type=Path, required=True)
    pre_terra_v3.add_argument("--train-fraction", type=float, default=0.70)
    pre_terra_v3.add_argument("--validation-fraction", type=float, default=0.15)
    pre_terra_v3.add_argument("--prediction-batch-size", type=int, default=256)
    fusion_v3 = subparsers.add_parser("train-pre-terra-v3", help="train and compare sequence calibration, engineered-only, and bounded late-fusion arms")
    fusion_v3.add_argument("--corpus-dir", type=Path, required=True)
    fusion_v3.add_argument("--output-dir", type=Path, required=True)
    fusion_v3.add_argument("--bootstrap-replicates", type=int, default=2_000)
    conditional_v2 = subparsers.add_parser("build-pre-terra-conditional-v2-corpus", help="append the observed self-action bridge and mask its pre-Terra-unavailable fields")
    conditional_v2.add_argument("--source-corpus", type=Path, required=True)
    conditional_v2.add_argument("--output-dir", type=Path, required=True)
    conditional_train = subparsers.add_parser("train-pre-terra-conditional-v2", help="train the conditioned engineered expert and fit the frozen per-family convex stack")
    conditional_train.add_argument("--corpus-dir", type=Path, required=True)
    conditional_train.add_argument("--sequence-release", type=Path, required=True)
    conditional_train.add_argument("--output-dir", type=Path, required=True)
    conditional_train.add_argument("--epochs", type=int, default=80)
    conditional_train.add_argument("--patience", type=int, default=8)
    conditional_freeze = subparsers.add_parser("freeze-pre-terra-conditional-v2", help="seal the conditional experts and validation-fitted stack as a behaviorally inert release")
    conditional_freeze.add_argument("--experiment-dir", type=Path, required=True)
    conditional_freeze.add_argument("--sequence-release", type=Path, required=True)
    conditional_freeze.add_argument("--corpus-dir", type=Path, required=True)
    conditional_freeze.add_argument("--output-dir", type=Path, required=True)
    conditional_freeze.add_argument("--candidate-id", required=True)
    conditional_smoke = subparsers.add_parser("conditional-shadow-smoke", help="verify candidate substitution and conditional release inference without policy influence")
    conditional_smoke.add_argument("--release-dir", type=Path, required=True)
    conditional_smoke.add_argument("--corpus-dir", type=Path, required=True)
    conditional_smoke.add_argument("--output", type=Path, required=True)
    buyer_continuation = subparsers.add_parser("build-persuasion-buyer-continuation", help="derive buyer-action-to-next-seller-signal targets and an isolated Fieldglass augmentation arm")
    buyer_continuation.add_argument("--source-corpus", type=Path, required=True)
    buyer_continuation.add_argument("--output-dir", type=Path, required=True)
    buyer_continuation.add_argument("--account-groups", type=Path, default=NOMMD_ARENA_ROOT / "reports" / "glee-account-linkage-v1-20260814" / "groups.json")
    buyer_continuation.add_argument("--fieldglass-root", type=Path, action="append", default=[])
    buyer_continuation_train = subparsers.add_parser("train-persuasion-buyer-continuation", help="train reversed Persuasion heads on frozen Mamba-2 backbones and gate Fieldglass by DeepRMM-only validation")
    buyer_continuation_train.add_argument("--corpus-dir", type=Path, required=True)
    buyer_continuation_train.add_argument("--sequence-release", type=Path, required=True)
    buyer_continuation_train.add_argument("--output-dir", type=Path, required=True)
    buyer_continuation_train.add_argument("--epochs", type=int, default=24)
    buyer_continuation_train.add_argument("--patience", type=int, default=4)
    buyer_continuation_train.add_argument("--batch-size", type=int, default=256)
    buyer_continuation_train.add_argument("--evaluation-batch-size", type=int, default=512)
    buyer_continuation_train.add_argument("--learning-rate", type=float, default=3e-3)
    buyer_continuation_train.add_argument("--bootstrap-replicates", type=int, default=2_000)
    buyer_continuation_train.add_argument("--no-mixed-precision", action="store_true")
    buyer_continuation_freeze = subparsers.add_parser("freeze-persuasion-buyer-continuation", help="freeze the validation-selected reversed Persuasion heads without granting live authority")
    buyer_continuation_freeze.add_argument("--experiment-dir", type=Path, required=True)
    buyer_continuation_freeze.add_argument("--output-dir", type=Path, required=True)
    buyer_continuation_freeze.add_argument("--release-id", required=True)
    wording_audit = subparsers.add_parser("audit-post-planner-wording", help="measure same-polarity Persuasion wording sensitivity without claiming counterfactual accuracy")
    wording_audit.add_argument("--release-dir", type=Path, required=True)
    wording_audit.add_argument("--corpus-dir", type=Path, required=True)
    wording_audit.add_argument("--output-dir", type=Path, required=True)
    meta_replay = subparsers.add_parser("replay-meta-controller-v2-15", help="run the resumable 12-turn chronological replay of the 1.5-round Terra controller")
    meta_replay.add_argument("--corpus-dir", type=Path, default=NOMMD_ARENA_ROOT / "runs" / "glee-pre-terra-conditional-v2-corpus" / "ratings-v3-cut-20260816-core")
    meta_replay.add_argument("--release-dir", type=Path, required=True)
    meta_replay.add_argument("--run-dir", type=Path, required=True)
    meta_replay.add_argument("--model", default="gpt-5.6-terra")
    meta_replay.add_argument("--effort", default="high")
    meta_replay.add_argument("--timeout", type=int, default=600)
    meta_replay.add_argument("--max-workers", type=int, default=3)
    meta_replay.add_argument("--prepare-only", action="store_true")
    local_selector = subparsers.add_parser("train-local-selector", help="fit and freeze one inert family-balanced local selector over a sealed event-log frontier")
    local_selector.add_argument("--frontier", type=Path, required=True)
    local_selector.add_argument("--output-dir", type=Path, required=True)
    local_selector.add_argument("--release-id", required=True)
    local_selector_evaluation = subparsers.add_parser("evaluate-local-selector", help="open one post-frontier suffix against a previously frozen inert selector release")
    local_selector_evaluation.add_argument("--frontier", type=Path, required=True)
    local_selector_evaluation.add_argument("--release-dir", type=Path, required=True)
    local_selector_evaluation.add_argument("--output-dir", type=Path, required=True)
    subparsers.add_parser("gpu-smoke", help="verify the isolated PyTorch CUDA path with one forward and backward pass")
    subparsers.add_parser("mamba2-gpu-smoke", help="verify the fused Mamba-2 CUDA path with one mixed-precision forward and backward pass")
    return parser


def _gpu_smoke() -> dict[str, object]:
    available = torch.cuda.is_available()
    result: dict[str, object] = {"torch": torch.__version__, "cuda_available": available, "compiled_cuda": torch.version.cuda}
    if not available:
        result["status"] = "cuda-unavailable"
        return result
    device = torch.device("cuda")
    recurrent = torch.nn.GRU(input_size=16, hidden_size=32, batch_first=True).to(device)
    inputs = torch.randn(8, 12, 16, device=device)
    target = torch.randn(8, 12, 32, device=device)
    output, _state = recurrent(inputs)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    torch.cuda.synchronize()
    result.update(
        {
            "status": "passed",
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "loss": float(loss.detach().cpu()),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        }
    )
    return result


def _mamba2_gpu_smoke() -> dict[str, object]:
    result: dict[str, object] = {"torch": torch.__version__, "compiled_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        result["status"] = "cuda-unavailable"
        return result
    import causal_conv1d
    import causal_conv1d_cuda

    from .model import Mamba2SequenceCore

    device = torch.device("cuda")
    config = ModelConfig(core="mamba2")
    core = Mamba2SequenceCore(config).to(device)
    inputs = torch.randn(8, 64, config.model_dim, device=device, requires_grad=True)
    lengths = torch.full((inputs.shape[0],), inputs.shape[1], dtype=torch.long, device=device)

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        core.zero_grad(set_to_none=True)
        inputs.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            current_output = core(inputs, lengths)
            current_loss = current_output.square().mean()
        current_loss.backward()
        return current_output, current_loss

    warmup_started = time.perf_counter()
    output, loss = step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    timed_steps = 3
    timed_started = time.perf_counter()
    for _index in range(timed_steps):
        output, loss = step()
    torch.cuda.synchronize()
    step_milliseconds = (time.perf_counter() - timed_started) * 1_000 / timed_steps
    result.update(
        {
            "status": "passed",
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "architecture_list": torch.cuda.get_arch_list(),
            "causal_conv1d": causal_conv1d.__version__,
            "causal_conv1d_cuda": causal_conv1d_cuda.__file__,
            "fused_memory_efficient_path": all(block.use_mem_eff_path for block in core.blocks),
            "loss": float(loss.detach().cpu()),
            "output_shape": list(output.shape),
            "gradient_finite": bool(torch.isfinite(inputs.grad).all()),
            "warmup_seconds": warmup_seconds,
            "steady_step_milliseconds": step_milliseconds,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "gpu-smoke":
        print(json.dumps(_gpu_smoke(), indent=2, sort_keys=True))
        return 0
    if args.command == "mamba2-gpu-smoke":
        print(json.dumps(_mamba2_gpu_smoke(), indent=2, sort_keys=True))
        return 0
    if args.command == "build-corpus":
        result = SequenceCorpusBuilder(
            analytics_root=args.analytics_root,
            archive_root=args.archive_root,
            account_groups=args.account_groups,
            output_dir=args.output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        ).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-public-self-mirror-corpus":
        result = PublicSelfMirrorCorpusBuilder(source_corpus=args.source_corpus, output_dir=args.output_dir).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-public-self-mirror":
        result = PublicSelfMirrorReleaseBuilder(experiment_dirs=args.experiment_dir, output_dir=args.output_dir, release_id=args.release_id).run()
        print(json.dumps({"contract": result["contract"], "release_id": result["release_id"], "components": result["ensemble"]["components"], "manifest_sha256": result["manifest_sha256"], "output_dir": result["output_dir"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "public-self-mirror-smoke":
        result = run_public_self_mirror_smoke(release_dir=args.release_dir, corpus_dir=args.corpus_dir, output_path=args.output, device=args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "public-self-mirror-serve":
        serve_public_self_mirror(release_dir=args.release_dir, registry_path=args.registry, socket_path=args.socket, status_path=args.status, device=args.device)
        return 0
    if args.command == "train":
        training = TrainingConfig(
            arm=args.arm,
            label=args.label,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            learning_rate=args.learning_rate,
            account_dropout=args.account_dropout,
            workers=args.workers,
            mixed_precision=not args.no_mixed_precision,
            source_types=tuple(args.train_source_type or ("real",)),
            account_disjoint_fold=args.account_disjoint_fold,
            history_window=args.history_window,
            mask_message_inputs=args.mask_message_inputs,
            equal_family_weighting=args.equal_family_weighting,
            selection_objective=args.selection_objective,
            cuda_memory_fraction=args.cuda_memory_fraction,
        )
        model = ModelConfig(
            core=training.core,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            event_streams=args.event_streams,
            message_model_dim=args.message_model_dim,
            message_hidden_dim=args.message_hidden_dim,
            message_gate_max=args.message_gate_max,
            delay_message_mode=args.delay_message_mode,
        )
        result = run_experiment(corpus_dirs=args.corpus_dir, output_dir=args.output_dir, training=training, model_config=model, initial_checkpoint=args.initial_checkpoint)
        summary = {
            "contract": result["contract"],
            "arm": result["arm"],
            "label": result["label"],
            "best_epoch": result["best_epoch"],
            "best_validation_selection_score": result["best_validation_selection_score"],
            "test": result["evaluation"]["test"]["all"],
            "elapsed_seconds": result["elapsed_seconds"],
            "output_dir": str(args.output_dir.resolve()),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-synthetic":
        result = SyntheticPolicyZooBuilder(output_dir=args.output_dir, games_per_family=args.games_per_family, seed=args.seed).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        result = compare_experiments(baseline_dir=args.baseline_dir, arm_dirs=args.arm_dir, output_dir=args.output_dir, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
        print(json.dumps({"contract": result["contract"], "baseline": result["baseline"]["arm"], "arms": list(result["arms"]), "output_dir": str(args.output_dir.resolve())}, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-account-ablation":
        result = compare_account_ablation(experiment_dir=args.experiment_dir, output_dir=args.output_dir, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
        print(json.dumps({"contract": result["contract"], "comparison": result["comparison"], "output_dir": str(args.output_dir.resolve())}, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-shadow-candidate":
        result = ShadowCandidateBuilder(experiment_dirs=args.experiment_dir, output_dir=args.output_dir, candidate_id=args.candidate_id).run()
        print(json.dumps({"contract": result["contract"], "candidate_id": result["candidate_id"], "components": result["ensemble"]["components"], "manifest_sha256": result["manifest_sha256"], "output_dir": result["output_dir"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "shadow-smoke":
        result = run_shadow_smoke(release_dir=args.release_dir, corpus_dir=args.corpus_dir, output_path=args.output, warmup=args.warmup, repetitions=args.repetitions)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "shadow-serve":
        serve_live_shadow(release_dir=args.release_dir, registry_path=args.registry, socket_path=args.socket, status_path=args.status, device=args.device)
        return 0
    if args.command == "conditional-serve":
        serve_live_conditional(release_dir=args.release_dir, activation_path=args.activation, socket_path=args.socket, status_path=args.status, buyer_continuation_release_dir=args.buyer_continuation_release, buyer_continuation_registry_path=args.buyer_continuation_registry, device=args.device, runtime_manifest_path=args.runtime_manifest)
        return 0
    if args.command == "build-pre-terra-v3-corpus":
        result = PreTerraV3CorpusBuilder(
            source_runs=args.source_run,
            sequence_release=args.sequence_release,
            reference_core_corpus=args.reference_core_corpus,
            account_groups=args.account_groups,
            output_dir=args.output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            prediction_batch_size=args.prediction_batch_size,
        ).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train-pre-terra-v3":
        result = run_v3_fusion_suite(corpus_dir=args.corpus_dir, output_dir=args.output_dir, config=V3TrainingConfig(bootstrap_replicates=args.bootstrap_replicates))
        print(json.dumps({"contract": result["contract"], "promotion_gate": result["promotion_gate"], "output_dir": result["output_dir"], "result_sha256": result["result_sha256"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "build-pre-terra-conditional-v2-corpus":
        result = ConditionalCorpusBuilder(source_corpus=args.source_corpus, output_dir=args.output_dir).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train-pre-terra-conditional-v2":
        result = run_conditional_experiment(corpus_dir=args.corpus_dir, sequence_release=args.sequence_release, output_dir=args.output_dir, config=ConditionalTrainingConfig(epochs=args.epochs, patience=args.patience))
        print(json.dumps({"contract": result["contract"], "validation_winner": result["validation_winner"], "stack": result["stack"], "output_dir": result["output_dir"], "result_sha256": result["result_sha256"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-pre-terra-conditional-v2":
        result = ConditionalReleaseBuilder(experiment_dir=args.experiment_dir, sequence_release=args.sequence_release, corpus_dir=args.corpus_dir, output_dir=args.output_dir, candidate_id=args.candidate_id).run()
        print(json.dumps({"contract": result["contract"], "candidate_id": result["candidate_id"], "stack": result["stack"], "output_dir": result["output_dir"], "manifest_sha256": result["manifest_sha256"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "conditional-shadow-smoke":
        result = run_conditional_smoke(release_dir=args.release_dir, corpus_dir=args.corpus_dir, output_path=args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-persuasion-buyer-continuation":
        result = PersuasionBuyerContinuationCorpusBuilder(source_corpus=args.source_corpus, output_dir=args.output_dir, account_groups=args.account_groups, fieldglass_roots=args.fieldglass_root).run()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train-persuasion-buyer-continuation":
        result = run_persuasion_buyer_continuation_experiment(
            corpus_dir=args.corpus_dir,
            sequence_release=args.sequence_release,
            output_dir=args.output_dir,
            config=BuyerContinuationTrainingConfig(
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                evaluation_batch_size=args.evaluation_batch_size,
                learning_rate=args.learning_rate,
                bootstrap_replicates=args.bootstrap_replicates,
                mixed_precision=not args.no_mixed_precision,
            ),
        )
        print(json.dumps({"contract": result["contract"], "selected_arm": result["selected_arm"], "fieldglass_assessment": result["fieldglass_assessment"], "selected_test": result["selected_test"], "elapsed_seconds": result["elapsed_seconds"], "output_dir": result["output_dir"], "result_sha256": result["result_sha256"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "freeze-persuasion-buyer-continuation":
        result = freeze_persuasion_buyer_continuation_release(experiment_dir=args.experiment_dir, output_dir=args.output_dir, release_id=args.release_id)
        print(json.dumps({"contract": result["contract"], "release_id": result["release_id"], "selected_arm": result["selected_arm"], "status": result["status"], "output_dir": result["output_dir"], "manifest_sha256": result["manifest_sha256"]}, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-post-planner-wording":
        result = run_wording_sensitivity_audit(release_dir=args.release_dir, corpus_dir=args.corpus_dir, output_dir=args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "replay-meta-controller-v2-15":
        result = MetaControllerV15Replay(project_root=NOMMD_ARENA_ROOT, corpus_dir=args.corpus_dir, release_dir=args.release_dir, run_dir=args.run_dir, model=args.model, effort=args.effort, timeout_s=args.timeout, max_workers=args.max_workers).run(prepare_only=args.prepare_only)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "train-local-selector":
        result = train_local_selector(frontier_path=args.frontier, output_dir=args.output_dir, release_id=args.release_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-local-selector":
        result = evaluate_local_selector(frontier_path=args.frontier, release_dir=args.release_dir, output_dir=args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
