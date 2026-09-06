"""Deadline-aware parallel GLEE supervisor with stateless model workers."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import time
from collections import Counter
from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from glee_sdk import GleeAPIError
from requests import ConnectionError as RequestsConnectionError
from requests import Timeout as RequestsTimeout

from .glee import load_glee_api_key
from .glee_account_prompt import OpponentAccountPromptModelReader
from .glee_activity_scheduler import DEFAULT_MINIMUM_TARGET, DEFAULT_WINDOW_S, LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT, IndependentActivityScheduler
from .glee_api_limiter import APIPriority, AgentWideGleeAPIGateway, AgentWideGleeAPIRateLimiter, GleeAPIBudgetDeferred
from .glee_advisor_contracts import PERSUASION_ADVISOR_MODEL_VERSION
from .glee_bargaining_intervention import build_bargaining_v217_context, intervention_receipt
from .glee_collector_identity import sample_collector_timing_target
from .glee_conditional_twin import GleeConditionalTwinClient
from .glee_dossier import DossierBroker, DossierSnapshot
from .glee_live_policy import BargainingLivePolicyStore, NegotiationLivePolicyStore, PersuasionLivePolicyStore
from .glee_message_style import MessageStylePolicyStore, realize_message_style, sample_timing_persona_target
from .glee_policy import GLEE_FAMILIES, apply_deterministic_safeguards, normalize_action, safe_action
from .glee_persuasion_twin_v2 import project_persuasion_terminal_state
from .glee_sensor import GleeSensorReader, SENSOR_CONTRACT
from .glee_sequence_shadow import GleeSequenceShadowClient, terra_synthetic_feature_bundle
from .glee_selector_backend import TerraSelectorBackend
from .glee_self_mirror import GleePublicSelfMirrorClient
from .glee_statistical_package import OpponentStatisticalPackageReader, bargaining_submitted_offer_forecast, compact_opponent_decision_forecast
from .glee_statistical_overlay import LiveOpponentStatisticalPackageReader
from .glee_tactics import GlobalTacticLedger
from .glee_timing import OpponentTimingStore, TIMING_CONTRACT
from .glee_transport import NON_REPLAYING_POST_TRANSPORT_CONTRACT, NonReplayingGleeClient
from .glee_worker import CAPACITY_MODEL_CHAIN, NEGOTIATION_ADVISOR_MODEL_VERSION, CapacityFallbackGleeTurnWorker, GleeTurnWorker, MaxHighGleeTurnWorker, MetaControllerV15GleeTurnWorker, TurnEnvelope, WorkerDecision, prepare_worker_envelope, worker_payload
from .model_runner import ArenaCodexRunner


_BARGAINING_ADVISOR_FALLBACK_MODEL_VERSION = "bargaining-live-advisor-v2.17"
_NEGOTIATION_ADVISOR_FALLBACK_MODEL_VERSION = NEGOTIATION_ADVISOR_MODEL_VERSION
_PERSUASION_ADVISOR_FALLBACK_MODEL_VERSION = PERSUASION_ADVISOR_MODEL_VERSION
_MOVE_DELAY_CONCEALMENT_CONTRACT = "glee-move-delay-concealment-v2"
_MOVE_SUBMISSION_TRANSPORT_CONTRACT = "glee-move-submission-transport-v2"
_MOVE_SUBMISSION_RESOLVER_ALLOWANCE_S = 10.0
_TERMINAL_MOVE_RACE = object()
_API_DISPATCH_SMOOTHING_CONTRACT = "glee-api-dispatch-smoothing-v1"
_QUEUE_SAFETY_QUARANTINE_CONTRACT = "glee-server-queue-safety-quarantine-v1"
_QUEUE_SAFETY_QUARANTINE_FALLBACK_S = 60.0
_QUEUE_DISPATCH_AMBIGUITY_GRACE_S = 12.0
_QUEUE_DISPATCH_AMBIGUITY_EMPTY_CHECKS = 2
_QUEUE_SAFETY_RETRY_RE = re.compile(r"try again after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _process_memory_mib() -> dict[str, float]:
    fields: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        name, separator, raw_value = line.partition(":")
        if separator and name in {"VmRSS", "VmHWM"}:
            fields[name] = int(raw_value.split()[0])
    if "VmRSS" not in fields:
        raise RuntimeError("/proc/self/status omitted VmRSS")
    return {
        "rss_mib": round(fields["VmRSS"] / 1024.0, 3),
        "peak_rss_mib": round(fields.get("VmHWM", fields["VmRSS"]) / 1024.0, 3),
    }


def _recorded_run_peak_rss_mib(events_path: Path) -> float:
    peak = 0.0
    if not events_path.is_file():
        return peak
    with events_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("kind") != "memory_sample":
                continue
            for key in ("peak_rss_mib", "rss_mib"):
                value = event.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    peak = max(peak, float(value))
    return peak


def _is_rate_limit(error: BaseException) -> bool:
    return isinstance(error, GleeAPIError) and error.status_code == 429


def _is_deferred_api_control(error: BaseException) -> bool:
    return _is_rate_limit(error) or isinstance(error, GleeAPIBudgetDeferred)


def _is_transient_server_error(error: BaseException) -> bool:
    return isinstance(error, GleeAPIError) and error.status_code >= 500


def _is_game_not_active(error: BaseException) -> bool:
    """Recognize the benign race where the server closed a turn before our POST."""
    return isinstance(error, GleeAPIError) and error.status_code == 400 and "game is not active" in str(error.message or error).casefold()


def _queue_safety_quarantine(error: BaseException, *, now_unix: float | None = None) -> dict[str, object] | None:
    if not isinstance(error, GleeAPIError) or error.status_code != 403:
        return None
    message = str(error.message or error)
    if "last 3 games all timed out on its turn" not in message or "Queue joins are paused" not in message:
        return None
    now = time.time() if now_unix is None else float(now_unix)
    match = _QUEUE_SAFETY_RETRY_RE.search(message)
    parsed = False
    retry_at_unix = now + _QUEUE_SAFETY_QUARANTINE_FALLBACK_S
    if match is not None:
        try:
            retry_at_unix = datetime.fromisoformat(match.group(1).replace("Z", "+00:00")).timestamp()
            parsed = True
        except ValueError:
            pass
    retry_at_unix = max(now, retry_at_unix)
    return {
        "contract": _QUEUE_SAFETY_QUARANTINE_CONTRACT,
        "retry_at": datetime.fromtimestamp(retry_at_unix, tz=timezone.utc).isoformat(timespec="microseconds"),
        "retry_at_unix": retry_at_unix,
        "retry_time_parsed": parsed,
        "server_status_code": error.status_code,
        "server_code": error.code,
        "server_message": message,
    }


def _submitted_final_state(envelope: TurnEnvelope, action: dict[str, Any], result: dict[str, Any], *, response_time_ms: int | None = None) -> dict[str, Any]:
    """Build the terminal state already established by an accepted game-over move."""
    game = envelope.game
    final_result = deepcopy(result.get("result") or {})
    state = deepcopy(game.get("game_state") if isinstance(game.get("game_state"), dict) else {})
    if game.get("game_family") == "bargaining" and str(game.get("phase")) == "decision" and isinstance(action.get("decision"), str):
        offer = deepcopy(state.get("last_offer"))
        history = deepcopy(state.get("history")) if isinstance(state.get("history"), list) else []
        if isinstance(offer, dict):
            round_number = int(offer.get("round") or state.get("round") or len(history) + 1)
            represented = any(isinstance(entry, dict) and int(entry.get("round") or (entry.get("offer") or {}).get("round") or -1) == round_number for entry in history)
            if not represented:
                entry: dict[str, Any] = {
                    "round": round_number,
                    "proposer": offer.get("proposer") or state.get("proposer"),
                    "offer": offer,
                    "decision": action["decision"],
                }
                if response_time_ms is not None:
                    entry["response_time_ms"] = response_time_ms
                history.append(entry)
                state["history"] = history
    if game.get("game_family") == "negotiation" and str(game.get("phase")) == "decision" and isinstance(action.get("decision"), str):
        offer = deepcopy(state.get("last_offer"))
        history = deepcopy(state.get("history")) if isinstance(state.get("history"), list) else []
        if isinstance(offer, dict):
            round_number = int(offer.get("round") or state.get("round") or len(history) + 1)
            represented = any(isinstance(entry, dict) and isinstance(entry.get("offer"), dict) and int(entry.get("round") or entry["offer"].get("round") or -1) == round_number and str(entry["offer"].get("from_player") or "") == str(offer.get("from_player") or "") for entry in history)
            if not represented:
                entry = {"round": round_number, "offer": offer, "decided_by": game.get("your_player"), "decision": action["decision"]}
                if action["decision"] == "RejectOffer" and action.get("product_price") is not None:
                    entry["counteroffer"] = action["product_price"]
                if response_time_ms is not None:
                    entry["response_time_ms"] = response_time_ms
                history.append(entry)
                state["history"] = history
    state["phase"] = "completed"
    state["result"] = final_result
    final_state = {
        "game_id": game["game_id"],
        "game_family": game["game_family"],
        "your_player": game.get("your_player"),
        "opponent": deepcopy(game.get("opponent")),
        "game_state": state,
        "status": "completed",
        "result": final_result,
    }
    if game.get("game_family") == "persuasion" and str(game.get("phase")) == "buyer_decision" and isinstance(action.get("decision"), str):
        final_state, _projection = project_persuasion_terminal_state(final_state, submitted_action=action, response_time_ms=response_time_ms)
    return final_state


class ParallelGleeRun:
    """Own credentials and submissions while bounded workers solve immutable turns."""

    def __init__(
        self,
        *,
        project_root: Path,
        run_dir: Path,
        env_file: Path | None,
        model: str,
        effort: str,
        worker_policy: str = "capacity-chain",
        model_timeout_s: int = 108,
        high_effort: str = "high",
        bargaining_opening_policy: str = "rmm",
        turn_deadline_s: float = 120.0,
        emergency_margin_s: float = 12.0,
        poll_interval_s: float = 2.0,
        max_parallel: int = 3,
        max_games: int | None = 3,
        max_time_s: float | None = None,
        families: tuple[str, ...] = GLEE_FAMILIES,
        family_slots: dict[str, int] | None = None,
        unified_activity_pool: bool = False,
        stop_after_family: tuple[str, int] | None = None,
        memory_retrieval_limit: int = 12,
        memory_decay: float = 0.5,
        max_rss_mib: float | None = None,
        memory_sample_interval_s: float = 30.0,
        opponent_statistical_package_root: Path | None = None,
        opponent_account_model_root: Path | None = None,
        bargaining_live_policy_root: Path | None = None,
        negotiation_live_policy_root: Path | None = None,
        persuasion_live_policy_root: Path | None = None,
        message_style_policy_root: Path | None = None,
        bargaining_advisor_seed: Path | None = None,
        negotiation_advisor_seed: Path | None = None,
        persuasion_advisor_seed: Path | None = None,
        sensor_feed_root: Path | None = None,
        sensor_max_age_s: float = 8.0,
        isolated_drain_quiet_s: float | None = None,
        opponent_timing_root: Path | None = None,
        move_delay_min_s: float = 0.0,
        move_delay_max_s: float = 0.0,
        move_delay_scope: str = "all",
        activity_window_targets: dict[str, int] | None = None,
        activity_window_s: float = DEFAULT_WINDOW_S,
        activity_minimum_target: int = DEFAULT_MINIMUM_TARGET,
        api_rate_limit_state: Path | None = None,
        api_request_limit: int = 60,
        api_window_s: float = 60.0,
        api_move_reserve: int = 8,
        api_control_reserve: int = 8,
        admission_dispatch_min_spacing_s: float = 0.0,
        move_submission_min_spacing_s: float = 0.0,
        bargaining_rating_canary_seed: Path | None = None,
        bargaining_rating_reporter_root: Path | None = None,
        bargaining_rating_history_path: Path | None = None,
        rating_v3_model_root: Path | None = None,
        rating_v3_registry_path: Path | None = None,
        rating_v3_reporter_root: Path | None = None,
        rating_v3_history_path: Path | None = None,
        rating_v3_corrections_path: Path | None = None,
        sequence_shadow_socket: Path | None = None,
        sequence_shadow_timeout_s: float = 1.0,
        conditional_twin_socket: Path | None = None,
        conditional_twin_timeout_s: float = 3.0,
        self_mirror_socket: Path | None = None,
        self_mirror_timeout_s: float = 3.0,
        meta_controller_planner_timeout_s: float = 48.0,
        meta_controller_minimum_selector_budget_s: float = 12.0,
        client: Any | None = None,
        worker: Any | None = None,
        broker: DossierBroker | None = None,
        opponent_statistical_package_reader: OpponentStatisticalPackageReader | None = None,
        opponent_account_model_reader: OpponentAccountPromptModelReader | None = None,
        global_tactic_ledger: GlobalTacticLedger | None = None,
        bargaining_advisor: Any | None = None,
        negotiation_advisor: Any | None = None,
        persuasion_advisor: Any | None = None,
        bargaining_rating_canary: Any | None = None,
        rating_v3_advisory: Any | None = None,
        sequence_shadow_client: Any | None = None,
        conditional_twin_client: Any | None = None,
        self_mirror_client: Any | None = None,
        api_rate_limiter: AgentWideGleeAPIRateLimiter | None = None,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if max_games is not None and max_games < 1:
            raise ValueError("max_games must be positive when supplied")
        if max_time_s is not None and max_time_s <= 0:
            raise ValueError("max_time_s must be positive when supplied")
        if max_rss_mib is not None and max_rss_mib <= 0:
            raise ValueError("max_rss_mib must be positive when supplied")
        if memory_sample_interval_s <= 0:
            raise ValueError("memory_sample_interval_s must be positive")
        if sensor_max_age_s <= 0:
            raise ValueError("sensor_max_age_s must be positive")
        if sequence_shadow_timeout_s <= 0:
            raise ValueError("sequence-shadow timeout must be positive")
        if conditional_twin_timeout_s <= 0 or self_mirror_timeout_s <= 0 or meta_controller_planner_timeout_s <= 0 or meta_controller_minimum_selector_budget_s <= 0:
            raise ValueError("conditional-twin, self-mirror, and 1.5-round budgets must be positive")
        if move_delay_min_s < 0 or move_delay_max_s < move_delay_min_s:
            raise ValueError("move-delay bounds must satisfy 0 <= minimum <= maximum")
        if move_delay_scope not in {"all", "known-only", "hidden-only", "none"}:
            raise ValueError("move_delay_scope must be all, known-only, hidden-only, or none")
        if not math.isfinite(admission_dispatch_min_spacing_s) or admission_dispatch_min_spacing_s < 0:
            raise ValueError("admission-dispatch minimum spacing must be finite and non-negative")
        if not math.isfinite(move_submission_min_spacing_s) or move_submission_min_spacing_s < 0:
            raise ValueError("move-submission minimum spacing must be finite and non-negative")
        if admission_dispatch_min_spacing_s > 0 and not unified_activity_pool:
            raise ValueError("admission-dispatch smoothing requires unified activity pooling")
        if activity_window_targets is not None:
            if set(activity_window_targets) != set(families):
                raise ValueError("activity_window_targets must assign every selected family exactly once")
            if family_slots is None and not unified_activity_pool:
                raise ValueError("activity scheduling requires explicit family_slots as a concurrency ceiling")
            if family_slots is not None and unified_activity_pool:
                raise ValueError("unified activity pooling and fixed family slots are mutually exclusive")
        elif unified_activity_pool:
            raise ValueError("unified activity pooling requires independent activity-window targets")
        if unified_activity_pool and api_rate_limit_state is None and api_rate_limiter is None:
            raise ValueError("unified activity pooling requires an explicit agent-wide API rate-limit state")
        if not unified_activity_pool and (api_rate_limit_state is not None or api_rate_limiter is not None):
            raise ValueError("the agent-wide API rate limiter is restricted to unified activity pooling")
        if api_rate_limit_state is not None and api_rate_limiter is not None:
            raise ValueError("supply either an API rate-limit state path or an injected limiter, not both")
        if unified_activity_pool and sensor_feed_root is not None:
            raise ValueError("unified activity pooling requires direct in-process API sensing through its agent-wide limiter")
        if turn_deadline_s <= emergency_margin_s:
            raise ValueError("turn_deadline_s must exceed emergency_margin_s")
        if worker_policy not in {"single", "max-high", "bargaining-high", "bargaining-capacity-chain", "capacity-chain", "meta15", "collector-local"}:
            raise ValueError("worker_policy must be 'single', 'max-high', 'bargaining-high', 'bargaining-capacity-chain', 'capacity-chain', 'meta15', or 'collector-local'")
        if bargaining_opening_policy not in {"analytic-cold", "rmm"}:
            raise ValueError("bargaining_opening_policy must be 'analytic-cold' or 'rmm'")
        if model_timeout_s + emergency_margin_s > turn_deadline_s:
            raise ValueError("model_timeout_s plus emergency_margin_s cannot exceed turn_deadline_s")
        if worker_policy == "single" and model_timeout_s + emergency_margin_s >= turn_deadline_s:
            raise ValueError("single-call model timeout plus emergency margin must be smaller than turn_deadline_s")
        if bargaining_advisor_seed is not None and bargaining_advisor is not None:
            raise ValueError("supply either bargaining_advisor_seed or bargaining_advisor, not both")
        if negotiation_advisor_seed is not None and negotiation_advisor is not None:
            raise ValueError("supply either negotiation_advisor_seed or negotiation_advisor, not both")
        if persuasion_advisor_seed is not None and persuasion_advisor is not None:
            raise ValueError("supply either persuasion_advisor_seed or persuasion_advisor, not both")
        if opponent_statistical_package_root is not None and opponent_statistical_package_reader is not None:
            raise ValueError("supply either opponent_statistical_package_root or opponent_statistical_package_reader, not both")
        if opponent_account_model_root is not None and opponent_account_model_reader is not None:
            raise ValueError("supply either opponent_account_model_root or opponent_account_model_reader, not both")
        if bargaining_live_policy_root is not None and "bargaining" not in families:
            raise ValueError("a Bargaining live-policy root requires the Bargaining family")
        if negotiation_live_policy_root is not None and "negotiation" not in families:
            raise ValueError("a Negotiation live-policy root requires the Negotiation family")
        if persuasion_live_policy_root is not None and "persuasion" not in families:
            raise ValueError("a Persuasion live-policy root requires the Persuasion family")
        canary_paths = (bargaining_rating_canary_seed, bargaining_rating_reporter_root, bargaining_rating_history_path)
        if bargaining_rating_canary is not None and any(path is not None for path in canary_paths):
            raise ValueError("supply either a Bargaining rating canary or its 3 source paths, not both")
        if any(path is not None for path in canary_paths) and not all(path is not None for path in canary_paths):
            raise ValueError("Bargaining rating canary requires seed, reporter root, and rating-history path")
        rating_v3_paths = (rating_v3_model_root, rating_v3_registry_path, rating_v3_reporter_root, rating_v3_history_path, rating_v3_corrections_path)
        if rating_v3_advisory is not None and any(path is not None for path in rating_v3_paths):
            raise ValueError("supply either a rating v3 advisory or its 5 source paths, not both")
        if sequence_shadow_socket is not None and sequence_shadow_client is not None:
            raise ValueError("supply either a sequence-shadow socket or an injected client, not both")
        if conditional_twin_socket is not None and conditional_twin_client is not None:
            raise ValueError("supply either a conditional-twin socket or an injected client, not both")
        if self_mirror_socket is not None and self_mirror_client is not None:
            raise ValueError("supply either a public self-mirror socket or an injected client, not both")
        if worker_policy == "meta15" and conditional_twin_socket is None and conditional_twin_client is None:
            raise ValueError("the 1.5-round worker requires a conditional-twin socket or injected client")
        if worker_policy != "meta15" and (conditional_twin_socket is not None or conditional_twin_client is not None):
            raise ValueError("the conditional twin is restricted to the 1.5-round worker")
        if worker_policy != "meta15" and (self_mirror_socket is not None or self_mirror_client is not None):
            raise ValueError("the public self-mirror is restricted to the 1.5-round worker")
        if any(path is not None for path in rating_v3_paths) and not all(path is not None for path in rating_v3_paths):
            raise ValueError("rating v3 advisory requires model, registry, reporter, rating-history, and rating-effect-corrections paths")
        if (bargaining_advisor_seed is not None or bargaining_advisor is not None) and worker_policy not in {"bargaining-high", "bargaining-capacity-chain", "capacity-chain", "meta15"}:
            raise ValueError("the v2 bargaining advisor requires a Bargaining-specific worker policy")
        if (negotiation_advisor_seed is not None or negotiation_advisor is not None) and worker_policy not in {"capacity-chain", "meta15"}:
            raise ValueError("the v2 Negotiation advisor requires the all-family capacity-chain or 1.5-round worker policy")
        if (persuasion_advisor_seed is not None or persuasion_advisor is not None) and worker_policy not in {"capacity-chain", "meta15"}:
            raise ValueError("the v2 Persuasion advisor requires the all-family capacity-chain or 1.5-round worker policy")
        if not families or any(family not in GLEE_FAMILIES for family in families):
            raise ValueError(f"families must be drawn from {GLEE_FAMILIES}")
        if worker_policy == "bargaining-capacity-chain":
            if families != ("bargaining",):
                raise ValueError("the capacity fallback chain is restricted to a Bargaining-only run")
        if worker_policy in {"bargaining-capacity-chain", "capacity-chain"} and worker is None and (model, effort) != ("gpt-5.6-terra", "high"):
            raise ValueError("the capacity fallback chain requires Terra High as its primary model")
        if worker_policy == "meta15" and worker is None and (model, effort) not in {("gpt-5.6-terra", "high"), ("gpt-5.6-sol", "high")}:
            raise ValueError("the 1.5-round controller requires Terra High or Sol High")
        collector_worker_receipt: dict[str, object] | None = None
        if worker_policy == "collector-local":
            receipt = getattr(worker, "manifest_receipt", None)
            if callable(receipt):
                receipt = receipt()
            if not isinstance(receipt, Mapping) or receipt.get("contract") != "glee-collector-local-turn-worker-v1" or receipt.get("live_authority") is not False or receipt.get("cloud_calls") != 0:
                raise ValueError("collector-local mode requires an injected hash-pinned local collector worker with zero cloud authority")
            collector_worker_receipt = dict(receipt)
            if (model, effort) != ("local", "none"):
                raise ValueError("collector-local mode must identify its inference substrate as model='local' and effort='none'")
            if not unified_activity_pool or family_slots is not None or activity_window_targets is None or tuple(families) != tuple(GLEE_FAMILIES):
                raise ValueError("collector-local mode requires all 3 families and one unified independently scheduled activity pool")
            if max_parallel < 6:
                raise ValueError("collector-local mode requires at least 6 shared worker slots")
            if int(activity_window_targets["bargaining"]) != 8 * int(activity_window_targets["persuasion"]) or int(activity_window_targets["negotiation"]) != 3 * int(activity_window_targets["persuasion"]):
                raise ValueError("collector-local mode requires the accepted 8:3:1 family admission ratio")
            if activity_minimum_target != int(activity_window_targets["persuasion"]):
                raise ValueError("collector-local mode requires its explicit activity minimum to equal the 1-part Persuasion target")
            if message_style_policy_root is not None:
                raise ValueError("collector-local candidates are identity-rendered before forecasting and cannot use a post-selection message-style rewriter")
            if move_delay_scope != "all" or move_delay_min_s > 0.7 or move_delay_max_s < 26.0:
                raise ValueError("collector-local mode requires the full collector timing-policy range on every identity mode")
            if opponent_timing_root is None:
                raise ValueError("collector-local mode requires an explicit source-local opponent-timing root")
            collector_incompatible = (bargaining_live_policy_root, negotiation_live_policy_root, persuasion_live_policy_root, bargaining_advisor_seed, negotiation_advisor_seed, persuasion_advisor_seed, bargaining_advisor, negotiation_advisor, persuasion_advisor, opponent_statistical_package_root, opponent_statistical_package_reader, opponent_account_model_root, opponent_account_model_reader, bargaining_rating_canary_seed, bargaining_rating_reporter_root, bargaining_rating_history_path, bargaining_rating_canary, rating_v3_model_root, rating_v3_registry_path, rating_v3_reporter_root, rating_v3_history_path, rating_v3_corrections_path, rating_v3_advisory, sequence_shadow_socket, sequence_shadow_client, global_tactic_ledger)
            if any(value is not None for value in collector_incompatible):
                raise ValueError("collector-local mode cannot inherit cloud-agent advisors, memories, shadows, ratings, or tactic ledgers")
        if family_slots is not None:
            if set(family_slots) != set(families):
                raise ValueError("family_slots must assign every selected family exactly once")
            if any(isinstance(slots, bool) or not isinstance(slots, int) or slots < 1 for slots in family_slots.values()):
                raise ValueError("every family slot allocation must be a positive integer")
            if sum(family_slots.values()) != max_parallel:
                raise ValueError("family slot allocations must sum to max_parallel")
        if stop_after_family is not None:
            stop_family, stop_games = stop_after_family
            if stop_family not in families or isinstance(stop_games, bool) or not isinstance(stop_games, int) or stop_games < 1:
                raise ValueError("stop_after_family must name a selected family and a positive game count")
            if max_games is not None:
                raise ValueError("max_games and stop_after_family are mutually exclusive")
        self.project_root = project_root
        self.run_dir = run_dir
        self.model = model
        self.effort = effort
        self.worker_policy = worker_policy
        self.collector_worker_receipt = collector_worker_receipt
        self.model_timeout_s = model_timeout_s
        self.high_effort = high_effort
        self.bargaining_opening_policy = bargaining_opening_policy
        self.turn_deadline_s = turn_deadline_s
        self.emergency_margin_s = emergency_margin_s
        self.poll_interval_s = poll_interval_s
        self.max_parallel = max_parallel
        self.max_games = max_games
        self.max_time_s = max_time_s
        self.families = families
        self.family_slots = dict(family_slots) if family_slots is not None else None
        self.unified_activity_pool = bool(unified_activity_pool)
        self.stop_after_family = stop_after_family
        self.memory_retrieval_limit = memory_retrieval_limit
        self.memory_decay = memory_decay
        self.max_rss_mib = max_rss_mib
        self.memory_sample_interval_s = memory_sample_interval_s
        self.sensor_feed_root = sensor_feed_root.resolve() if sensor_feed_root is not None else None
        self.sensor_max_age_s = sensor_max_age_s
        self.sensor_reader = GleeSensorReader(self.sensor_feed_root, max_age_s=sensor_max_age_s) if self.sensor_feed_root is not None else None
        self.isolated_drain_quiet_s = isolated_drain_quiet_s if isolated_drain_quiet_s is not None else turn_deadline_s + sensor_max_age_s
        if self.isolated_drain_quiet_s <= 0:
            raise ValueError("isolated_drain_quiet_s must be positive")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.opponent_timing_root = (opponent_timing_root or project_root / "runs" / "glee-opponent-timing-v1").resolve()
        self.opponent_timing = OpponentTimingStore(self.opponent_timing_root, poll_resolution_s=poll_interval_s * 2.0)
        self.move_delay_min_s = move_delay_min_s
        self.move_delay_max_s = move_delay_max_s
        self.move_delay_scope = move_delay_scope
        self._delay_random = secrets.SystemRandom()
        self.activity_window_targets = dict(activity_window_targets) if activity_window_targets is not None else None
        self.activity_window_s = float(activity_window_s)
        self.activity_minimum_target = int(activity_minimum_target)
        self.activity_scheduler = IndependentActivityScheduler(state_path=self.run_dir / "activity-scheduler.json", targets=self.activity_window_targets, window_s=self.activity_window_s, minimum_target=self.activity_minimum_target, clock=time.time) if self.activity_window_targets is not None else None
        self.api_rate_limit_state = api_rate_limit_state.resolve() if api_rate_limit_state is not None else None
        self.api_rate_limiter = api_rate_limiter or (AgentWideGleeAPIRateLimiter(state_path=self.api_rate_limit_state, request_limit=api_request_limit, window_s=api_window_s, move_reserve=api_move_reserve, control_reserve=api_control_reserve) if self.api_rate_limit_state is not None else None)
        self.admission_dispatch_min_spacing_s = float(admission_dispatch_min_spacing_s)
        self.move_submission_min_spacing_s = float(move_submission_min_spacing_s)
        self.events_path = self.run_dir / "events.jsonl"
        self.drain_request_path = self.run_dir / "drain.requested"
        self.pause_request_path = self.run_dir / "pause.requested"
        self.pause_ack_path = self.run_dir / "pause.active.json"
        self.activity_target_request_path = self.run_dir / "activity-target.request.json"
        self.activity_target_ack_path = self.run_dir / "activity-target.ack.json"
        self._event_lock = threading.Lock()
        self._event_sequence = sum(1 for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()) if self.events_path.is_file() else 0
        self.manifest_path = self.run_dir / "manifest.json"
        self.bargaining_live_policy_root = bargaining_live_policy_root.resolve() if bargaining_live_policy_root is not None else None
        self.bargaining_live_policy_store = BargainingLivePolicyStore(root=self.bargaining_live_policy_root, assignments_path=self.run_dir / "bargaining-live-policy-assignments.jsonl") if self.bargaining_live_policy_root is not None else None
        self.negotiation_live_policy_root = negotiation_live_policy_root.resolve() if negotiation_live_policy_root is not None else None
        self.negotiation_live_policy_store = NegotiationLivePolicyStore(root=self.negotiation_live_policy_root, assignments_path=self.run_dir / "negotiation-live-policy-assignments.jsonl") if self.negotiation_live_policy_root is not None else None
        self.persuasion_live_policy_root = persuasion_live_policy_root.resolve() if persuasion_live_policy_root is not None else None
        self.persuasion_live_policy_store = PersuasionLivePolicyStore(root=self.persuasion_live_policy_root, assignments_path=self.run_dir / "persuasion-live-policy-assignments.jsonl") if self.persuasion_live_policy_root is not None else None
        self.message_style_policy_root = message_style_policy_root.resolve() if message_style_policy_root is not None else None
        self.message_style_policy_store = MessageStylePolicyStore(root=self.message_style_policy_root, assignments_path=self.run_dir / "message-style-assignments.jsonl") if self.message_style_policy_root is not None else None
        self.bargaining_advisor_seed_path: Path | None = None
        if bargaining_advisor_seed is not None:
            from .glee_bargaining_live_v2 import BargainingLiveAdvisorV2, install_seed as install_bargaining_seed

            self.bargaining_advisor_seed_path = self.run_dir / "bargaining-advisor-seed.json"
            install_bargaining_seed(bargaining_advisor_seed, self.bargaining_advisor_seed_path, project_root=self.project_root)
            bargaining_advisor = BargainingLiveAdvisorV2(seed_path=self.bargaining_advisor_seed_path, journal_path=self.run_dir / "bargaining-advisor-completions.jsonl", project_root=self.project_root)
        self.bargaining_advisor = bargaining_advisor
        self.bargaining_advisor_model_version = str((self.bargaining_advisor.manifest_receipt() if self.bargaining_advisor is not None else {}).get("model_version") or _BARGAINING_ADVISOR_FALLBACK_MODEL_VERSION)
        self.negotiation_advisor_seed_path: Path | None = None
        if negotiation_advisor_seed is not None:
            from .glee_negotiation_live_v2 import NegotiationLiveAdvisorV2, install_seed as install_negotiation_seed

            self.negotiation_advisor_seed_path = self.run_dir / "negotiation-advisor-seed.json"
            install_negotiation_seed(negotiation_advisor_seed, self.negotiation_advisor_seed_path, project_root=self.project_root)
            negotiation_advisor = NegotiationLiveAdvisorV2(seed_path=self.negotiation_advisor_seed_path, journal_path=self.run_dir / "negotiation-advisor-completions.jsonl", project_root=self.project_root)
        self.negotiation_advisor = negotiation_advisor
        self.negotiation_advisor_model_version = str((self.negotiation_advisor.manifest_receipt() if self.negotiation_advisor is not None else {}).get("model_version") or _NEGOTIATION_ADVISOR_FALLBACK_MODEL_VERSION)
        self.persuasion_advisor_seed_path: Path | None = None
        if persuasion_advisor_seed is not None:
            from .glee_persuasion_live_v2 import PersuasionLiveAdvisorV2, install_seed as install_persuasion_seed

            self.persuasion_advisor_seed_path = self.run_dir / "persuasion-advisor-seed.json"
            install_persuasion_seed(persuasion_advisor_seed, self.persuasion_advisor_seed_path, project_root=self.project_root)
            persuasion_advisor = PersuasionLiveAdvisorV2(seed_path=self.persuasion_advisor_seed_path, journal_path=self.run_dir / "persuasion-advisor-completions.jsonl", project_root=self.project_root)
        self.persuasion_advisor = persuasion_advisor
        self.persuasion_advisor_model_version = str((self.persuasion_advisor.manifest_receipt() if self.persuasion_advisor is not None else {}).get("model_version") or _PERSUASION_ADVISOR_FALLBACK_MODEL_VERSION)
        self.opponent_statistical_package_root = opponent_statistical_package_root.resolve() if opponent_statistical_package_root is not None else None
        self.opponent_statistical_package_reader = opponent_statistical_package_reader or (LiveOpponentStatisticalPackageReader(self.opponent_statistical_package_root) if self.opponent_statistical_package_root is not None else None)
        self.opponent_account_model_root = opponent_account_model_root.resolve() if opponent_account_model_root is not None else None
        self.opponent_account_model_reader = opponent_account_model_reader or (OpponentAccountPromptModelReader(self.opponent_account_model_root) if self.opponent_account_model_root is not None else None)
        if bargaining_rating_canary is not None or bargaining_rating_canary_seed is not None:
            if self.families != ("bargaining",):
                raise ValueError("the prospective rating canary is restricted to a Bargaining-only run")
            if self.bargaining_advisor is None:
                raise ValueError("the prospective rating canary requires the executable Bargaining advisor")
            if self.opponent_statistical_package_reader is None:
                raise ValueError("the prospective rating canary requires active statistical packages")
        if bargaining_rating_canary_seed is not None:
            from .glee_rating_canary import BargainingRatingCanary

            bargaining_rating_canary = BargainingRatingCanary(
                seed_path=bargaining_rating_canary_seed,
                protocol_path=self.project_root / "protocols" / "glee-bargaining-rating-canary-v1.md",
                registry_path=self.run_dir / "bargaining-rating-canary.sqlite3",
                reporter_root=bargaining_rating_reporter_root,
                history_path=bargaining_rating_history_path,
                package_reader=self.opponent_statistical_package_reader,
            )
        self.bargaining_rating_canary = bargaining_rating_canary
        if rating_v3_model_root is not None:
            from .glee_rating_v3_advisory import RatingV3Advisory

            rating_v3_advisory = RatingV3Advisory(model_root=rating_v3_model_root, registry_path=rating_v3_registry_path, reporter_root=rating_v3_reporter_root, history_path=rating_v3_history_path, corrections_path=rating_v3_corrections_path)
        self.rating_v3_advisory = rating_v3_advisory
        self.sequence_shadow_socket = sequence_shadow_socket.resolve() if sequence_shadow_socket is not None else None
        self.sequence_shadow_timeout_s = float(sequence_shadow_timeout_s)
        self.sequence_shadow_client = sequence_shadow_client or (GleeSequenceShadowClient(self.sequence_shadow_socket, timeout_s=self.sequence_shadow_timeout_s) if self.sequence_shadow_socket is not None else None)
        self.conditional_twin_socket = conditional_twin_socket.resolve() if conditional_twin_socket is not None else None
        self.conditional_twin_timeout_s = float(conditional_twin_timeout_s)
        self.conditional_twin_client = conditional_twin_client or (GleeConditionalTwinClient(self.conditional_twin_socket, timeout_s=self.conditional_twin_timeout_s) if self.conditional_twin_socket is not None else None)
        self.self_mirror_socket = self_mirror_socket.resolve() if self_mirror_socket is not None else None
        self.self_mirror_timeout_s = float(self_mirror_timeout_s)
        self.self_mirror_client = self_mirror_client or (GleePublicSelfMirrorClient(self.self_mirror_socket, timeout_s=self.self_mirror_timeout_s) if self.self_mirror_socket is not None else None)
        self.meta_controller_planner_timeout_s = float(meta_controller_planner_timeout_s)
        self.meta_controller_minimum_selector_budget_s = float(meta_controller_minimum_selector_budget_s)
        self.global_tactic_ledger = global_tactic_ledger if worker_policy == "collector-local" else global_tactic_ledger or GlobalTacticLedger(self.project_root / "tactics" / "glee-global-tactics.json")
        if client is None:
            base_url = os.environ.get("GLEE_API_URL")
            client_args: dict[str, Any] = {"api_key": load_glee_api_key(project_root, env_file), "timeout": 10}
            if base_url:
                client_args["base_url"] = base_url
            client = NonReplayingGleeClient(**client_args)
        self.client = client
        self.api_gateway = AgentWideGleeAPIGateway(client=self.client, limiter=self.api_rate_limiter) if self.api_rate_limiter is not None else None
        os.environ.pop("GLEE_API_KEY", None)
        initial_frontier = self._read_sensor_frontier()
        initial_stats = dict(initial_frontier["stats"]) if initial_frontier is not None else self._api_call(operation="stats", priority="background", callback=self.client.stats, wait=True)
        self.agent_name = str(initial_stats.get("agent_name") or "DeepRMM-01")
        self.broker = broker or DossierBroker(
            root=self.run_dir / "dossier-broker",
            agent_name=self.agent_name,
            retrieval_limit=memory_retrieval_limit,
            decay=memory_decay,
            opponent_statistical_package_reader=self.opponent_statistical_package_reader,
            global_tactic_ledger=self.global_tactic_ledger,
        )
        if worker is None:
            def runner_factory(timeout_s: int) -> Any:
                return ArenaCodexRunner(
                    prompts_dir=project_root / "prompts",
                    log_path=self.run_dir / "llm_calls.jsonl",
                    session_dir=self.run_dir / ".cli-session",
                    timeout_s=timeout_s,
                    validation_retries=0,
                )

            if worker_policy == "meta15":
                selector_backend_id = "terra-cli-v2-position-permuted" if model == "gpt-5.6-terra" else "sol-cli-v1-position-permuted"
                selector_backend = TerraSelectorBackend(runner_factory=runner_factory, model=model, effort=effort, backend_id=selector_backend_id)
                ineligible_model_chain = CAPACITY_MODEL_CHAIN if (model, effort) == ("gpt-5.6-terra", "high") else (("primary", model, effort),)
                worker = MetaControllerV15GleeTurnWorker(
                    runner_factory=runner_factory,
                    conditional_client=self.conditional_twin_client,
                    self_mirror_client=self.self_mirror_client,
                    model=model,
                    effort=effort,
                    model_timeout_s=model_timeout_s,
                    planner_timeout_s=self.meta_controller_planner_timeout_s,
                    finalization_margin_s=emergency_margin_s,
                    minimum_selector_budget_s=self.meta_controller_minimum_selector_budget_s,
                    selector_backend=selector_backend,
                    ineligible_model_chain=ineligible_model_chain,
                )
            elif worker_policy in {"bargaining-capacity-chain", "capacity-chain"}:
                worker = CapacityFallbackGleeTurnWorker(
                    runner_factory=runner_factory,
                    model_chain=CAPACITY_MODEL_CHAIN,
                    model_timeout_s=model_timeout_s,
                    finalization_margin_s=emergency_margin_s,
                )
            elif worker_policy in {"max-high", "bargaining-high"}:
                worker = MaxHighGleeTurnWorker(
                    runner_factory=runner_factory,
                    model=model,
                    max_effort=effort,
                    high_effort=high_effort,
                    model_timeout_s=model_timeout_s,
                    finalization_margin_s=emergency_margin_s,
                    bargaining_opening_policy=bargaining_opening_policy,
                    high_only_families=("bargaining",) if worker_policy == "bargaining-high" else (),
                )
            else:
                worker = GleeTurnWorker(model_runner=runner_factory(model_timeout_s), model=model, effort=effort, minimum_start_budget_s=model_timeout_s + emergency_margin_s)
        self.worker = worker
        self.initial_stats = initial_stats
        self._inflight: dict[str, tuple[Future[WorkerDecision], TurnEnvelope]] = {}
        self._waiting: dict[str, TurnEnvelope] = {}
        self._ready: dict[str, tuple[TurnEnvelope, WorkerDecision, dict[str, object]]] = {}
        self._transport_blocked_turns: set[str] = self.broker.transport_suspended_turn_ids()
        self._first_seen: dict[str, float] = {}
        self._queued_families: set[str] = set()
        self._ambiguous_queue_deadlines: dict[str, float] = {}
        self._ambiguous_queue_empty_checks: dict[str, int] = {}
        self._queue_cursor = 0
        self._draining = False
        self._paused = self.pause_ack_path.is_file()
        self._last_family_activity = time.monotonic()
        self._last_sensor_failure_event = float("-inf")
        self._last_topup = 0.0
        self._last_refresh = 0.0
        self._refresh_cursor = 0
        self._last_pause_leave_attempt = float("-inf")
        self._last_admission_dispatch_monotonic = float("-inf")
        self._admission_quarantine_until_unix = 0.0
        self._last_move_submission_release_monotonic = float("-inf")
        self._cached_stats = initial_stats
        self._last_stats_read = time.monotonic()
        self._cached_memory: dict[str, float] = {}
        self._run_peak_rss_mib = _recorded_run_peak_rss_mib(self.events_path)
        self._last_memory_sample = float("-inf")
        if self.activity_scheduler is not None:
            for receipt in self.activity_scheduler.recover_outstanding_after_restart():
                self._event("activity_outstanding_recovered", schedule=receipt)
            pause_receipt = self.activity_scheduler.set_paused(self._paused, now=time.time())
            if pause_receipt is not None:
                self._event("activity_scheduler_pause_changed", schedule=pause_receipt)
            if not self._paused:
                for receipt in self.activity_scheduler.rebase_overdue_after_restart(now=time.time()):
                    self._event("activity_schedule_rebased_after_restart", schedule=receipt)
        self._write_or_validate_manifest()

    def _mode(self) -> str:
        if self.worker_policy == "collector-local":
            return "glee-parallel-collector-local-v1"
        if self.admission_dispatch_min_spacing_s > 0 or self.move_submission_min_spacing_s > 0:
            return "glee-parallel-v38-smoothed-api-dispatch"
        if self.self_mirror_client is not None:
            return "glee-parallel-v37-meta15-public-self-mirror"
        if self.conditional_twin_client is not None:
            return "glee-parallel-v36-meta15"
        if self.sequence_shadow_client is not None:
            return "glee-parallel-v35"
        if self.rating_v3_advisory is not None:
            return "glee-parallel-v34"
        if self.opponent_account_model_reader is not None:
            return "glee-parallel-v33"
        if self.message_style_policy_store is not None:
            return "glee-parallel-v32"
        if self.persuasion_live_policy_store is not None:
            return "glee-parallel-v31"
        if self.negotiation_live_policy_store is not None:
            return "glee-parallel-v29"
        if self.bargaining_live_policy_store is not None:
            return "glee-parallel-v28"
        if self.bargaining_rating_canary is not None:
            return "glee-parallel-v27"
        if isinstance(self.opponent_statistical_package_reader, LiveOpponentStatisticalPackageReader):
            return "glee-parallel-v26"
        if self.opponent_statistical_package_reader is not None:
            return "glee-parallel-v25"
        if self.worker_policy == "single":
            return "glee-parallel-v1"
        if self.worker_policy == "capacity-chain":
            return "glee-parallel-v24" if self.persuasion_advisor is not None else "glee-parallel-v23" if self.negotiation_advisor is not None else "glee-parallel-v17"
        if self.worker_policy == "bargaining-capacity-chain":
            return "glee-parallel-v16"
        if self.worker_policy == "bargaining-high":
            return "glee-parallel-v15" if self.bargaining_advisor is not None else "glee-parallel-v14"
        return "glee-parallel-v5" if self.bargaining_opening_policy == "analytic-cold" else "glee-parallel-v13"

    def _write_or_validate_manifest(self) -> None:
        move_delay_manifest = {"contract": _MOVE_DELAY_CONCEALMENT_CONTRACT, "target_min_s": self.move_delay_min_s, "target_max_s": self.move_delay_max_s, "scope": self.move_delay_scope, "distribution": "collector-game-pinned-profile-quantile-with-per-move-jitter", "deadline_reserve_s": self.emergency_margin_s} if self.worker_policy == "collector-local" else {"contract": _MOVE_DELAY_CONCEALMENT_CONTRACT, "target_min_s": self.move_delay_min_s, "target_max_s": self.move_delay_max_s, **({"scope": self.move_delay_scope} if self.move_delay_scope != "all" else {}), "ki_distribution": "bounded-beta-2-2-total-latency", "hi_distribution": "game-pinned-joint-lexical-timing-persona-when-assigned", "deadline_reserve_s": self.emergency_margin_s}
        configuration = {
            "schema_version": 1,
            "mode": self._mode(),
            "model": self.model,
            "effort": self.effort,
            "worker_policy": self.worker_policy,
            "model_timeout_s": self.model_timeout_s,
            "high_effort": self.high_effort,
            "capacity_model_chain": self.worker.manifest_chain if isinstance(self.worker, CapacityFallbackGleeTurnWorker) else None,
            "capacity_retry_policy": {"primary_model": "gpt-5.6-terra", "maximum_same_model_retries": 1, "minimum_retry_budget_s": self.worker.minimum_retry_budget_s, "ineligible_failures": ["timeout", "capacity-without-zero-inference-proof"]} if isinstance(self.worker, CapacityFallbackGleeTurnWorker) else None,
            "meta_controller": self.worker.manifest_receipt if isinstance(self.worker, MetaControllerV15GleeTurnWorker) else None,
            "collector_turn_worker": self.collector_worker_receipt,
            "bargaining_opening_policy": self.bargaining_opening_policy,
            "turn_deadline_s": self.turn_deadline_s,
            "emergency_margin_s": self.emergency_margin_s,
            "poll_interval_s": self.poll_interval_s,
            "max_parallel": self.max_parallel,
            "max_games": self.max_games,
            "max_time_s": self.max_time_s,
            "families": list(self.families),
            "family_slots": {family: self.family_slots[family] for family in self.families} if self.family_slots is not None else None,
            "stop_after_family": {"family": self.stop_after_family[0], "games": self.stop_after_family[1]} if self.stop_after_family is not None else None,
            "memory_retrieval_limit": self.memory_retrieval_limit,
            "memory_decay": self.memory_decay,
            "max_rss_mib": self.max_rss_mib,
            "memory_sample_interval_s": self.memory_sample_interval_s,
            "opponent_statistical_package": self.opponent_statistical_package_reader.receipt if self.opponent_statistical_package_reader is not None else None,
            "opponent_account_model": self.opponent_account_model_reader.receipt if self.opponent_account_model_reader is not None else None,
            "bargaining_live_policy": self.bargaining_live_policy_store.manifest_receipt() if self.bargaining_live_policy_store is not None else None,
            "negotiation_live_policy": self.negotiation_live_policy_store.manifest_receipt() if self.negotiation_live_policy_store is not None else None,
            "persuasion_live_policy": self.persuasion_live_policy_store.manifest_receipt() if self.persuasion_live_policy_store is not None else None,
            "message_style_policy": self.message_style_policy_store.manifest_receipt() if self.message_style_policy_store is not None else None,
            "global_tactic_ledger_sha256": self.global_tactic_ledger.sha256 if self.global_tactic_ledger is not None else None,
            "bargaining_advisor": self.bargaining_advisor.manifest_receipt() if self.bargaining_advisor is not None else None,
            "negotiation_advisor": self.negotiation_advisor.manifest_receipt() if self.negotiation_advisor is not None else None,
            "persuasion_advisor": self.persuasion_advisor.manifest_receipt() if self.persuasion_advisor is not None else None,
            "bargaining_rating_canary": self.bargaining_rating_canary.manifest_receipt() if self.bargaining_rating_canary is not None else None,
            "rating_v3_advisory": self.rating_v3_advisory.manifest_receipt() if self.rating_v3_advisory is not None else None,
            "sequence_shadow": self.sequence_shadow_client.receipt if self.sequence_shadow_client is not None else None,
            "conditional_twin": self.conditional_twin_client.receipt if self.conditional_twin_client is not None else None,
            "public_self_mirror": self.self_mirror_client.receipt if self.self_mirror_client is not None else None,
            "sensor_feed": {"contract": SENSOR_CONTRACT, "root": str(self.sensor_feed_root), "max_age_s": self.sensor_max_age_s} if self.sensor_feed_root is not None else None,
            "opponent_timing": {"contract": TIMING_CONTRACT, "root": str(self.opponent_timing_root), "poll_resolution_s": self.poll_interval_s * 2.0},
            "move_delay_concealment": move_delay_manifest,
            "move_submission_transport": {
                "contract": _MOVE_SUBMISSION_TRANSPORT_CONTRACT,
                "client_contract": NON_REPLAYING_POST_TRANSPORT_CONTRACT,
                "post_network_attempts": 1,
                "ambiguous_post_policy": "durably suspend only the affected turn without replay",
                "transient_server_policy": "treat 5xx move responses as ambiguous and contain them per turn",
            },
            "queue_safety_quarantine": {
                "contract": _QUEUE_SAFETY_QUARANTINE_CONTRACT,
                "policy": "restore the dispatched arrival, pause admissions until the server retry time, and keep active games alive",
            },
            "activity_scheduler": self.activity_scheduler.manifest_receipt() if self.activity_scheduler is not None else None,
            "queue_scope": "selected-families-v1",
            "pause_control": "run-directory-markers-v1",
            "isolated_drain_quiet_s": self.isolated_drain_quiet_s,
            "agent": {"agent_id": self.initial_stats.get("agent_id"), "agent_name": self.agent_name},
        }
        if self.unified_activity_pool:
            configuration["unified_activity_pool"] = True
            configuration["agent_wide_api_rate_limiter"] = self.api_gateway.manifest_receipt() if self.api_gateway is not None else None
        if self.admission_dispatch_min_spacing_s > 0 or self.move_submission_min_spacing_s > 0:
            configuration["api_dispatch_smoothing"] = {
                "contract": _API_DISPATCH_SMOOTHING_CONTRACT,
                "admission_dispatch_min_spacing_s": self.admission_dispatch_min_spacing_s,
                "move_submission_min_spacing_s": self.move_submission_min_spacing_s,
                "admission_order": "retained-global-fifo",
                "move_order": "earliest-original-turn-deadline-first",
                "model_call_serialization": False,
                "deadline_authority": "original-turn-deadline-and-transport-wait-budget",
            }
        if self.manifest_path.is_file():
            actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            comparable = {key: actual.get(key) for key in configuration if key != "agent"}
            expected = {key: value for key, value in configuration.items() if key != "agent"}
            if comparable != expected:
                raise RuntimeError(f"parallel-run resume configuration differs: {comparable!r} != {expected!r}")
            return
        configuration["started_at"] = _now()
        configuration["initial_stats"] = self.initial_stats
        _atomic_json(self.manifest_path, configuration)

    def _event(self, kind: str, **values: object) -> dict[str, object]:
        with self._event_lock:
            self._event_sequence += 1
            record = {"schema_version": 1, "event_sequence": self._event_sequence, "ts": _now(), "kind": kind, **values}
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return record

    @staticmethod
    def _compact_timing_receipt(receipt: dict[str, object]) -> dict[str, object]:
        profile = receipt.get("profile") if isinstance(receipt.get("profile"), dict) else {}
        exact = profile.get("exact") if isinstance(profile.get("exact"), dict) else {}
        causal_wall = profile.get("causal_wall") if isinstance(profile.get("causal_wall"), dict) else {}
        candidates = receipt.get("timing_candidates") if isinstance(receipt.get("timing_candidates"), list) else []
        return {
            "contract": receipt.get("contract"),
            "inserted_exact": receipt.get("inserted_exact"),
            "inserted_wall": receipt.get("inserted_wall"),
            "fingerprint_sha256": profile.get("fingerprint_sha256"),
            "engine_hint": profile.get("engine_hint"),
            "exact_count": exact.get("count", 0),
            "causal_wall_count": causal_wall.get("count", 0),
            "latest_hardness": receipt.get("latest_hardness"),
            "timing_candidates": candidates[:5],
        }

    def _observe_opponent_timing(self, game: dict[str, Any], *, turn_id: str, event: dict[str, object], terminal: bool = False) -> None:
        try:
            receipt = self.opponent_timing.observe_turn(
                game=game,
                turn_id=turn_id,
                observed_at=str(event["ts"]),
                source_run=str(self.run_dir),
                source_event_sequence=int(event["event_sequence"]),
                terminal=terminal,
            )
        except Exception as error:
            self._event("opponent_timing_failed", game_id=game.get("game_id"), turn_id=turn_id, error=f"{type(error).__name__}: {error}")
            return
        compact = self._compact_timing_receipt(receipt)
        if compact["inserted_exact"] or compact["inserted_wall"]:
            self._event("opponent_timing_observed", game_id=game.get("game_id"), family=game.get("game_family"), turn_id=turn_id, timing=compact)

    def _record_timing_anchor(self, envelope: TurnEnvelope, *, submission_event: dict[str, object], result: dict[str, Any]) -> None:
        if result.get("game_over"):
            return
        try:
            self.opponent_timing.record_submission(
                game=envelope.game,
                turn_id=envelope.snapshot.turn_id,
                submitted_at=str(submission_event["ts"]),
                source_run=str(self.run_dir),
                source_event_sequence=int(submission_event["event_sequence"]),
            )
        except Exception as error:
            self._event("opponent_timing_anchor_failed", game_id=envelope.game.get("game_id"), turn_id=envelope.snapshot.turn_id, error=f"{type(error).__name__}: {error}")

    @staticmethod
    def _identity_mode(game: dict[str, Any]) -> str:
        opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
        return "hidden" if opponent.get("type") == "hidden" or not str(opponent.get("name") or "").strip() else "known"

    def _sample_legacy_concealment_target_s(self) -> float:
        if self.move_delay_max_s == self.move_delay_min_s:
            return self.move_delay_min_s
        fraction = self._delay_random.betavariate(2.0, 2.0)
        return self.move_delay_min_s + fraction * (self.move_delay_max_s - self.move_delay_min_s)

    def _sample_concealment_target_s(self, game: dict[str, Any], profile: dict[str, object] | None, *, collector_assignment: Mapping[str, object] | None = None) -> tuple[float, dict[str, object]]:
        identity_mode = self._identity_mode(game)
        if self.worker_policy == "collector-local":
            if collector_assignment is None:
                return 0.0, {"status": "collector-assignment-missing", "identity_mode": identity_mode, "authority": "fail-closed-zero-delay"}
            target, receipt = sample_collector_timing_target(game, collector_assignment, self._delay_random)
            if target is None:
                return 0.0, {**receipt, "identity_mode": identity_mode, "authority": "fail-closed-zero-delay"}
            bounded = min(self.move_delay_max_s, max(self.move_delay_min_s, target))
            return bounded, {**receipt, "identity_mode": identity_mode, "authority": "collector-game-pinned-timing", "bounded_target_elapsed_s": round(bounded, 6), "configured_min_s": self.move_delay_min_s, "configured_max_s": self.move_delay_max_s}
        identity_scope_disabled = (self.move_delay_scope == "known-only" and identity_mode == "hidden") or (self.move_delay_scope == "hidden-only" and identity_mode == "known")
        if self.move_delay_scope == "none" or identity_scope_disabled or self.move_delay_max_s <= 0.0:
            return 0.0, {"status": "scope-disabled", "identity_mode": identity_mode}
        persona_target, persona_receipt = sample_timing_persona_target(game, profile, self._delay_random)
        if persona_target is not None:
            bounded_persona_target = min(self.move_delay_max_s, max(self.move_delay_min_s, persona_target))
            persona_receipt = {**persona_receipt, "bounded_target_elapsed_s": round(bounded_persona_target, 6), "configured_min_s": self.move_delay_min_s, "configured_max_s": self.move_delay_max_s}
            if persona_receipt.get("activation") == "authoritative":
                return bounded_persona_target, {**persona_receipt, "authority": "hi-timing-persona"}
            legacy_target = self._sample_legacy_concealment_target_s()
            return legacy_target, {**persona_receipt, "authority": "shadow-only", "authoritative_legacy_target_elapsed_s": round(legacy_target, 6)}
        legacy_target = self._sample_legacy_concealment_target_s()
        return legacy_target, {**persona_receipt, "authority": "ki-or-legacy-beta-2-2", "authoritative_legacy_target_elapsed_s": round(legacy_target, 6)}

    def _realize_message_style(self, envelope: TurnEnvelope, action: dict[str, Any], *, stage: str) -> tuple[dict[str, Any], dict[str, object] | None]:
        if self.message_style_policy_store is None:
            return action, envelope.message_style_profile
        try:
            profile, created, pointer_error = self.message_style_policy_store.profile_for_action(envelope.game, action)
            if profile is None:
                return action, None
            if created:
                timing_persona = profile.get("timing_persona") if isinstance(profile.get("timing_persona"), dict) else None
                self._event("message_style_profile_pinned", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], family=envelope.game["game_family"], profile_id=profile.get("profile_id"), economic_style=profile.get("economic_style"), identity_mode=profile.get("identity_mode"), policy_revision=profile.get("policy_revision"), release_sha256=profile.get("release_sha256"), timing_profile_id=timing_persona.get("timing_profile_id") if timing_persona is not None else None, timing_activation=timing_persona.get("activation") if timing_persona is not None else None, pointer_error=pointer_error)
            styled, receipt = realize_message_style(envelope.game, action, profile)
            if receipt.get("status") == "styled":
                self._event("message_style_realized", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], family=envelope.game["game_family"], stage=stage, receipt=receipt, action_changed=styled != action, economic_action_changed=False)
            return styled, profile
        except Exception as error:
            self._event("message_style_realization_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], family=envelope.game["game_family"], stage=stage, error=f"{type(error).__name__}: {error}", action_changed=False)
            return action, envelope.message_style_profile

    def _stage_submission(self, envelope: TurnEnvelope, decision: WorkerDecision) -> None:
        turn_id = envelope.snapshot.turn_id
        styled_action, profile = self._realize_message_style(envelope, decision.action, stage="primary")
        if styled_action != decision.action or profile != envelope.message_style_profile:
            envelope = replace(envelope, message_style_profile=profile)
            decision = replace(decision, action=styled_action)
        self._prepare_submission(envelope, decision)
        now = time.monotonic()
        first_seen = self._first_seen.get(turn_id, now)
        natural_elapsed_s = max(0.0, now - first_seen)
        identity_mode = self._identity_mode(envelope.game)
        metadata = decision.call_metadata if isinstance(decision.call_metadata, Mapping) else {}
        collector_assignment = metadata.get("collector_identity_assignment") if isinstance(metadata.get("collector_identity_assignment"), Mapping) else None
        target_elapsed_s, timing_persona = self._sample_concealment_target_s(envelope.game, profile, collector_assignment=collector_assignment)
        if self.worker_policy == "collector-local" and timing_persona.get("status") != "sampled":
            self._enter_drain(f"collector timing authority unavailable: {timing_persona.get('status')}")
        hard_latest = envelope.deadline_at_monotonic - self.emergency_margin_s
        submission_spacing_latest = min(hard_latest, envelope.deadline_at_monotonic - self._submission_api_call_budget_s())
        scheduler_guard_s = max(0.25, self.poll_interval_s + 0.25)
        safe_not_before = max(now, hard_latest - scheduler_guard_s)
        requested_not_before = first_seen + target_elapsed_s
        not_before = min(requested_not_before, safe_not_before)
        requested_delay_s = max(0.0, requested_not_before - now)
        scheduled_delay_s = max(0.0, not_before - now)
        schedule: dict[str, object] = {
            "contract": _MOVE_DELAY_CONCEALMENT_CONTRACT,
            "scope": self.move_delay_scope,
            "identity_mode": identity_mode,
            "scope_applied": target_elapsed_s > 0.0,
            "natural_elapsed_s": round(natural_elapsed_s, 6),
            "target_elapsed_s": round(target_elapsed_s, 6),
            "requested_delay_s": round(requested_delay_s, 6),
            "scheduled_delay_s": round(scheduled_delay_s, 6),
            "deadline_reserve_s": self.emergency_margin_s,
            "scheduled_at_monotonic": now,
            "not_before_monotonic": not_before,
            "hard_latest_monotonic": hard_latest,
            "submission_spacing_latest_monotonic": submission_spacing_latest,
            "selection_branch": decision.selection_branch,
            "timing_persona": timing_persona,
        }
        self._event("move_delay_scheduled", turn_id=turn_id, game_id=envelope.game["game_id"], family=envelope.game["game_family"], concealment={key: value for key, value in schedule.items() if not key.endswith("_monotonic")})
        if scheduled_delay_s <= 0.0 and self.move_submission_min_spacing_s <= 0.0:
            identity_scope_disabled = (self.move_delay_scope == "known-only" and identity_mode == "hidden") or (self.move_delay_scope == "hidden-only" and identity_mode == "known")
            reason = "identity-scope-disabled" if identity_scope_disabled else "disabled" if target_elapsed_s <= 0.0 else "natural-latency-reached-target" if natural_elapsed_s >= target_elapsed_s else "deadline-reserve"
            self._release_submission_safely(envelope, decision, schedule, reason=reason)
            return
        self._ready[turn_id] = (envelope, decision, schedule)
        if scheduled_delay_s <= 0.0:
            self._release_ready()

    def _release_submission(self, envelope: TurnEnvelope, decision: WorkerDecision, schedule: dict[str, object], *, reason: str) -> None:
        now = time.monotonic()
        self._last_move_submission_release_monotonic = now
        applied_delay_s = max(0.0, now - float(schedule["scheduled_at_monotonic"]))
        first_seen = self._first_seen.get(envelope.snapshot.turn_id, now)
        self._event(
            "move_delay_released",
            turn_id=envelope.snapshot.turn_id,
            game_id=envelope.game["game_id"],
            family=envelope.game["game_family"],
            concealment={
                "contract": schedule["contract"],
                "scope": schedule["scope"],
                "identity_mode": schedule["identity_mode"],
                "scope_applied": schedule["scope_applied"],
                "release_reason": reason,
                "target_elapsed_s": schedule["target_elapsed_s"],
                "requested_delay_s": schedule["requested_delay_s"],
                "scheduled_delay_s": schedule["scheduled_delay_s"],
                "applied_delay_s": round(applied_delay_s, 6),
                "queued_beyond_preferred_release_s": round(max(0.0, now - max(float(schedule["scheduled_at_monotonic"]), float(schedule["not_before_monotonic"]))), 6),
                "move_submission_min_spacing_s": self.move_submission_min_spacing_s,
                "total_elapsed_before_submit_s": round(max(0.0, now - first_seen), 6),
                "remaining_deadline_s": round(envelope.deadline_at_monotonic - now, 6),
                "deadline_reserve_s": schedule["deadline_reserve_s"],
                "timing_persona": schedule.get("timing_persona"),
            },
        )
        self._submit(envelope, decision, already_prepared=True)

    def _release_submission_safely(self, envelope: TurnEnvelope, decision: WorkerDecision, schedule: dict[str, object], *, reason: str) -> None:
        """Contain an unexpected release failure to one durable turn and drain safely."""
        turn_id = envelope.snapshot.turn_id
        try:
            self._release_submission(envelope, decision, schedule, reason=reason)
            return
        except Exception as error:
            receipt = self.broker.turn_receipt(turn_id)
            broker_status = str(receipt.get("status")) if receipt is not None else "missing"
            containment_error = None
            if broker_status in {"prepared", "submitting"}:
                issue = f"unexpected per-turn submission failure was contained: {type(error).__name__}: {error}"
                try:
                    self.broker.suspend_transport_submission(turn_id, issue=issue)
                    self._transport_blocked_turns.add(turn_id)
                    broker_status = "transport-suspended"
                except Exception as nested:
                    containment_error = f"{type(nested).__name__}: {nested}"
            self._event(
                "move_submission_turn_failure_contained",
                contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT,
                turn_id=turn_id,
                game_id=envelope.game["game_id"],
                family=envelope.game["game_family"],
                release_reason=reason,
                error=f"{type(error).__name__}: {error}",
                broker_status=broker_status,
                containment_error=containment_error,
                supervisor_survived=True,
            )
            try:
                self._enter_drain(f"unexpected move-submission failure on {turn_id}")
            except Exception as drain_error:
                self._draining = True
                self._event("move_submission_containment_drain_failed", turn_id=turn_id, game_id=envelope.game["game_id"], error=f"{type(drain_error).__name__}: {drain_error}", supervisor_survived=True)

    def _release_ready(self) -> None:
        now = time.monotonic()
        scheduler_guard_s = max(0.25, self.poll_interval_s + 0.25)
        ordered = sorted(self._ready.items(), key=lambda item: item[1][0].deadline_at_monotonic)
        for turn_id, (envelope, decision, schedule) in ordered:
            receipt = self.broker.turn_receipt(turn_id)
            broker_status = str(receipt.get("status")) if receipt is not None else "missing"
            if broker_status not in {"prepared", "submitting"}:
                del self._ready[turn_id]
                self._event("stale_ready_submission_discarded", turn_id=turn_id, game_id=envelope.game["game_id"], family=envelope.game["game_family"], broker_status=broker_status)
                continue
            hard_latest = float(schedule["hard_latest_monotonic"])
            submission_spacing_latest = float(schedule.get("submission_spacing_latest_monotonic", hard_latest))
            not_before = float(schedule["not_before_monotonic"])
            target_due = now >= not_before
            deadline_reserve_due = hard_latest - now <= scheduler_guard_s
            transport_reserve_due = self.move_submission_min_spacing_s > 0.0 and submission_spacing_latest - now <= scheduler_guard_s
            if not target_due and not deadline_reserve_due and not transport_reserve_due:
                continue
            spacing_remaining_s = max(0.0, self._last_move_submission_release_monotonic + self.move_submission_min_spacing_s - now)
            spacing_must_yield = self.move_submission_min_spacing_s > 0.0 and submission_spacing_latest - now <= max(spacing_remaining_s, scheduler_guard_s)
            if spacing_remaining_s > 0.0 and not spacing_must_yield:
                continue
            del self._ready[turn_id]
            if spacing_remaining_s > 0.0 and spacing_must_yield:
                reason = "submission-spacing-transport-reserve"
            elif target_due:
                reason = "target-reached"
            elif transport_reserve_due:
                reason = "transport-wait-budget-reserve"
            else:
                reason = "deadline-reserve"
            self._release_submission_safely(envelope, decision, schedule, reason=reason)
            now = time.monotonic()

    def _update_opponent_statistical_package(self, final_state: dict[str, Any], completion_event: dict[str, object]) -> None:
        reader = self.opponent_statistical_package_reader
        if reader is None or not hasattr(reader, "update_completed_game"):
            return
        try:
            receipt = reader.update_completed_game(final_state, completed_at=str(completion_event["ts"]), completion_order=int(completion_event["event_sequence"]))
        except Exception as error:
            self._event("opponent_statistical_package_update_failed", game_id=final_state.get("game_id"), family=final_state.get("game_family"), error=f"{type(error).__name__}: {error}")
            try:
                self._enter_drain("opponent statistical-package update failed")
            except Exception as drain_error:
                self._draining = True
                self._event("opponent_statistical_package_failure_drain_error", game_id=final_state.get("game_id"), error=f"{type(drain_error).__name__}: {drain_error}")
            return
        self._event("opponent_statistical_package_updated", game_id=final_state.get("game_id"), family=final_state.get("game_family"), receipt=receipt)

    def _reconcile_opponent_statistical_package(self) -> None:
        reader = self.opponent_statistical_package_reader
        games_dir = self.run_dir / "games"
        if reader is None or not hasattr(reader, "update_completed_game") or not games_dir.is_dir():
            return
        completion_events: dict[str, dict[str, object]] = {}
        if self.events_path.is_file():
            with self.events_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("kind") in {"game_completed", "game_completed_during_opponent_turn"} and event.get("game_id"):
                        completion_events[str(event["game_id"])] = event
        statuses: Counter[str] = Counter()
        for path in sorted(games_dir.glob("*.json")):
            final_state = json.loads(path.read_text(encoding="utf-8"))
            game_id = str(final_state.get("game_id") or "")
            event = completion_events.get(game_id, {})
            receipt = reader.update_completed_game(final_state, completed_at=str(event.get("ts") or "recovered-from-terminal-game-file"), completion_order=int(event.get("event_sequence") or 0))
            statuses[str(receipt["status"])] += 1
        self._event("opponent_statistical_package_reconciled", terminal_game_files=sum(statuses.values()), statuses=dict(sorted(statuses.items())), overlay=reader.status() if hasattr(reader, "status") else None)

    @staticmethod
    def _rating_package_context(snapshot: DossierSnapshot) -> object:
        return snapshot.memory_context.get("opponent_statistical_package")

    def _capture_rating_canary_game(self, game: dict[str, Any], snapshot: DossierSnapshot, *, observed_at: str) -> None:
        if self.bargaining_rating_canary is None or game.get("game_family") != "bargaining":
            return
        try:
            inserted = self.bargaining_rating_canary.capture_game(game, package_context=self._rating_package_context(snapshot), sensor_frontier=self._read_sensor_frontier(), observed_at=observed_at)
        except Exception as error:
            self.bargaining_rating_canary.failure(stage="game-context", error=error, game_id=str(game.get("game_id") or ""))
            self._event("bargaining_rating_canary_game_context_failed", game_id=game.get("game_id"), error=f"{type(error).__name__}: {error}")
            return
        self._event("bargaining_rating_canary_game_context", game_id=game.get("game_id"), inserted=inserted, package_overlay=self._rating_package_context(snapshot).get("live_overlay") if isinstance(self._rating_package_context(snapshot), dict) else None)

    def _register_rating_canary_turn(self, envelope: TurnEnvelope, action: dict[str, Any], *, stage: str) -> None:
        if self.bargaining_rating_canary is None or envelope.game.get("game_family") != "bargaining":
            return
        turn_key = f"{envelope.snapshot.turn_id}:{stage}"
        try:
            inserted = self.bargaining_rating_canary.register_turn(turn_id=envelope.snapshot.turn_id, stage=stage, game=envelope.game, action=action, advisor_handle=envelope.bargaining_advisor_handle, package_context=self._rating_package_context(envelope.snapshot), causal_observation_at=_now())
        except Exception as error:
            self.bargaining_rating_canary.failure(stage="turn-shadow", error=error, game_id=str(envelope.game.get("game_id") or ""), turn_key=turn_key)
            self._event("bargaining_rating_canary_turn_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game.get("game_id"), stage=stage, error=f"{type(error).__name__}: {error}")
            return
        self._event("bargaining_rating_canary_turn_registered", turn_id=envelope.snapshot.turn_id, game_id=envelope.game.get("game_id"), stage=stage, inserted=inserted, action_changed=False)

    def _register_rating_canary_terminal(self, final_state: dict[str, Any], *, terminal_at: str) -> None:
        if self.bargaining_rating_canary is None or final_state.get("game_family") != "bargaining":
            return
        try:
            inserted = self.bargaining_rating_canary.register_terminal(final_state, terminal_at=terminal_at)
        except Exception as error:
            self.bargaining_rating_canary.failure(stage="terminal-forecast", error=error, game_id=str(final_state.get("game_id") or ""))
            self._event("bargaining_rating_canary_terminal_failed", game_id=final_state.get("game_id"), error=f"{type(error).__name__}: {error}")
            return
        self._event("bargaining_rating_canary_terminal_registered", game_id=final_state.get("game_id"), inserted=inserted, terminal_at=terminal_at)

    def _reconcile_rating_canary(self, *, force: bool = False) -> None:
        if self.bargaining_rating_canary is None:
            return
        try:
            receipt = self.bargaining_rating_canary.reconcile(force=force)
        except Exception as error:
            self.bargaining_rating_canary.failure(stage="self-maturation", error=error)
            self._event("bargaining_rating_canary_reconcile_failed", error=f"{type(error).__name__}: {error}")
            return
        if receipt.get("status") != "throttled" and (force or int(receipt.get("matured") or 0) > 0):
            self._event("bargaining_rating_canary_reconciled", receipt=receipt)

    def _capture_rating_v3_game(self, game: dict[str, Any], snapshot: DossierSnapshot, *, observed_at: str) -> None:
        if self.rating_v3_advisory is None:
            return
        try:
            inserted = self.rating_v3_advisory.capture_game(game, package_context=self._rating_package_context(snapshot), sensor_frontier=self._read_sensor_frontier(), fallback_stats=self._cached_stats, observed_at=observed_at)
        except Exception as error:
            self._event("rating_v3_game_context_failed", game_id=game.get("game_id"), family=game.get("game_family"), error=f"{type(error).__name__}: {error}")
            return
        self._event("rating_v3_game_context", game_id=game.get("game_id"), family=game.get("game_family"), inserted=inserted, action_changed=False)

    def _rating_v3_turn_advisory(self, *, turn_id: str, game: dict[str, Any], observed_at: str, bargaining_advisor_handle: object | None, negotiation_advisor_handle: object | None, negotiation_advisor_context: object, persuasion_advisor_context: object) -> dict[str, object] | None:
        if self.rating_v3_advisory is None:
            return None
        try:
            advisory = self.rating_v3_advisory.turn_advisory(turn_id=turn_id, game=game, observed_at=observed_at, bargaining_advisor_handle=bargaining_advisor_handle, negotiation_advisor_handle=negotiation_advisor_handle, negotiation_advisor_context=negotiation_advisor_context, persuasion_advisor_context=persuasion_advisor_context)
        except Exception as error:
            advisory = {"contract": "glee-rating-v3-dual-channel-advisory-v1", "status": "unavailable", "error_type": type(error).__name__, "prompt_authority": "advisory-only", "guard_authority": "none"}
            self._event("rating_v3_turn_advisory_failed", turn_id=turn_id, game_id=game.get("game_id"), family=game.get("game_family"), error=f"{type(error).__name__}: {error}", action_changed=False)
            return advisory
        self._event("rating_v3_turn_advisory_registered", turn_id=turn_id, game_id=game.get("game_id"), family=game.get("game_family"), advisory=advisory, registered_before_model_inference=True, action_changed=False)
        return advisory

    def _register_rating_v3_terminal(self, final_state: dict[str, Any], *, terminal_at: str) -> None:
        if self.rating_v3_advisory is None:
            return
        try:
            inserted = self.rating_v3_advisory.register_terminal(final_state, terminal_at=terminal_at)
        except Exception as error:
            self._event("rating_v3_terminal_forecast_failed", game_id=final_state.get("game_id"), family=final_state.get("game_family"), error=f"{type(error).__name__}: {error}")
            return
        self._event("rating_v3_terminal_forecast_registered", game_id=final_state.get("game_id"), family=final_state.get("game_family"), inserted=inserted, terminal_at=terminal_at, action_changed=False)

    def _reconcile_rating_v3(self, *, force: bool = False) -> None:
        if self.rating_v3_advisory is None:
            return
        try:
            receipt = self.rating_v3_advisory.reconcile(force=force)
        except Exception as error:
            self._event("rating_v3_reconcile_failed", error=f"{type(error).__name__}: {error}")
            return
        if receipt.get("status") != "throttled" and (force or int(receipt.get("matured") or 0) > 0):
            self._event("rating_v3_reconciled", receipt=receipt)

    def _update_bargaining_advisor(self, final_state: dict[str, Any], completion_event: dict[str, object]) -> None:
        if self.bargaining_advisor is None or final_state.get("game_family") != "bargaining":
            return
        try:
            receipt = self.bargaining_advisor.update_completed_game(final_state, completed_at=str(completion_event["ts"]), completion_order=int(completion_event["event_sequence"]))
        except Exception as error:
            self._event("bargaining_v2_update_failed", game_id=final_state.get("game_id"), error=f"{type(error).__name__}: {error}")
            return
        self._event("bargaining_v2_updated", game_id=final_state.get("game_id"), receipt=receipt)

    def _update_negotiation_advisor(self, final_state: dict[str, Any], completion_event: dict[str, object]) -> None:
        if self.negotiation_advisor is None or final_state.get("game_family") != "negotiation":
            return
        try:
            receipt = self.negotiation_advisor.update_completed_game(final_state, completed_at=str(completion_event["ts"]), completion_order=int(completion_event["event_sequence"]))
        except Exception as error:
            self._event("negotiation_v2_update_failed", game_id=final_state.get("game_id"), error=f"{type(error).__name__}: {error}")
            return
        self._event("negotiation_v2_updated", game_id=final_state.get("game_id"), receipt=receipt)

    def _update_persuasion_advisor(self, final_state: dict[str, Any], completion_event: dict[str, object]) -> None:
        if self.persuasion_advisor is None or final_state.get("game_family") != "persuasion":
            return
        try:
            receipt = self.persuasion_advisor.update_completed_game(final_state, completed_at=str(completion_event["ts"]), completion_order=int(completion_event["event_sequence"]))
        except Exception as error:
            self._event("persuasion_v2_update_failed", game_id=final_state.get("game_id"), error=f"{type(error).__name__}: {error}")
            return
        self._event("persuasion_v2_updated", game_id=final_state.get("game_id"), receipt=receipt)

    def _sample_memory(self, *, force: bool = False) -> dict[str, float]:
        now = time.monotonic()
        if force or now - self._last_memory_sample >= self.memory_sample_interval_s:
            sample = _process_memory_mib()
            reported_peak = float(sample["peak_rss_mib"])
            self._run_peak_rss_mib = max(self._run_peak_rss_mib, reported_peak, float(sample["rss_mib"]))
            self._cached_memory = {**sample, "process_reported_peak_rss_mib": reported_peak, "peak_rss_mib": round(self._run_peak_rss_mib, 3)}
            self._last_memory_sample = now
            self._event("memory_sample", **self._cached_memory)
        return self._cached_memory

    def _read_sensor_frontier(self) -> dict[str, Any] | None:
        if self.sensor_reader is None:
            return None
        try:
            return self.sensor_reader.read()
        except Exception:
            return None

    def _record_sensor_fallback(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_sensor_failure_event >= 30.0:
            self._event("sensor_feed_fallback", reason=reason, contract=SENSOR_CONTRACT)
            self._last_sensor_failure_event = now

    def _api_reserve(self, *, operation: str, priority: APIPriority, wait: bool = False, deadline_monotonic: float | None = None) -> dict[str, object] | None:
        if self.api_gateway is None:
            return None
        return self.api_gateway.reserve(operation=operation, priority=priority, wait=wait, deadline_monotonic=deadline_monotonic)

    def _api_call(self, *, operation: str, priority: APIPriority, callback: Any, wait: bool = False, deadline_monotonic: float | None = None, prepaid: dict[str, object] | None = None) -> Any:
        if self.api_gateway is None:
            return callback()
        return self.api_gateway.call(operation=operation, priority=priority, callback=callback, wait=wait, deadline_monotonic=deadline_monotonic, prepaid=prepaid)

    def _pending_games(self) -> list[dict[str, Any]]:
        if self.sensor_reader is not None:
            try:
                frontier = self.sensor_reader.read()
            except Exception as error:
                self._record_sensor_fallback(f"{type(error).__name__}: {error}")
            else:
                if frontier is not None:
                    return [dict(game) for game in frontier["pending_games"] if str(game.get("game_family")) in self.families]
                self._record_sensor_fallback("sensor frontier missing or stale")
        try:
            pending = self._api_call(operation="pending_games", priority="control", callback=self.client.pending_games)
        except Exception as error:
            if not _is_deferred_api_control(error):
                raise
            self._event("pending_games_poll_failed", error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None)
            return []
        return [game for game in pending if str(game.get("game_family")) in self.families]

    def _leave_selected_queues(self) -> dict[str, object]:
        responses: dict[str, object] = {}
        for family in self.families:
            responses[family] = self._api_call(operation="leave_queue", priority="control", callback=lambda family=family: self.client.leave_queue(family))
        self._queued_families.clear()
        self._ambiguous_queue_deadlines.clear()
        self._ambiguous_queue_empty_checks.clear()
        return responses

    def _queue_ambiguity_grace_s(self) -> float:
        return max(_QUEUE_DISPATCH_AMBIGUITY_GRACE_S, self.poll_interval_s * 3.0)

    def _admission_quarantine_active(self, *, now_unix: float | None = None) -> bool:
        now = time.time() if now_unix is None else float(now_unix)
        if self._admission_quarantine_until_unix <= 0.0:
            return False
        if now < self._admission_quarantine_until_unix:
            return True
        expired_at = self._admission_quarantine_until_unix
        self._admission_quarantine_until_unix = 0.0
        self._event("activity_admission_quarantine_ended", contract=_QUEUE_SAFETY_QUARANTINE_CONTRACT, retry_at=datetime.fromtimestamp(expired_at, tz=timezone.utc).isoformat(timespec="microseconds"), ended_at=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="microseconds"))
        return False

    def _defer_server_queue_quarantine(self, *, family: str, error: BaseException, restored_schedule: Mapping[str, object] | None, allocation: str) -> bool:
        receipt = _queue_safety_quarantine(error)
        if receipt is None:
            return False
        prior_until = self._admission_quarantine_until_unix
        self._admission_quarantine_until_unix = max(prior_until, float(receipt["retry_at_unix"]))
        self._queued_families.discard(family)
        self._ambiguous_queue_deadlines.pop(family, None)
        self._ambiguous_queue_empty_checks.pop(family, None)
        self._event(
            "activity_admission_quarantined",
            contract=_QUEUE_SAFETY_QUARANTINE_CONTRACT,
            family=family,
            allocation=allocation,
            quarantine=receipt,
            effective_retry_at=datetime.fromtimestamp(self._admission_quarantine_until_unix, tz=timezone.utc).isoformat(timespec="microseconds"),
            remaining_s=round(max(0.0, self._admission_quarantine_until_unix - time.time()), 6),
            restored_schedule=dict(restored_schedule) if restored_schedule is not None else None,
            policy="pause only new admissions while preserving the supervisor and all active games",
        )
        return True

    def _record_ambiguous_queue_dispatch(self, *, family: str, error: Exception, schedule: Mapping[str, object] | None, allocation: str) -> None:
        now = time.monotonic()
        self._queued_families.add(family)
        self._ambiguous_queue_deadlines[family] = now + self._queue_ambiguity_grace_s()
        self._ambiguous_queue_empty_checks[family] = 0
        self._last_admission_dispatch_monotonic = now
        self._event("activity_dispatch_outcome_uncertain", contract=_API_DISPATCH_SMOOTHING_CONTRACT, family=family, error=f"{type(error).__name__}: {error}", allocation=allocation, activity_schedule=dict(schedule) if schedule is not None else None, grace_s=self._queue_ambiguity_grace_s(), reconciliation="observe pending games, then issue up to two idempotent leave-queue checks")

    def _resolve_ambiguous_queue_by_match(self, *, family: str, game_id: str) -> None:
        deadline = self._ambiguous_queue_deadlines.pop(family, None)
        checks = self._ambiguous_queue_empty_checks.pop(family, None)
        if deadline is None:
            return
        self._event("activity_dispatch_transport_reconciled", contract=_API_DISPATCH_SMOOTHING_CONTRACT, family=family, game_id=game_id, resolution="matched", empty_checks=checks or 0)

    def _reconcile_ambiguous_queues(self) -> None:
        if not self._ambiguous_queue_deadlines:
            return
        now = time.monotonic()
        for family, deadline in list(self._ambiguous_queue_deadlines.items()):
            if now < deadline:
                continue
            try:
                response = self._api_call(operation="leave_queue", priority="control", callback=lambda family=family: self.client.leave_queue(family))
            except Exception as error:
                if not (_is_deferred_api_control(error) or isinstance(error, (RequestsConnectionError, RequestsTimeout)) or _is_transient_server_error(error)):
                    raise
                self._ambiguous_queue_deadlines[family] = time.monotonic() + self._queue_ambiguity_grace_s()
                self._event("activity_dispatch_reconciliation_deferred", contract=_API_DISPATCH_SMOOTHING_CONTRACT, family=family, error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None, empty_checks=self._ambiguous_queue_empty_checks.get(family, 0))
                continue
            removed = int(response.get("removed") or 0) if isinstance(response, Mapping) else 0
            empty_checks = self._ambiguous_queue_empty_checks.get(family, 0) + int(removed == 0)
            self._ambiguous_queue_empty_checks[family] = empty_checks
            if removed == 0 and empty_checks < _QUEUE_DISPATCH_AMBIGUITY_EMPTY_CHECKS:
                self._ambiguous_queue_deadlines[family] = time.monotonic() + self._queue_ambiguity_grace_s()
                self._event("activity_dispatch_reconciliation_observation_extended", contract=_API_DISPATCH_SMOOTHING_CONTRACT, family=family, response=response, empty_checks=empty_checks, grace_s=self._queue_ambiguity_grace_s())
                continue
            restored = self.activity_scheduler.return_outstanding(family, queue_error=True) if self.activity_scheduler is not None else None
            self._queued_families.discard(family)
            self._ambiguous_queue_deadlines.pop(family, None)
            self._ambiguous_queue_empty_checks.pop(family, None)
            self._event("activity_dispatch_transport_reconciled", contract=_API_DISPATCH_SMOOTHING_CONTRACT, family=family, resolution="queue-cancelled" if removed > 0 else "queue-absent-after-two-checks", response=response, empty_checks=empty_checks, restored_schedule=restored)

    def _sync_pause(self) -> None:
        requested = self.pause_request_path.is_file()
        if requested and not self._paused:
            now = time.monotonic()
            if now - self._last_pause_leave_attempt < max(15.0, self.poll_interval_s * 5):
                return
            self._last_pause_leave_attempt = now
            try:
                responses = self._leave_selected_queues()
            except Exception as error:
                if not _is_deferred_api_control(error):
                    raise
                self._event("pause_queue_leave_deferred", families=list(self.families), error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None)
                return
            self._paused = True
            activity_schedule = self.activity_scheduler.set_paused(True, now=time.time()) if self.activity_scheduler is not None else None
            acknowledgement = {"schema_version": 1, "kind": "glee-family-pause-active", "paused_at": _now(), "families": list(self.families), "queue_responses": responses, "activity_schedule": activity_schedule}
            _atomic_json(self.pause_ack_path, acknowledgement)
            self._event("pause_started", families=list(self.families), queue_responses=responses, activity_schedule=activity_schedule)
        elif not requested and self._paused:
            self._paused = False
            activity_schedule = self.activity_scheduler.set_paused(False, now=time.time()) if self.activity_scheduler is not None else None
            self.pause_ack_path.unlink(missing_ok=True)
            self._event("pause_ended", families=list(self.families), activity_schedule=activity_schedule)

    def _sync_activity_target(self) -> None:
        if self.activity_scheduler is None or not self.activity_target_request_path.is_file():
            return
        request = json.loads(self.activity_target_request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise RuntimeError("activity target request must be a JSON object")
        request_id = str(request.get("request_id") or "")
        prior_ack = json.loads(self.activity_target_ack_path.read_text(encoding="utf-8")) if self.activity_target_ack_path.is_file() else None
        if isinstance(prior_ack, dict) and request_id and prior_ack.get("request_id") == request_id:
            return
        try:
            if request.get("schema_version") != 1 or request.get("contract") != LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT:
                raise ValueError("activity target request has an incompatible contract")
            if not request_id:
                raise ValueError("activity target request requires a request id")
            family = str(request.get("family") or "")
            if family not in self.families:
                raise ValueError(f"activity target request names an unserved family: {family!r}")
            target = request.get("target_per_48h")
            if isinstance(target, bool) or not isinstance(target, int):
                raise ValueError("activity target per 48 hours must be an integer")
            if float(request.get("window_s") or 0.0) != self.activity_scheduler.window_s or self.activity_scheduler.window_s != DEFAULT_WINDOW_S:
                raise ValueError("g48 control requires the run's 48-hour activity window")
            schedule = self.activity_scheduler.set_target(family, target, request_id=request_id, now=time.time())
            acknowledgement = {"schema_version": 1, "contract": LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT, "request_id": request_id, "status": "applied", "family": family, "target_per_48h": target, "acknowledged_at": _now(), "schedule": schedule}
            _atomic_json(self.activity_target_ack_path, acknowledgement)
            self._event("activity_target_applied", request=request, acknowledgement=acknowledgement)
        except (KeyError, TypeError, ValueError) as error:
            acknowledgement = {"schema_version": 1, "contract": LIVE_ACTIVITY_TARGET_CONTROL_CONTRACT, "request_id": request_id or None, "status": "rejected", "acknowledged_at": _now(), "error": f"{type(error).__name__}: {error}"}
            _atomic_json(self.activity_target_ack_path, acknowledgement)
            self._event("activity_target_rejected", request=request, acknowledgement=acknowledgement)

    def _limit_reason(self, started: float) -> str | None:
        memory = self._sample_memory()
        if self.drain_request_path.is_file():
            return "persistent graceful-drain request"
        if self.max_rss_mib is not None and memory["rss_mib"] >= self.max_rss_mib:
            return f"process RSS {memory['rss_mib']:.3f} MiB reached the configured ceiling of {self.max_rss_mib:.3f} MiB"
        completed = len(self.broker.completed_game_ids())
        if self.stop_after_family is not None:
            family, target = self.stop_after_family
            actual = self.broker.game_counts_by_family("completed").get(family, 0)
            if actual >= target:
                return f"completed {actual} {family} games, reaching the configured target of {target}"
        if self.max_games is not None and completed >= self.max_games:
            return f"completed {completed} games, reaching the configured target of {self.max_games}"
        if self.max_time_s is not None and time.monotonic() - started >= self.max_time_s:
            return f"reached the configured wall-time limit of {self.max_time_s} seconds"
        return None

    def _limit_reached(self, started: float) -> bool:
        return self._limit_reason(started) is not None

    def _enter_drain(self, reason: str) -> None:
        if self._draining:
            return
        self._draining = True
        self._last_family_activity = time.monotonic()
        try:
            responses = self._leave_selected_queues()
        except Exception as error:
            if not _is_deferred_api_control(error):
                raise
            responses = {"deferred": True, "error": f"{type(error).__name__}: {error}", "rate_limited": _is_rate_limit(error), "locally_deferred": isinstance(error, GleeAPIBudgetDeferred), "budget_receipt": error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None}
        activity_schedule = self.activity_scheduler.set_paused(True, now=time.time()) if self.activity_scheduler is not None else None
        self._event("drain_started", reason=reason, queue_responses=responses, activity_schedule=activity_schedule)

    def _top_up(self, active_games: int) -> None:
        self._sync_pause()
        if self._draining or self._paused or self.pause_request_path.is_file():
            return
        now = time.monotonic()
        if self.activity_scheduler is not None:
            if now - self._last_topup < self.poll_interval_s:
                return
            self._top_up_scheduled(active_games)
            self._last_topup = now
            return
        interval = max(15.0, self.poll_interval_s * 5)
        if now - self._last_topup < interval:
            return
        if self.family_slots is not None:
            self._top_up_fixed_family_slots(active_games)
            self._last_topup = now
            return
        completed = len(self.broker.completed_game_ids())
        remaining = self.max_parallel - active_games
        if self.max_games is not None:
            remaining = min(remaining, self.max_games - completed - active_games)
        target_queues = max(0, min(remaining, len(self.families)))
        while len(self._queued_families) < target_queues:
            available = [family for family in self.families if family not in self._queued_families]
            if not available:
                break
            family = self.families[self._queue_cursor % len(self.families)]
            self._queue_cursor += 1
            if family in self._queued_families:
                continue
            try:
                response = self.client.queue(family)
            except Exception as error:
                if isinstance(error, (RequestsConnectionError, RequestsTimeout)) or _is_transient_server_error(error):
                    self._record_ambiguous_queue_dispatch(family=family, error=error, schedule=None, allocation="round-robin")
                    break
                if self._defer_server_queue_quarantine(family=family, error=error, restored_schedule=None, allocation="round-robin"):
                    break
                if not _is_rate_limit(error):
                    raise
                self._event("family_queue_deferred", family=family, error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error))
                break
            self._queued_families.add(family)
            self._event("family_queued", family=family, response=response)
        self._last_topup = now

    def _top_up_scheduled(self, active_games: int) -> None:
        if self.activity_scheduler is None:
            raise RuntimeError("scheduled top-up requires an activity scheduler")
        if self.unified_activity_pool:
            self._top_up_unified_scheduled(active_games)
            return
        if self.family_slots is None:
            raise RuntimeError("family-ceiling scheduled top-up requires family slots")
        active_counts = self.broker.game_counts_by_family("active")
        completed = len(self.broker.completed_game_ids())
        known_active = sum(active_counts.values())
        admitted_active = max(active_games, known_active)
        global_capacity = self.max_parallel - admitted_active - len(self._queued_families)
        if self.max_games is not None:
            global_capacity = min(global_capacity, self.max_games - completed - admitted_active - len(self._queued_families))
        now = time.time()
        quarantine_active = self._admission_quarantine_active(now_unix=now)
        for family in self.families:
            arrivals = self.activity_scheduler.materialize_due(family, now=now)
            for arrival in arrivals:
                self._event("activity_arrival", family=family, schedule=arrival, active_family_games=active_counts.get(family, 0), active_games=admitted_active, queued_families=sorted(self._queued_families))
            if quarantine_active:
                continue
            if self.activity_scheduler.pending_count(family) == 0 or family in self._queued_families or self.activity_scheduler.has_outstanding(family):
                continue
            family_capacity = self.family_slots[family] - active_counts.get(family, 0)
            if global_capacity <= 0 or family_capacity <= 0:
                continue
            receipt = self.activity_scheduler.dispatch_oldest(family, now=time.time())
            try:
                response = self.client.queue(family)
            except Exception as error:
                if isinstance(error, (RequestsConnectionError, RequestsTimeout)) or _is_transient_server_error(error):
                    self._record_ambiguous_queue_dispatch(family=family, error=error, schedule=receipt, allocation="independent-poisson")
                    break
                restored = self.activity_scheduler.return_outstanding(family, queue_error=True)
                if self._defer_server_queue_quarantine(family=family, error=error, restored_schedule=restored, allocation="independent-poisson"):
                    break
                if not _is_rate_limit(error):
                    raise
                self._event("activity_dispatch_deferred", family=family, error=f"{type(error).__name__}: {error}", rate_limited=True, pending_arrivals=self.activity_scheduler.pending_count(family), restored_schedule=restored)
                continue
            self._queued_families.add(family)
            global_capacity -= 1
            self._event("family_queued", family=family, response=response, allocation="independent-poisson", family_slots=self.family_slots[family], active_family_games=active_counts.get(family, 0), completed_family_games=self.broker.game_counts_by_family("completed").get(family, 0), activity_schedule=receipt)

    def _top_up_unified_scheduled(self, active_games: int) -> None:
        if self.activity_scheduler is None or not self.unified_activity_pool or self.family_slots is not None:
            raise RuntimeError("unified scheduled top-up requires one activity scheduler and no family slots")
        active_counts = self.broker.game_counts_by_family("active")
        completed = len(self.broker.completed_game_ids())
        known_active = sum(active_counts.values())
        admitted_active = max(active_games, known_active)
        global_capacity = self.max_parallel - admitted_active - len(self._queued_families)
        if self.max_games is not None:
            global_capacity = min(global_capacity, self.max_games - completed - admitted_active - len(self._queued_families))
        now = time.time()
        for family in self.families:
            arrivals = self.activity_scheduler.materialize_due(family, now=now)
            for arrival in arrivals:
                self._event("activity_arrival", family=family, schedule=arrival, active_family_games=active_counts.get(family, 0), active_games=admitted_active, queued_families=sorted(self._queued_families), allocation="unified-global-fifo")
        if self._admission_quarantine_active(now_unix=now):
            return
        while global_capacity > 0:
            family = self.activity_scheduler.oldest_pending_family(excluded=frozenset(self._queued_families))
            if family is None:
                break
            dispatch_spacing_remaining_s = max(0.0, self._last_admission_dispatch_monotonic + self.admission_dispatch_min_spacing_s - time.monotonic())
            if dispatch_spacing_remaining_s > 0.0:
                self._event("activity_dispatch_spacing_deferred", contract=_API_DISPATCH_SMOOTHING_CONTRACT, family=family, minimum_spacing_s=self.admission_dispatch_min_spacing_s, remaining_spacing_s=round(dispatch_spacing_remaining_s, 6), pending_arrivals=self.activity_scheduler.pending_total(), allocation="unified-global-fifo")
                break
            try:
                api_grant = self._api_reserve(operation="queue", priority="background")
            except GleeAPIBudgetDeferred as error:
                self._event("activity_dispatch_deferred", family=family, error=f"{type(error).__name__}: {error}", rate_limited=False, locally_deferred=True, budget_receipt=error.receipt, pending_arrivals=self.activity_scheduler.pending_total(), allocation="unified-global-fifo")
                break
            receipt = self.activity_scheduler.dispatch_oldest_global(excluded=frozenset(self._queued_families), now=time.time())
            if receipt is None:
                raise RuntimeError("globally selected activity arrival disappeared before durable dispatch")
            if str(receipt["family"]) != family:
                raise RuntimeError("globally selected activity family changed before durable dispatch")
            try:
                response = self._api_call(operation="queue", priority="background", callback=lambda family=family: self.client.queue(family), prepaid=api_grant)
            except Exception as error:
                if isinstance(error, (RequestsConnectionError, RequestsTimeout)) or _is_transient_server_error(error):
                    self._record_ambiguous_queue_dispatch(family=family, error=error, schedule=receipt, allocation="unified-global-fifo")
                    break
                restored = self.activity_scheduler.return_outstanding(family, queue_error=True)
                if self._defer_server_queue_quarantine(family=family, error=error, restored_schedule=restored, allocation="unified-global-fifo"):
                    break
                if not _is_deferred_api_control(error):
                    raise
                self._event("activity_dispatch_deferred", family=family, error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None, pending_arrivals=self.activity_scheduler.pending_total(), restored_schedule=restored, allocation="unified-global-fifo")
                break
            self._queued_families.add(family)
            self._last_admission_dispatch_monotonic = time.monotonic()
            global_capacity -= 1
            self._event("family_queued", family=family, response=response, allocation="unified-global-fifo", family_slots=None, active_family_games=active_counts.get(family, 0), active_games=admitted_active, completed_family_games=self.broker.game_counts_by_family("completed").get(family, 0), pending_arrivals=self.activity_scheduler.pending_total(), activity_schedule=receipt, dispatch_smoothing={"contract": _API_DISPATCH_SMOOTHING_CONTRACT, "minimum_spacing_s": self.admission_dispatch_min_spacing_s})

    def _top_up_fixed_family_slots(self, active_games: int) -> None:
        if self.family_slots is None:
            raise RuntimeError("fixed family top-up requires family slots")
        active_counts = self.broker.game_counts_by_family("active")
        completed_counts = self.broker.game_counts_by_family("completed")
        known_active = sum(active_counts.values())
        admitted_active = max(active_games, known_active)
        queued = len(self._queued_families)
        global_remaining = self.max_parallel - admitted_active - queued
        if self.max_games is not None:
            completed = len(self.broker.completed_game_ids())
            global_remaining = min(global_remaining, self.max_games - completed - admitted_active - queued)
        if global_remaining <= 0:
            return
        needs: list[tuple[int, int, str]] = []
        for order, family in enumerate(self.families):
            queued = int(family in self._queued_families)
            need = self.family_slots[family] - active_counts.get(family, 0) - queued
            if self.stop_after_family is not None and family == self.stop_after_family[0]:
                remaining_target = self.stop_after_family[1] - completed_counts.get(family, 0) - active_counts.get(family, 0) - queued
                need = min(need, remaining_target)
            if need > 0:
                needs.append((need, -order, family))
        for _need, _order, family in sorted(needs, reverse=True):
            if global_remaining <= 0:
                break
            try:
                response = self.client.queue(family)
            except Exception as error:
                if isinstance(error, (RequestsConnectionError, RequestsTimeout)) or _is_transient_server_error(error):
                    self._record_ambiguous_queue_dispatch(family=family, error=error, schedule=None, allocation="fixed")
                    break
                if self._defer_server_queue_quarantine(family=family, error=error, restored_schedule=None, allocation="fixed"):
                    break
                if not _is_rate_limit(error):
                    raise
                self._event("family_queue_deferred", family=family, error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), allocation="fixed")
                break
            self._queued_families.add(family)
            global_remaining -= 1
            self._event(
                "family_queued",
                family=family,
                response=response,
                allocation="fixed",
                family_slots=self.family_slots[family],
                active_family_games=active_counts.get(family, 0),
                completed_family_games=completed_counts.get(family, 0),
            )

    def _stats(self, *, force: bool = False) -> dict[str, Any]:
        if self.sensor_reader is not None:
            try:
                frontier = self.sensor_reader.read()
            except Exception as error:
                self._record_sensor_fallback(f"{type(error).__name__}: {error}")
            else:
                if frontier is not None:
                    self._cached_stats = dict(frontier["stats"])
                    return self._cached_stats
                self._record_sensor_fallback("sensor frontier missing or stale")
        now = time.monotonic()
        if force or now - self._last_stats_read >= max(15.0, self.poll_interval_s * 5):
            try:
                self._cached_stats = self._api_call(operation="stats", priority="background", callback=self.client.stats)
            except Exception as error:
                if not _is_deferred_api_control(error):
                    raise
                self._event("agent_stats_poll_failed", error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None)
            self._last_stats_read = now
        return self._cached_stats

    def _observe_sequence_shadow(self, game: dict[str, Any], *, observed_at: str, source: str) -> None:
        if self.sequence_shadow_client is None:
            return
        try:
            receipt = self.sequence_shadow_client.observe(game=game, observed_at=observed_at, source=source)
        except Exception as error:
            self._event("sequence_shadow_observation_failed", game_id=game.get("game_id"), family=game.get("game_family"), source=source, error=f"{type(error).__name__}: {error}", action_changed=False)
            return
        if receipt.get("matured") or receipt.get("mismatches"):
            self._event("sequence_shadow_outcome_observed", game_id=game.get("game_id"), family=game.get("game_family"), source=source, receipt=receipt, action_changed=False)

    def _register_sequence_shadow_before_terra(self, envelope: TurnEnvelope, *, observed_at: str) -> TurnEnvelope:
        if self.sequence_shadow_client is None:
            return envelope
        prepared = prepare_worker_envelope(envelope)
        try:
            synthetic_features = terra_synthetic_feature_bundle(worker_payload(prepared))
            receipt = self.sequence_shadow_client.forecast(game=prepared.game, turn_id=prepared.snapshot.turn_id, synthetic_features=synthetic_features, observed_at=observed_at)
            forbidden = {"action_probabilities", "component_probabilities", "predicted_action", "labels"}.intersection(receipt)
            if forbidden:
                raise ValueError(f"sequence-shadow IPC exposed forbidden forecast fields: {sorted(forbidden)}")
        except Exception as error:
            self._event("sequence_shadow_forecast_failed", turn_id=prepared.snapshot.turn_id, game_id=prepared.game.get("game_id"), family=prepared.game.get("game_family"), frontier="before-terra", error=f"{type(error).__name__}: {error}", action_changed=False)
            return prepared
        event_kind = "sequence_shadow_forecast_registered" if receipt.get("status") in {"registered", "already-registered"} else "sequence_shadow_forecast_ineligible"
        self._event(event_kind, turn_id=prepared.snapshot.turn_id, game_id=prepared.game.get("game_id"), family=prepared.game.get("game_family"), frontier="before-terra", receipt=receipt, forecast_exposed_to_worker=False, synthetic_features_consumed_by_v2=False, action_changed=False)
        return prepared

    def _discover_pending(self, games: list[dict[str, Any]]) -> None:
        now = time.monotonic()
        for game in games:
            family = str(game["game_family"])
            if family not in self.families:
                continue
            self._last_family_activity = now
            game_id = str(game["game_id"])
            newly_observed_game = game_id not in self.broker.known_active_game_ids()
            if newly_observed_game:
                self._resolve_ambiguous_queue_by_match(family=family, game_id=game_id)
                self._queued_families.discard(family)
                if self.activity_scheduler is not None:
                    receipt = self.activity_scheduler.mark_matched(family, game_id=game_id, now=time.time())
                    if receipt is not None:
                        self._event("activity_game_started", family=family, game_id=game_id, schedule=receipt)
            turn_id = self.broker.turn_id(game)
            if turn_id in self._inflight or turn_id in self._waiting or turn_id in self._ready or turn_id in self._transport_blocked_turns:
                continue
            prior_receipt = self.broker.turn_receipt(turn_id)
            snapshot = self.broker.observe_turn(game)
            receipt = self.broker.turn_receipt(turn_id)
            if receipt is None:
                raise RuntimeError(f"broker failed to create turn receipt: {turn_id}")
            observed_at = _now()
            if prior_receipt is None:
                observation_event = self._event("turn_observed", turn_id=turn_id, game_id=game["game_id"], family=game["game_family"], game=game)
                observed_at = str(observation_event["ts"])
                self._observe_opponent_timing(game, turn_id=turn_id, event=observation_event)
            self._observe_sequence_shadow(game, observed_at=observed_at, source="turn-observed")
            self._capture_rating_canary_game(game, snapshot, observed_at=observed_at)
            self._capture_rating_v3_game(game, snapshot, observed_at=observed_at)
            status = str(receipt["status"])
            created_at = datetime.fromisoformat(str(receipt["created_at"]))
            observed_age = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds())
            first_seen = self._first_seen.setdefault(turn_id, now - observed_age)
            if status == "accepted":
                self._event("accepted_turn_still_pending", turn_id=turn_id, game_id=game["game_id"])
                continue
            if status == "reconciled":
                self._event("reconciled_turn_still_pending", turn_id=turn_id, game_id=game["game_id"])
                continue
            if status in {"prepared", "submitting"} and receipt.get("prepared_action") is not None:
                envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=first_seen + self.turn_deadline_s)
                action = dict(receipt["prepared_action"])
                prior = receipt.get("worker_decision") if isinstance(receipt.get("worker_decision"), dict) else {}
                decision = WorkerDecision(
                    action=action,
                    proposal=prior.get("proposal") if isinstance(prior.get("proposal"), dict) else None,
                    tetrad_update=None,
                    tetrad_transport_issues=[*list(prior.get("tetrad_transport_issues") or []), "recovered prepared action after supervisor restart; discarded its call-local cognitive update"],
                    deterministic_safeguards=list(prior.get("deterministic_safeguards") or []),
                    fallback=bool(prior.get("fallback")),
                    fallback_reason=prior.get("fallback_reason") if isinstance(prior.get("fallback_reason"), str) else None,
                    role=str(prior.get("role") or f"glee_nommd_{family}"),
                    elapsed_s=float(prior.get("elapsed_s") or 0.0),
                    call_metadata=prior.get("call_metadata") if isinstance(prior.get("call_metadata"), dict) else None,
                    selection_branch=str(prior.get("selection_branch") or prior.get("selection_tier") or "single"),
                    branch_receipts=list(prior.get("branch_receipts") or prior.get("tier_receipts") or []),
                    bargaining_advisor_submission=prior.get("bargaining_advisor_submission") if isinstance(prior.get("bargaining_advisor_submission"), dict) else None,
                    negotiation_advisor_submission=prior.get("negotiation_advisor_submission") if isinstance(prior.get("negotiation_advisor_submission"), dict) else None,
                    persuasion_advisor_submission=prior.get("persuasion_advisor_submission") if isinstance(prior.get("persuasion_advisor_submission"), dict) else None,
                )
                self._event("prepared_turn_recovered", turn_id=turn_id, prior_status=status)
                self._event("move_delay_recovery_bypass", turn_id=turn_id, game_id=game["game_id"], reason="prepared action recovered after supervisor restart; submit immediately to preserve deadline")
                self._submit(envelope, decision, already_prepared=True, recovered_prepared=True)
                continue
            advisor_context = None
            advisor_handle = None
            negotiation_context = None
            negotiation_handle = None
            persuasion_context = None
            persuasion_handle = None
            bargaining_intervention_context = None
            bargaining_live_policy = None
            negotiation_live_policy = None
            persuasion_live_policy = None
            if family == "bargaining" and self.bargaining_live_policy_store is not None:
                bargaining_live_policy, created, pointer_error = self.bargaining_live_policy_store.policy_for_game(str(game["game_id"]))
                if created:
                    self._event("bargaining_live_policy_pinned", game_id=game["game_id"], revision=bargaining_live_policy.get("revision"), release_sha256=bargaining_live_policy.get("release_sha256"), pointer_error=pointer_error)
            if family == "bargaining" and self.bargaining_advisor is not None:
                try:
                    advisor_handle = self.bargaining_advisor.forecast_turn(game)
                    advisor_context = advisor_handle.prompt_context
                    self._event("bargaining_v2_turn_forecast", turn_id=turn_id, game_id=game["game_id"], forecast=advisor_context)
                except Exception as error:
                    advisor_context = {"schema_version": 1, "model_version": self.bargaining_advisor_model_version, "status": "unavailable", "fallback_policy_version": "glee-parallel-v17", "error_type": type(error).__name__}
                    self._event("bargaining_v2_forecast_failed", turn_id=turn_id, game_id=game["game_id"], error=f"{type(error).__name__}: {error}")
            if family == "negotiation" and self.negotiation_live_policy_store is not None:
                negotiation_live_policy, created, pointer_error = self.negotiation_live_policy_store.policy_for_game(str(game["game_id"]))
                if created:
                    self._event("negotiation_live_policy_pinned", game_id=game["game_id"], revision=negotiation_live_policy.get("revision"), release_sha256=negotiation_live_policy.get("release_sha256"), pointer_error=pointer_error)
            if family == "negotiation" and self.negotiation_advisor is not None:
                try:
                    negotiation_handle = self.negotiation_advisor.forecast_turn(game, live_policy=negotiation_live_policy) if negotiation_live_policy is not None else self.negotiation_advisor.forecast_turn(game)
                    negotiation_context = negotiation_handle.prompt_context
                    self._event("negotiation_v2_turn_forecast", turn_id=turn_id, game_id=game["game_id"], forecast=negotiation_handle.forecast_receipt)
                except Exception as error:
                    negotiation_context = {"schema_version": 1, "model_version": self.negotiation_advisor_model_version, "status": "unavailable", "fallback_policy_version": "glee-parallel-v23", "error_type": type(error).__name__}
                    self._event("negotiation_v2_forecast_failed", turn_id=turn_id, game_id=game["game_id"], error=f"{type(error).__name__}: {error}")
            if family == "persuasion" and self.persuasion_live_policy_store is not None:
                persuasion_live_policy, created, pointer_error = self.persuasion_live_policy_store.policy_for_game(str(game["game_id"]))
                if created:
                    self._event("persuasion_live_policy_pinned", game_id=game["game_id"], revision=persuasion_live_policy.get("revision"), release_sha256=persuasion_live_policy.get("release_sha256"), pointer_error=pointer_error)
            if family == "persuasion" and self.persuasion_advisor is not None:
                try:
                    persuasion_handle = self.persuasion_advisor.forecast_turn(game, live_policy=persuasion_live_policy) if persuasion_live_policy is not None else self.persuasion_advisor.forecast_turn(game)
                    persuasion_context = persuasion_handle.prompt_context
                    self._event("persuasion_v2_turn_forecast", turn_id=turn_id, game_id=game["game_id"], forecast=persuasion_handle.forecast_receipt)
                except Exception as error:
                    persuasion_context = {"schema_version": 1, "model_version": self.persuasion_advisor_model_version, "status": "unavailable", "fallback_policy_version": "glee-parallel-v24", "error_type": type(error).__name__}
                    self._event("persuasion_v2_forecast_failed", turn_id=turn_id, game_id=game["game_id"], error=f"{type(error).__name__}: {error}")
            family_advisor_context = advisor_context if family == "bargaining" else negotiation_context if family == "negotiation" else persuasion_context
            opponent_decision_forecast = compact_opponent_decision_forecast(snapshot.memory_context.get("opponent_statistical_package"), game, family_advisor_context)
            if opponent_decision_forecast is not None:
                event_kind = "bargaining_statistical_decision_forecast" if family == "bargaining" else "opponent_statistical_decision_forecast"
                self._event(event_kind, turn_id=turn_id, game_id=game["game_id"], family=family, forecast=opponent_decision_forecast, action_changed=False)
            opponent_account_hypothesis = None
            if self.opponent_account_model_reader is not None:
                try:
                    account_assessment = self.opponent_account_model_reader.assess(game)
                    opponent_account_hypothesis = account_assessment.prompt_context
                    self._event("opponent_account_prompt_assessed", turn_id=turn_id, game_id=game["game_id"], family=family, assessment=account_assessment.receipt, action_changed=False)
                except Exception as error:
                    self._event("opponent_account_prompt_failed", turn_id=turn_id, game_id=game["game_id"], family=family, error=f"{type(error).__name__}: {error}", action_changed=False)
            rating_estimate = None
            if family == "negotiation" and isinstance(negotiation_context, Mapping):
                deterministic = negotiation_context.get("deterministic_decision_facts") if isinstance(negotiation_context.get("deterministic_decision_facts"), Mapping) else {}
                rating_estimate = deterministic.get("rating_objective_surrogate")
            elif family == "persuasion" and isinstance(persuasion_context, Mapping):
                rating_estimate = persuasion_context.get("rating_objective_surrogate")
            if isinstance(rating_estimate, Mapping):
                self._event("family_rating_estimate_registered", turn_id=turn_id, game_id=game["game_id"], family=family, frontier="registered-before-model-inference", estimate=rating_estimate, action_changed=False)
            if family == "bargaining" and advisor_handle is not None:
                try:
                    bargaining_intervention_context = build_bargaining_v217_context(game=game, package_context=self._rating_package_context(snapshot), advisor_handle=advisor_handle, advisor_context=advisor_context, rating_canary=self.bargaining_rating_canary, observed_at=observed_at, live_policy=bargaining_live_policy)
                    self._event("bargaining_v217_intervention_forecast", turn_id=turn_id, game_id=game["game_id"], forecast=bargaining_intervention_context, action_changed=False)
                except Exception as error:
                    bargaining_intervention_context = {"contract": "glee-bargaining-intervention-v2.17", "status": "unavailable", "error_type": type(error).__name__}
                    self._event("bargaining_v217_intervention_forecast_failed", turn_id=turn_id, game_id=game["game_id"], error=f"{type(error).__name__}: {error}")
            rating_v3_advisory = self._rating_v3_turn_advisory(turn_id=turn_id, game=game, observed_at=observed_at, bargaining_advisor_handle=advisor_handle, negotiation_advisor_handle=negotiation_handle, negotiation_advisor_context=negotiation_context, persuasion_advisor_context=persuasion_context)
            message_style_profile = self.message_style_policy_store.assigned_profile(game) if self.message_style_policy_store is not None else None
            envelope = TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=first_seen + self.turn_deadline_s, bargaining_advisor_context=advisor_context, bargaining_advisor_handle=advisor_handle, negotiation_advisor_context=negotiation_context, negotiation_advisor_handle=negotiation_handle, negotiation_live_policy=negotiation_live_policy, persuasion_advisor_context=persuasion_context, persuasion_advisor_handle=persuasion_handle, opponent_decision_forecast=opponent_decision_forecast, opponent_account_hypothesis=opponent_account_hypothesis, rating_v3_advisory=rating_v3_advisory, bargaining_intervention_context=bargaining_intervention_context, message_style_profile=message_style_profile)
            envelope = self._register_sequence_shadow_before_terra(envelope, observed_at=observed_at)
            self._waiting[turn_id] = envelope
            self._event("turn_enqueued", turn_id=turn_id, game_id=game["game_id"], family=family, deadline_in_s=self.turn_deadline_s)

    def _dispatch_waiting(self, pool: ThreadPoolExecutor) -> None:
        available = self.max_parallel - len(self._inflight)
        inflight_by_family = Counter(str(envelope.game["game_family"]) for _future, envelope in self._inflight.values())
        ordered = sorted(self._waiting.items(), key=lambda item: item[1].deadline_at_monotonic)
        for turn_id, envelope in ordered:
            remaining = envelope.deadline_at_monotonic - time.monotonic()
            minimum_start_budget = float(getattr(self.worker, "minimum_start_budget_s", self.emergency_margin_s + 5.0))
            if remaining <= minimum_start_budget:
                decision = self.worker.fallback(envelope, f"supervisor fallback: {remaining:.3f}s remains, below the {minimum_start_budget:.3f}s minimum start budget")
                del self._waiting[turn_id]
                self._stage_submission(envelope, decision)
                continue
            if available <= 0:
                continue
            family = str(envelope.game["game_family"])
            if self.family_slots is not None and inflight_by_family[family] >= self.family_slots[family]:
                continue
            future = pool.submit(self.worker.solve, envelope)
            self._inflight[turn_id] = (future, envelope)
            del self._waiting[turn_id]
            available -= 1
            inflight_by_family[family] += 1
            self._event("worker_started", turn_id=turn_id, family=envelope.game["game_family"], deadline_in_s=round(remaining, 6))

    def _collect_workers(self) -> None:
        completed = [(turn_id, future, envelope) for turn_id, (future, envelope) in self._inflight.items() if future.done()]
        completed.sort(key=lambda item: item[2].deadline_at_monotonic)
        for turn_id, future, envelope in completed:
            del self._inflight[turn_id]
            try:
                decision = future.result()
            except Exception as error:
                decision = self.worker.fallback(envelope, f"uncaught worker failure: {type(error).__name__}: {error}")
            self._event("worker_finished", turn_id=turn_id, decision=decision.receipt())
            receipt = self.broker.turn_receipt(turn_id)
            if receipt is not None and receipt["status"] in {"accepted", "reconciled"}:
                self._event("obsolete_worker_discarded", turn_id=turn_id, broker_status=receipt["status"])
                continue
            self._stage_submission(envelope, decision)

    def _prepare_submission(self, envelope: TurnEnvelope, decision: WorkerDecision) -> None:
        advisor_submission = None
        negotiation_submission = None
        persuasion_submission = None
        if envelope.bargaining_advisor_handle is not None:
            try:
                advisor_submission = envelope.bargaining_advisor_handle.submission_prediction(decision.action)
                self._event("bargaining_v2_submission_prediction", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=advisor_submission)
            except Exception as error:
                self._event("bargaining_v2_submission_prediction_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], error=f"{type(error).__name__}: {error}")
        if envelope.negotiation_advisor_handle is not None:
            try:
                negotiation_submission = envelope.negotiation_advisor_handle.submission_prediction(decision.action)
                self._event("negotiation_v2_submission_prediction", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=negotiation_submission)
            except Exception as error:
                self._event("negotiation_v2_submission_prediction_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], error=f"{type(error).__name__}: {error}")
        if envelope.persuasion_advisor_handle is not None:
            try:
                persuasion_submission = envelope.persuasion_advisor_handle.submission_prediction(decision.action)
                self._event("persuasion_v2_submission_prediction", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=persuasion_submission)
            except Exception as error:
                self._event("persuasion_v2_submission_prediction_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], error=f"{type(error).__name__}: {error}")
        worker_receipt = decision.receipt()
        worker_receipt["bargaining_advisor_submission"] = advisor_submission
        worker_receipt["negotiation_advisor_submission"] = negotiation_submission
        worker_receipt["persuasion_advisor_submission"] = persuasion_submission
        intervention = intervention_receipt(envelope, decision)
        if intervention is not None:
            self._event("bargaining_v217_intervention_selected", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], receipt=intervention, action_changed=bool(intervention["interventions"]))
        package_prediction = bargaining_submitted_offer_forecast(self._rating_package_context(envelope.snapshot), envelope.game, decision.action, advisor_submission)
        if package_prediction is not None:
            self._event("bargaining_statistical_package_prediction_registered", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], stage="primary", prediction=package_prediction, registered_before_network_submission=True, action_changed=False)
        self._register_rating_canary_turn(envelope, decision.action, stage="primary")
        self.broker.prepare_turn(envelope.snapshot, worker_receipt, decision.action)

    def _submission_api_call_budget_s(self) -> float:
        timeout = getattr(self.client, "timeout", 10.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 10.0
        return float(timeout) + _MOVE_SUBMISSION_RESOLVER_ALLOWANCE_S

    @staticmethod
    def _transport_error_chain(error: BaseException) -> list[str]:
        chain: list[str] = []
        current: BaseException | None = error
        while current is not None and len(chain) < 6:
            chain.append(type(current).__name__)
            current = current.__cause__ or current.__context__
        return chain

    def _move_with_transport_guard(self, envelope: TurnEnvelope, action: dict[str, Any], *, stage: str) -> dict[str, Any] | object | None:
        turn_id = envelope.snapshot.turn_id
        game_id = str(envelope.game["game_id"])
        remaining_before = envelope.deadline_at_monotonic - time.monotonic()
        if remaining_before <= 0:
            self._event("move_submission_transport_exhausted", contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT, turn_id=turn_id, game_id=game_id, stage=stage, network_attempts=0, remaining_deadline_s=round(remaining_before, 6), reason="original-turn-deadline-elapsed")
            return None
        started = time.monotonic()
        try:
            budget_deadline = envelope.deadline_at_monotonic - self._submission_api_call_budget_s()
            result = self._api_call(operation="move", priority="critical", callback=lambda: self.client.move(game_id, action), wait=True, deadline_monotonic=budget_deadline)
        except GleeAPIBudgetDeferred as error:
            now = time.monotonic()
            self._event("move_submission_transport_exhausted", contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT, turn_id=turn_id, game_id=game_id, stage=stage, network_attempts=0, elapsed_s=round(now - started, 6), remaining_deadline_s=round(envelope.deadline_at_monotonic - now, 6), reason="agent-wide-api-budget-deadline", budget_receipt=error.receipt)
            return None
        except (RequestsConnectionError, RequestsTimeout) as error:
            now = time.monotonic()
            connection_error = isinstance(error, RequestsConnectionError)
            self._event(
                "move_submission_transport_ambiguous",
                contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT,
                client_contract=NON_REPLAYING_POST_TRANSPORT_CONTRACT,
                turn_id=turn_id,
                game_id=game_id,
                stage=stage,
                network_attempts=1,
                elapsed_s=round(now - started, 6),
                remaining_deadline_s=round(envelope.deadline_at_monotonic - now, 6),
                error_chain=self._transport_error_chain(error),
                ambiguity="the peer may have closed the connection after receiving the POST" if connection_error else "the POST may have reached the server before the response timed out",
                replay_policy="do not replay; durably suspend only this turn",
                reason="ambiguous-post-connection-error" if connection_error else "ambiguous-post-timeout",
            )
            return None
        except GleeAPIError as error:
            if _is_game_not_active(error):
                now = time.monotonic()
                self._event("move_submission_terminal_race", contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT, turn_id=turn_id, game_id=game_id, stage=stage, network_attempts=1, elapsed_s=round(now - started, 6), remaining_deadline_s=round(envelope.deadline_at_monotonic - now, 6), status_code=error.status_code, server_code=error.code, server_message=error.message, resolution="reconcile this turn without replay; keep the supervisor alive")
                return _TERMINAL_MOVE_RACE
            if not (_is_transient_server_error(error) or _is_rate_limit(error)):
                raise
            now = time.monotonic()
            ambiguous = _is_transient_server_error(error)
            self._event(
                "move_submission_transport_ambiguous" if ambiguous else "move_submission_transport_exhausted",
                contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT,
                client_contract=NON_REPLAYING_POST_TRANSPORT_CONTRACT,
                turn_id=turn_id,
                game_id=game_id,
                stage=stage,
                network_attempts=1,
                elapsed_s=round(now - started, 6),
                remaining_deadline_s=round(envelope.deadline_at_monotonic - now, 6),
                status_code=error.status_code,
                server_code=error.code,
                server_message=error.message,
                ambiguity="a 5xx response cannot prove that the POST had no server-side effect" if ambiguous else "none; the server rejected the request before move processing",
                replay_policy="do not replay; durably suspend only this turn",
                reason="ambiguous-server-error" if ambiguous else "server-rate-limit-exhausted",
            )
            return None
        if not isinstance(result, dict):
            raise RuntimeError("GLEE move submission returned a non-object response")
        return result

    def _suspend_transport_failed_turn(self, envelope: TurnEnvelope, *, stage: str) -> None:
        turn_id = envelope.snapshot.turn_id
        issue = f"{stage} move submission was suspended after transport outcome could not be safely completed or replayed"
        self.broker.suspend_transport_submission(turn_id, issue=issue)
        self._transport_blocked_turns.add(turn_id)
        self._event("move_submission_suspended", contract=_MOVE_SUBMISSION_TRANSPORT_CONTRACT, turn_id=turn_id, game_id=envelope.game["game_id"], stage=stage, broker_status="transport-suspended", durable=True, policy="keep the family supervisor alive and never replay this prepared move after an ambiguous POST outcome")

    def _submit(self, envelope: TurnEnvelope, decision: WorkerDecision, *, already_prepared: bool = False, recovered_prepared: bool = False) -> None:
        if not already_prepared:
            self._prepare_submission(envelope, decision)
        if recovered_prepared and decision.bargaining_advisor_submission is not None:
            self._event("bargaining_v2_submission_prediction_recovered", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=decision.bargaining_advisor_submission)
        if recovered_prepared and decision.negotiation_advisor_submission is not None:
            self._event("negotiation_v2_submission_prediction_recovered", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=decision.negotiation_advisor_submission)
        if recovered_prepared and decision.persuasion_advisor_submission is not None:
            self._event("persuasion_v2_submission_prediction_recovered", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=decision.persuasion_advisor_submission)
        self.broker.mark_submitting(envelope.snapshot.turn_id)
        result = self._move_with_transport_guard(envelope, decision.action, stage="primary")
        if result is _TERMINAL_MOVE_RACE:
            self.broker.reconcile_terminal_submission(envelope.snapshot.turn_id, issue="server reported that the game was no longer active before this prepared move could be acknowledged")
            self._first_seen.pop(envelope.snapshot.turn_id, None)
            return
        if result is None:
            self._suspend_transport_failed_turn(envelope, stage="primary")
            return
        action = decision.action
        update = decision.tetrad_update
        issues = list(decision.tetrad_transport_issues)
        successful_submission_event = self._event("move_submitted", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], action=action, result=result, supervisor_thread=threading.get_ident())
        if result.get("valid") is False:
            self.broker.record_rejection(envelope.snapshot.turn_id, action, result)
            if (envelope.game.get("game_family") == "negotiation" and envelope.negotiation_advisor_handle is not None) or (envelope.game.get("game_family") == "persuasion" and envelope.persuasion_advisor_handle is not None):
                fallback_decision = self.worker.fallback(envelope, f"server rejected the selected action: {result.get('error')}")
                fallback = fallback_decision.action
                safeguards = fallback_decision.deterministic_safeguards
            else:
                fallback = normalize_action(envelope.game, safe_action(envelope.game))
                fallback, safeguards = apply_deterministic_safeguards(envelope.game, fallback)
            fallback, _fallback_profile = self._realize_message_style(envelope, fallback, stage="server-rejection-fallback")
            if fallback == action:
                raise RuntimeError(f"server rejected the deterministic fallback for {envelope.game['game_family']}: {result.get('error')}")
            bargaining_fallback_prediction = None
            if envelope.bargaining_advisor_handle is not None:
                try:
                    bargaining_fallback_prediction = envelope.bargaining_advisor_handle.submission_prediction(fallback)
                    self._event("bargaining_v2_fallback_prediction", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=bargaining_fallback_prediction)
                except Exception as error:
                    self._event("bargaining_v2_fallback_prediction_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], error=f"{type(error).__name__}: {error}")
            if envelope.negotiation_advisor_handle is not None:
                try:
                    fallback_prediction = envelope.negotiation_advisor_handle.submission_prediction(fallback)
                    self._event("negotiation_v2_fallback_prediction", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=fallback_prediction)
                except Exception as error:
                    self._event("negotiation_v2_fallback_prediction_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], error=f"{type(error).__name__}: {error}")
            if envelope.persuasion_advisor_handle is not None:
                try:
                    fallback_prediction = envelope.persuasion_advisor_handle.submission_prediction(fallback)
                    self._event("persuasion_v2_fallback_prediction", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], prediction=fallback_prediction)
                except Exception as error:
                    self._event("persuasion_v2_fallback_prediction_failed", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], error=f"{type(error).__name__}: {error}")
            package_prediction = bargaining_submitted_offer_forecast(self._rating_package_context(envelope.snapshot), envelope.game, fallback, bargaining_fallback_prediction)
            if package_prediction is not None:
                self._event("bargaining_statistical_package_prediction_registered", turn_id=envelope.snapshot.turn_id, game_id=envelope.game["game_id"], stage="fallback", prediction=package_prediction, registered_before_network_submission=True, action_changed=False)
            self._register_rating_canary_turn(envelope, fallback, stage="fallback")
            retry_result = self._move_with_transport_guard(envelope, fallback, stage="server-rejection-fallback")
            if retry_result is _TERMINAL_MOVE_RACE:
                self.broker.reconcile_terminal_submission(envelope.snapshot.turn_id, issue="server reported that the game was no longer active before the fallback move could be acknowledged")
                self._first_seen.pop(envelope.snapshot.turn_id, None)
                return
            if retry_result is None:
                self._suspend_transport_failed_turn(envelope, stage="server-rejection-fallback")
                return
            fallback_event = self._event("server_rejection_fallback", turn_id=envelope.snapshot.turn_id, rejected_action=action, rejection=result, fallback_action=fallback, result=retry_result)
            if retry_result.get("valid") is False:
                self.broker.record_rejection(envelope.snapshot.turn_id, fallback, retry_result)
                raise RuntimeError(f"server rejected the emergency fallback for {envelope.game['game_family']}: {retry_result.get('error')}")
            action = fallback
            result = retry_result
            successful_submission_event = fallback_event
            update = None
            issues.extend(["cognitive update discarded because server rejected the paired model action", *safeguards])
        self._record_timing_anchor(envelope, submission_event=successful_submission_event, result=result)
        commit = self.broker.commit_accepted(snapshot=envelope.snapshot, action=action, result=result, tetrad_update=update, issues=issues)
        self._event("broker_committed", turn_id=envelope.snapshot.turn_id, receipt=commit)
        first_seen = self._first_seen.pop(envelope.snapshot.turn_id, None)
        if result.get("game_over"):
            response_time_ms = round(max(0.0, time.monotonic() - first_seen) * 1000) if first_seen is not None else None
            final_state = _submitted_final_state(envelope, action, result, response_time_ms=response_time_ms)
            self._observe_sequence_shadow(final_state, observed_at=str(successful_submission_event["ts"]), source="accepted-move-terminal")
            self._register_rating_canary_terminal(final_state, terminal_at=str(successful_submission_event["ts"]))
            self._register_rating_v3_terminal(final_state, terminal_at=str(successful_submission_event["ts"]))
            _atomic_json(self.run_dir / "games" / f"{envelope.game['game_family']}-{envelope.game['game_id']}.json", final_state)
            self.broker.mark_game_completed(str(envelope.game["game_id"]), final_state)
            completion_event = self._event("game_completed", game_id=envelope.game["game_id"], family=envelope.game["game_family"], result=final_state.get("result"), final_state_source="accepted-move-result")
            self._update_opponent_statistical_package(final_state, completion_event)
            self._update_bargaining_advisor(final_state, completion_event)
            self._update_negotiation_advisor(final_state, completion_event)
            self._update_persuasion_advisor(final_state, completion_event)

    def _refresh_known_games(self, pending_game_ids: set[str], *, force: bool = False) -> None:
        now = time.monotonic()
        interval = max(10.0 if force else 15.0, self.poll_interval_s * 5)
        if now - self._last_refresh < interval:
            return
        candidates = sorted(self.broker.known_active_game_ids() - pending_game_ids)
        if not candidates:
            self._last_refresh = now
            return
        game_id = candidates[self._refresh_cursor % len(candidates)]
        self._refresh_cursor += 1
        self._last_refresh = now
        try:
            state = self._api_call(operation="game_state", priority="control", callback=lambda: self.client.game_state(game_id))
        except Exception as error:
            if not _is_deferred_api_control(error):
                raise
            self._event("known_game_refresh_failed", game_id=game_id, candidate_count=len(candidates), error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None)
            return
        refresh_event = self._event("known_game_refreshed", game_id=game_id, candidate_count=len(candidates), terminal=state.get("status") in {"completed", "no_deal"} or state.get("result") is not None)
        if state.get("status") in {"completed", "no_deal"} or state.get("result") is not None:
            self._observe_opponent_timing(state, turn_id=f"{game_id}:terminal", event=refresh_event, terminal=True)
            self._observe_sequence_shadow(state, observed_at=str(refresh_event["ts"]), source="opponent-turn-terminal")
            family = str(state.get("game_family") or "unknown")
            self._register_rating_canary_terminal(state, terminal_at=str(refresh_event["ts"]))
            self._register_rating_v3_terminal(state, terminal_at=str(refresh_event["ts"]))
            _atomic_json(self.run_dir / "games" / f"{family}-{game_id}.json", state)
            self.broker.mark_game_completed(game_id, state)
            completion_event = self._event("game_completed_during_opponent_turn", game_id=game_id, family=family, result=state.get("result"), final_state_source="rate-bounded-game-state")
            self._update_opponent_statistical_package(state, completion_event)
            self._update_bargaining_advisor(state, completion_event)
            self._update_negotiation_advisor(state, completion_event)
            self._update_persuasion_advisor(state, completion_event)
            self._last_family_activity = now

    def run(self) -> dict[str, object]:
        started = time.monotonic()
        self._event("parallel_run_entered", completed_games=sorted(self.broker.completed_game_ids()), max_parallel=self.max_parallel)
        self._sample_memory(force=True)
        try:
            self._reconcile_opponent_statistical_package()
            self._reconcile_rating_canary(force=True)
            self._reconcile_rating_v3(force=True)
            if not self.pause_request_path.is_file():
                try:
                    responses = self._leave_selected_queues()
                    self._event("startup_queues_left", responses=responses)
                except Exception as error:
                    if not _is_deferred_api_control(error):
                        raise
                    self._event("startup_queue_cleanup_deferred", error=f"{type(error).__name__}: {error}", rate_limited=_is_rate_limit(error), locally_deferred=isinstance(error, GleeAPIBudgetDeferred), budget_receipt=error.receipt if isinstance(error, GleeAPIBudgetDeferred) else None)
            self._sync_pause()
            self._sync_activity_target()
            with ThreadPoolExecutor(max_workers=self.max_parallel, thread_name_prefix="glee-model") as pool:
                while True:
                    self._sync_pause()
                    self._sync_activity_target()
                    self._collect_workers()
                    self._release_ready()
                    limit_reason = self._limit_reason(started)
                    if limit_reason is not None:
                        self._enter_drain(limit_reason)
                    pending = self._pending_games()
                    pending_game_ids = {str(game["game_id"]) for game in pending}
                    self._discover_pending(pending)
                    self._reconcile_ambiguous_queues()
                    self._dispatch_waiting(pool)
                    self._release_ready()
                    self._refresh_known_games(pending_game_ids)
                    self._reconcile_rating_canary()
                    self._reconcile_rating_v3()
                    limit_reason = self._limit_reason(started)
                    if limit_reason is not None:
                        self._enter_drain(limit_reason)
                    stats = self._stats(force=self._draining and not self._inflight and not self._waiting)
                    global_active_games = int(stats.get("active_games") or 0)
                    active_games = len(self.broker.known_active_game_ids()) if self.sensor_reader is not None and len(self.families) == 1 else global_active_games
                    if self._draining and not self._inflight and not self._waiting and not self._ready and active_games == 0:
                        self._refresh_known_games(pending_game_ids, force=True)
                        if not self.broker.known_active_game_ids():
                            if global_active_games == 0 or self.sensor_reader is None or len(self.families) != 1 or time.monotonic() - self._last_family_activity >= self.isolated_drain_quiet_s:
                                break
                    self._top_up(active_games)
                    sleep_s = self.poll_interval_s
                    if self.activity_scheduler is not None:
                        until_arrival = self.activity_scheduler.seconds_until_next_arrival(now=time.time())
                        if until_arrival is not None:
                            sleep_s = min(sleep_s, max(0.05, until_arrival))
                    time.sleep(sleep_s)
            self._reconcile_opponent_statistical_package()
            self._reconcile_rating_canary(force=True)
            self._reconcile_rating_v3(force=True)
            final_stats = self._api_call(operation="stats", priority="background", callback=self.client.stats, wait=True)
            final = {
                "schema_version": 1,
                "mode": self._mode(),
                "completed_at": _now(),
                "elapsed_s": round(time.monotonic() - started, 6),
                "agent": {"agent_id": self.initial_stats.get("agent_id"), "agent_name": self.agent_name},
                "model": self.model,
                "effort": self.effort,
                "worker_policy": self.worker_policy,
                "model_timeout_s": self.model_timeout_s,
                "high_effort": self.high_effort,
                "capacity_model_chain": self.worker.manifest_chain if isinstance(self.worker, CapacityFallbackGleeTurnWorker) else None,
                "capacity_retry_policy": {"primary_model": "gpt-5.6-terra", "maximum_same_model_retries": 1, "minimum_retry_budget_s": self.worker.minimum_retry_budget_s, "ineligible_failures": ["timeout", "capacity-without-zero-inference-proof"]} if isinstance(self.worker, CapacityFallbackGleeTurnWorker) else None,
                "emergency_margin_s": self.emergency_margin_s,
                "move_delay_concealment": {"contract": _MOVE_DELAY_CONCEALMENT_CONTRACT, "target_min_s": self.move_delay_min_s, "target_max_s": self.move_delay_max_s, "scope": self.move_delay_scope, **({"distribution": "collector-game-pinned-profile-quantile-with-per-move-jitter"} if self.worker_policy == "collector-local" else {"ki_distribution": "bounded-beta-2-2-total-latency", "hi_distribution": "game-pinned-joint-lexical-timing-persona-when-assigned"})},
                "api_dispatch_smoothing": {"contract": _API_DISPATCH_SMOOTHING_CONTRACT, "admission_dispatch_min_spacing_s": self.admission_dispatch_min_spacing_s, "move_submission_min_spacing_s": self.move_submission_min_spacing_s, "model_call_serialization": False} if self.admission_dispatch_min_spacing_s > 0 or self.move_submission_min_spacing_s > 0 else None,
                "activity_scheduler": self.activity_scheduler.status() if self.activity_scheduler is not None else None,
                "agent_wide_api_rate_limiter": self.api_rate_limiter.status() if self.api_rate_limiter is not None else None,
                "bargaining_live_policy": self.bargaining_live_policy_store.status() if self.bargaining_live_policy_store is not None else None,
                "negotiation_live_policy": self.negotiation_live_policy_store.status() if self.negotiation_live_policy_store is not None else None,
                "persuasion_live_policy": self.persuasion_live_policy_store.status() if self.persuasion_live_policy_store is not None else None,
                "message_style_policy": self.message_style_policy_store.status() if self.message_style_policy_store is not None else None,
                "opponent_timing": {"contract": TIMING_CONTRACT, "counts": self.opponent_timing.counts()},
                "bargaining_opening_policy": self.bargaining_opening_policy,
                "families": list(self.families),
                "family_slots": self.family_slots,
                "stop_after_family": {"family": self.stop_after_family[0], "games": self.stop_after_family[1]} if self.stop_after_family is not None else None,
                "completed_game_ids": sorted(self.broker.completed_game_ids()),
                "completed_games_by_family": self.broker.game_counts_by_family("completed"),
                "broker_summary": self.broker.summary(),
                "memory": self._sample_memory(force=True),
                "bargaining_advisor": self.bargaining_advisor.status() if self.bargaining_advisor is not None else None,
                "negotiation_advisor": self.negotiation_advisor.status() if self.negotiation_advisor is not None else None,
                "persuasion_advisor": self.persuasion_advisor.status() if self.persuasion_advisor is not None else None,
                "bargaining_rating_canary": self.bargaining_rating_canary.status() if self.bargaining_rating_canary is not None else None,
                "rating_v3_advisory": self.rating_v3_advisory.status() if self.rating_v3_advisory is not None else None,
                "opponent_statistical_package": self.opponent_statistical_package_reader.status() if self.opponent_statistical_package_reader is not None and hasattr(self.opponent_statistical_package_reader, "status") else None,
                "opponent_account_model": self.opponent_account_model_reader.receipt if self.opponent_account_model_reader is not None else None,
                "final_stats": final_stats,
            }
            _atomic_json(self.run_dir / "complete.json", final)
            self._event("parallel_run_completed", completed_game_ids=final["completed_game_ids"], final_stats=final["final_stats"])
            return final
        finally:
            try:
                responses = self._leave_selected_queues()
                self._event("all_queues_left", responses=responses)
            except Exception as error:
                self._event("queue_cleanup_failed", error=f"{type(error).__name__}: {error}")
            self.pause_ack_path.unlink(missing_ok=True)
            self.broker.close()
            self.opponent_timing.close()
            if self.opponent_statistical_package_reader is not None and hasattr(self.opponent_statistical_package_reader, "close"):
                self.opponent_statistical_package_reader.close()
            if self.bargaining_rating_canary is not None and hasattr(self.bargaining_rating_canary, "close"):
                self.bargaining_rating_canary.close()
            if self.rating_v3_advisory is not None and hasattr(self.rating_v3_advisory, "close"):
                self.rating_v3_advisory.close()
            if self.api_gateway is not None:
                self.api_gateway.close()
            if self.api_rate_limiter is not None:
                self.api_rate_limiter.close()
