"""Atomic per-game policy pinning for persistent GLEE family processes."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Mapping

from .glee_advisor_contracts import NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE, NEGOTIATION_BUYER_SCALE_AWARE_DECISION_VALUES, NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE, NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_VALUES, NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE, NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_VALUES


BARGAINING_LIVE_POLICY_CONTRACT = "glee-bargaining-live-policy-v1"
BARGAINING_LIVE_POLICY_POINTER_CONTRACT = "glee-bargaining-live-policy-pointer-v1"
BARGAINING_LIVE_POLICY_ASSIGNMENT_CONTRACT = "glee-bargaining-live-policy-assignment-v1"
NEGOTIATION_LIVE_POLICY_CONTRACT = "glee-negotiation-live-policy-v1"
NEGOTIATION_LIVE_POLICY_POINTER_CONTRACT = "glee-negotiation-live-policy-pointer-v1"
NEGOTIATION_LIVE_POLICY_ASSIGNMENT_CONTRACT = "glee-negotiation-live-policy-assignment-v1"
PERSUASION_LIVE_POLICY_CONTRACT = "glee-persuasion-live-policy-v1"
PERSUASION_LIVE_POLICY_POINTER_CONTRACT = "glee-persuasion-live-policy-pointer-v1"
PERSUASION_LIVE_POLICY_ASSIGNMENT_CONTRACT = "glee-persuasion-live-policy-assignment-v1"

_PARAMETER_BOUNDS = {
    "minimum_package_support": (0.0, 1_000_000.0),
    "maximum_package_blend_weight": (0.0, 1.0),
    "minimum_package_value_improvement": (0.0, 1.0),
    "minimum_nonterminal_own_share": (0.0, 1.0),
    "extreme_opponent_share": (0.0, 1.0),
    "continuation_tolerance": (0.0, 1.0),
    "rating_point_catastrophe": (-100.0, 0.0),
}
_FEATURE_VALUES = {
    "selected_curve_comparator": {"nearest-grid", "linear-interpolation"},
    "patient_acceptance_guard": {"continuation-nondominated", "accept-if-next-settlement-no-better", "bidirectional-continuation-coherence"},
}
_OPTIONAL_FEATURE_VALUES = {
    "response_rating_loss_guard": {"bounded-authoritative", "shadow-only"},
    "rating_v3_low_share_guard": {"bounded-authoritative", "shadow-only"},
}

_NEGOTIATION_PARAMETER_BOUNDS = {
    "positive_surplus_minimum_tick": (0.000000001, 1_000_000.0),
    "positive_surplus_relative_tick": (0.000000001, 1.0),
    "reciprocal_concession_match_ratio": (0.000000001, 1.0),
    "shadow_reject_max_current_share": (0.0, 1.0),
    "shadow_reject_min_expected_gain_share": (0.000000001, 1.0),
    "shadow_reject_min_candidate_share": (0.000000001, 1.0),
    "shadow_reject_min_population_games": (1, 1_000_000),
    "one_round_incomplete_seller_markup": (0.000000001, 1000.0),
    "complete_information_min_own_surplus_share": (0.000000001, 0.999999999),
    "stalled_exit_min_opponent_offers": (2, 1_000_000),
    "stalled_exit_recent_offer_window": (2, 1_000_000),
    "stalled_exit_min_reservation_gap_ratio": (0.000000001, 1000.0),
    "stalled_exit_min_projected_opponent_offers": (0.000000001, 1_000_000.0),
}
_NEGOTIATION_OPTIONAL_FEATURE_VALUES = {
    NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_FEATURE: set(NEGOTIATION_POSITIVE_SURPLUS_EXIT_AUTHORITY_VALUES),
    NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_FEATURE: set(NEGOTIATION_ONE_ROUND_INCOMPLETE_SELLER_AUTHORITY_VALUES),
    NEGOTIATION_BUYER_SCALE_AWARE_DECISION_FEATURE: set(NEGOTIATION_BUYER_SCALE_AWARE_DECISION_VALUES),
}

_PERSUASION_PARAMETER_BOUNDS = {
    "authority_min_population_purchased_rows": (1, 1_000_000),
    "buyer_buy_margin_ratio": (0.0, 10.0),
    "buyer_pass_margin_ratio": (0.0, 10.0),
    "buyer_min_current_revealed_purchases": (0, 1_000_000),
    "seller_deception_advantage_margin": (0.0, 1.0),
    "seller_no_response_min_passes": (1, 1_000_000),
    "seller_no_response_max_smoothed_buy_rate": (0.0, 1.0),
}
_PERSUASION_FEATURE_VALUES = {
    "buyer_channel_scope": {"binary-only", "all-channels"},
    "buyer_evidence_boundary": {"credible-interval", "posterior-mean-after-current-support"},
    "buyer_pass_scope": {"final-round-only", "all-rounds"},
}
_PERSUASION_OPTIONAL_FEATURE_VALUES = {
    "seller_no_response_routing": {"advisory-only", "deterministic-quality-consistent"},
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_fsynced(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class BargainingLivePolicyStore:
    """Validate one atomic current pointer and pin its release per game."""

    def __init__(self, *, root: Path, assignments_path: Path) -> None:
        self.root = root.resolve()
        self.current_path = self.root / "current.json"
        self.assignments_path = assignments_path
        self._lock = threading.Lock()
        self._assignments: dict[str, dict[str, object]] = {}
        self._release_cache: dict[str, dict[str, object]] = {}
        self._last_good: tuple[dict[str, object], str, Path] | None = None
        if assignments_path.is_file():
            for line in assignments_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_assignment(record)
                game_id = str(record["game_id"])
                prior = self._assignments.get(game_id)
                if prior is not None and prior != record:
                    raise RuntimeError(f"Bargaining live-policy game {game_id} has conflicting assignments")
                self._assignments[game_id] = record
        self._last_good = self._load_current()

    @staticmethod
    def _validate_release(release: object) -> dict[str, object]:
        if not isinstance(release, dict) or release.get("schema_version") != 1 or release.get("contract") != BARGAINING_LIVE_POLICY_CONTRACT:
            raise RuntimeError("Bargaining live-policy release has an incompatible contract")
        revision = release.get("revision")
        parameters = release.get("parameters")
        features = release.get("features")
        feature_names = set(features) if isinstance(features, dict) else set()
        if not isinstance(revision, str) or not revision or not isinstance(parameters, dict) or set(parameters) != set(_PARAMETER_BOUNDS) or not isinstance(features, dict) or not set(_FEATURE_VALUES) <= feature_names or not feature_names <= set(_FEATURE_VALUES) | set(_OPTIONAL_FEATURE_VALUES):
            raise RuntimeError("Bargaining live-policy release is incomplete")
        for name, (minimum, maximum) in _PARAMETER_BOUNDS.items():
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum:
                raise RuntimeError(f"Bargaining live-policy parameter {name} is outside its contract")
        for name, allowed in _FEATURE_VALUES.items():
            if features[name] not in allowed:
                raise RuntimeError(f"Bargaining live-policy feature {name} is outside its contract")
        for name, allowed in _OPTIONAL_FEATURE_VALUES.items():
            if name in features and features[name] not in allowed:
                raise RuntimeError(f"Bargaining live-policy feature {name} is outside its contract")
        return release

    @staticmethod
    def _validate_assignment(record: object) -> None:
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("contract") != BARGAINING_LIVE_POLICY_ASSIGNMENT_CONTRACT or not isinstance(record.get("game_id"), str) or not isinstance(record.get("release_sha256"), str):
            raise RuntimeError("Bargaining live-policy assignment is invalid")
        BargainingLivePolicyStore._validate_release(record.get("release"))

    def _load_current(self) -> tuple[dict[str, object], str, Path]:
        if not self.current_path.is_file():
            raise RuntimeError(f"Bargaining live-policy pointer is missing: {self.current_path}")
        pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("schema_version") != 1 or pointer.get("contract") != BARGAINING_LIVE_POLICY_POINTER_CONTRACT or not isinstance(pointer.get("release"), str) or not isinstance(pointer.get("release_sha256"), str):
            raise RuntimeError("Bargaining live-policy pointer is invalid")
        release_path = (self.root / str(pointer["release"])).resolve()
        try:
            release_path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("Bargaining live-policy pointer escapes its root") from error
        actual_sha256 = _file_sha256(release_path)
        if actual_sha256 != pointer["release_sha256"]:
            raise RuntimeError("Bargaining live-policy release hash differs from its pointer")
        if actual_sha256 not in self._release_cache:
            release = self._validate_release(json.loads(release_path.read_text(encoding="utf-8")))
            self._release_cache[actual_sha256] = release
        return self._release_cache[actual_sha256], actual_sha256, release_path

    def policy_for_game(self, game_id: str) -> tuple[dict[str, object], bool, str | None]:
        with self._lock:
            prior = self._assignments.get(game_id)
            if prior is not None:
                policy = dict(prior["release"])
                policy["release_sha256"] = prior["release_sha256"]
                return policy, False, str(prior.get("pointer_error")) if prior.get("pointer_error") else None
            pointer_error = None
            try:
                release, release_sha256, release_path = self._load_current()
                self._last_good = release, release_sha256, release_path
            except Exception as error:
                if self._last_good is None:
                    raise
                release, release_sha256, release_path = self._last_good
                pointer_error = f"{type(error).__name__}: {error}"
            record = {
                "schema_version": 1,
                "contract": BARGAINING_LIVE_POLICY_ASSIGNMENT_CONTRACT,
                "game_id": game_id,
                "revision": release["revision"],
                "release_sha256": release_sha256,
                "release_path": str(release_path),
                "release": release,
                "pointer_error": pointer_error,
            }
            _append_fsynced(self.assignments_path, record)
            self._assignments[game_id] = record
            policy = dict(release)
            policy["release_sha256"] = release_sha256
            return policy, True, pointer_error

    def _manifest_receipt(self, release: Mapping[str, object], release_sha256: str, release_path: Path) -> dict[str, object]:
        return {
            "contract": BARGAINING_LIVE_POLICY_CONTRACT,
            "pointer_contract": BARGAINING_LIVE_POLICY_POINTER_CONTRACT,
            "root": str(self.root),
            "current_revision_at_startup": release["revision"],
            "current_release_sha256_at_startup": release_sha256,
            "current_release_path_at_startup": str(release_path),
            "assignment_contract": BARGAINING_LIVE_POLICY_ASSIGNMENT_CONTRACT,
            "assignments_path": str(self.assignments_path),
            "promotion_scope": "new games only under the stable v1 declarative contract",
        }

    def manifest_receipt(self) -> dict[str, object]:
        release, release_sha256, release_path = self._load_current()
        return self._manifest_receipt(release, release_sha256, release_path)

    def status(self) -> dict[str, object]:
        pointer_error = None
        try:
            release, release_sha256, release_path = self._load_current()
            self._last_good = release, release_sha256, release_path
        except Exception as error:
            if self._last_good is None:
                raise
            release, release_sha256, release_path = self._last_good
            pointer_error = f"{type(error).__name__}: {error}"
        counts = Counter(str(record["revision"]) for record in self._assignments.values())
        return {
            **self._manifest_receipt(release, release_sha256, release_path),
            "current_revision": release["revision"],
            "current_release_sha256": release_sha256,
            "current_release_path": str(release_path),
            "current_pointer_error": pointer_error,
            "assigned_games": len(self._assignments),
            "assigned_games_by_revision": dict(sorted(counts.items())),
        }


class NegotiationLivePolicyStore:
    """Validate one atomic Negotiation control pointer and pin its release per game."""

    def __init__(self, *, root: Path, assignments_path: Path) -> None:
        self.root = root.resolve()
        self.current_path = self.root / "current.json"
        self.assignments_path = assignments_path
        self._lock = threading.Lock()
        self._assignments: dict[str, dict[str, object]] = {}
        self._release_cache: dict[str, dict[str, object]] = {}
        self._last_good: tuple[dict[str, object], str, Path] | None = None
        if assignments_path.is_file():
            for line in assignments_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_assignment(record)
                game_id = str(record["game_id"])
                prior = self._assignments.get(game_id)
                if prior is not None and prior != record:
                    raise RuntimeError(f"Negotiation live-policy game {game_id} has conflicting assignments")
                self._assignments[game_id] = record
        self._last_good = self._load_current()

    @staticmethod
    def _validate_release(release: object) -> dict[str, object]:
        if not isinstance(release, dict) or release.get("schema_version") != 1 or release.get("contract") != NEGOTIATION_LIVE_POLICY_CONTRACT:
            raise RuntimeError("Negotiation live-policy release has an incompatible contract")
        revision = release.get("revision")
        parameters = release.get("parameters")
        features = release.get("features")
        if not isinstance(revision, str) or not revision or not isinstance(parameters, dict) or set(parameters) != set(_NEGOTIATION_PARAMETER_BOUNDS):
            raise RuntimeError("Negotiation live-policy release is incomplete")
        if features is not None and (not isinstance(features, dict) or not set(features) <= set(_NEGOTIATION_OPTIONAL_FEATURE_VALUES)):
            raise RuntimeError("Negotiation live-policy features are invalid")
        integer_parameters = {"shadow_reject_min_population_games", "stalled_exit_min_opponent_offers", "stalled_exit_recent_offer_window"}
        for name, (minimum, maximum) in _NEGOTIATION_PARAMETER_BOUNDS.items():
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum or name in integer_parameters and not isinstance(value, int):
                raise RuntimeError(f"Negotiation live-policy parameter {name} is outside its contract")
        if float(parameters["shadow_reject_max_current_share"]) >= float(parameters["shadow_reject_min_candidate_share"]):
            raise RuntimeError("Negotiation live-policy shadow-reject share thresholds overlap")
        if int(parameters["stalled_exit_recent_offer_window"]) > int(parameters["stalled_exit_min_opponent_offers"]):
            raise RuntimeError("Negotiation live-policy stalled-exit window exceeds its support threshold")
        for name, allowed in _NEGOTIATION_OPTIONAL_FEATURE_VALUES.items():
            if isinstance(features, dict) and name in features and features[name] not in allowed:
                raise RuntimeError(f"Negotiation live-policy feature {name} is outside its contract")
        return release

    @staticmethod
    def _validate_assignment(record: object) -> None:
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("contract") != NEGOTIATION_LIVE_POLICY_ASSIGNMENT_CONTRACT or not isinstance(record.get("game_id"), str) or not isinstance(record.get("release_sha256"), str):
            raise RuntimeError("Negotiation live-policy assignment is invalid")
        NegotiationLivePolicyStore._validate_release(record.get("release"))

    def _load_current(self) -> tuple[dict[str, object], str, Path]:
        if not self.current_path.is_file():
            raise RuntimeError(f"Negotiation live-policy pointer is missing: {self.current_path}")
        pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("schema_version") != 1 or pointer.get("contract") != NEGOTIATION_LIVE_POLICY_POINTER_CONTRACT or not isinstance(pointer.get("release"), str) or not isinstance(pointer.get("release_sha256"), str):
            raise RuntimeError("Negotiation live-policy pointer is invalid")
        release_path = (self.root / str(pointer["release"])).resolve()
        try:
            release_path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("Negotiation live-policy pointer escapes its root") from error
        actual_sha256 = _file_sha256(release_path)
        if actual_sha256 != pointer["release_sha256"]:
            raise RuntimeError("Negotiation live-policy release hash differs from its pointer")
        if actual_sha256 not in self._release_cache:
            self._release_cache[actual_sha256] = self._validate_release(json.loads(release_path.read_text(encoding="utf-8")))
        return self._release_cache[actual_sha256], actual_sha256, release_path

    def policy_for_game(self, game_id: str) -> tuple[dict[str, object], bool, str | None]:
        with self._lock:
            prior = self._assignments.get(game_id)
            if prior is not None:
                policy = dict(prior["release"])
                policy["release_sha256"] = prior["release_sha256"]
                return policy, False, str(prior.get("pointer_error")) if prior.get("pointer_error") else None
            pointer_error = None
            try:
                release, release_sha256, release_path = self._load_current()
                self._last_good = release, release_sha256, release_path
            except Exception as error:
                if self._last_good is None:
                    raise
                release, release_sha256, release_path = self._last_good
                pointer_error = f"{type(error).__name__}: {error}"
            record = {
                "schema_version": 1,
                "contract": NEGOTIATION_LIVE_POLICY_ASSIGNMENT_CONTRACT,
                "game_id": game_id,
                "revision": release["revision"],
                "release_sha256": release_sha256,
                "release_path": str(release_path),
                "release": release,
                "pointer_error": pointer_error,
            }
            _append_fsynced(self.assignments_path, record)
            self._assignments[game_id] = record
            policy = dict(release)
            policy["release_sha256"] = release_sha256
            return policy, True, pointer_error

    def _manifest_receipt(self, release: Mapping[str, object], release_sha256: str, release_path: Path) -> dict[str, object]:
        return {
            "contract": NEGOTIATION_LIVE_POLICY_CONTRACT,
            "pointer_contract": NEGOTIATION_LIVE_POLICY_POINTER_CONTRACT,
            "root": str(self.root),
            "current_revision_at_startup": release["revision"],
            "current_release_sha256_at_startup": release_sha256,
            "current_release_path_at_startup": str(release_path),
            "assignment_contract": NEGOTIATION_LIVE_POLICY_ASSIGNMENT_CONTRACT,
            "assignments_path": str(self.assignments_path),
            "promotion_scope": "new games only under the stable v1 narrow-control contract",
        }

    def manifest_receipt(self) -> dict[str, object]:
        release, release_sha256, release_path = self._load_current()
        return self._manifest_receipt(release, release_sha256, release_path)

    def status(self) -> dict[str, object]:
        pointer_error = None
        try:
            release, release_sha256, release_path = self._load_current()
            self._last_good = release, release_sha256, release_path
        except Exception as error:
            if self._last_good is None:
                raise
            release, release_sha256, release_path = self._last_good
            pointer_error = f"{type(error).__name__}: {error}"
        counts = Counter(str(record["revision"]) for record in self._assignments.values())
        return {
            **self._manifest_receipt(release, release_sha256, release_path),
            "current_revision": release["revision"],
            "current_release_sha256": release_sha256,
            "current_release_path": str(release_path),
            "current_pointer_error": pointer_error,
            "assigned_games": len(self._assignments),
            "assigned_games_by_revision": dict(sorted(counts.items())),
        }


class PersuasionLivePolicyStore:
    """Validate one atomic Persuasion control pointer and pin its release per game."""

    def __init__(self, *, root: Path, assignments_path: Path) -> None:
        self.root = root.resolve()
        self.current_path = self.root / "current.json"
        self.assignments_path = assignments_path
        self._lock = threading.Lock()
        self._assignments: dict[str, dict[str, object]] = {}
        self._release_cache: dict[str, dict[str, object]] = {}
        self._last_good: tuple[dict[str, object], str, Path] | None = None
        if assignments_path.is_file():
            for line in assignments_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_assignment(record)
                game_id = str(record["game_id"])
                prior = self._assignments.get(game_id)
                if prior is not None and prior != record:
                    raise RuntimeError(f"Persuasion live-policy game {game_id} has conflicting assignments")
                self._assignments[game_id] = record
        self._last_good = self._load_current()

    @staticmethod
    def _validate_release(release: object) -> dict[str, object]:
        if not isinstance(release, dict) or release.get("schema_version") != 1 or release.get("contract") != PERSUASION_LIVE_POLICY_CONTRACT:
            raise RuntimeError("Persuasion live-policy release has an incompatible contract")
        revision = release.get("revision")
        parameters = release.get("parameters")
        features = release.get("features")
        allowed_features = set(_PERSUASION_FEATURE_VALUES) | set(_PERSUASION_OPTIONAL_FEATURE_VALUES)
        if not isinstance(revision, str) or not revision or not isinstance(parameters, dict) or set(parameters) != set(_PERSUASION_PARAMETER_BOUNDS) or not isinstance(features, dict) or not set(_PERSUASION_FEATURE_VALUES).issubset(features) or not set(features).issubset(allowed_features):
            raise RuntimeError("Persuasion live-policy release is incomplete")
        integer_parameters = {"authority_min_population_purchased_rows", "buyer_min_current_revealed_purchases", "seller_no_response_min_passes"}
        for name, (minimum, maximum) in _PERSUASION_PARAMETER_BOUNDS.items():
            value = parameters[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= float(value) <= maximum or name in integer_parameters and not isinstance(value, int):
                raise RuntimeError(f"Persuasion live-policy parameter {name} is outside its contract")
        for name, allowed in _PERSUASION_FEATURE_VALUES.items():
            if features[name] not in allowed:
                raise RuntimeError(f"Persuasion live-policy feature {name} is outside its contract")
        for name, allowed in _PERSUASION_OPTIONAL_FEATURE_VALUES.items():
            if name in features and features[name] not in allowed:
                raise RuntimeError(f"Persuasion live-policy feature {name} is outside its contract")
        return release

    @staticmethod
    def _validate_assignment(record: object) -> None:
        if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("contract") != PERSUASION_LIVE_POLICY_ASSIGNMENT_CONTRACT or not isinstance(record.get("game_id"), str) or not isinstance(record.get("release_sha256"), str):
            raise RuntimeError("Persuasion live-policy assignment is invalid")
        PersuasionLivePolicyStore._validate_release(record.get("release"))

    def _load_current(self) -> tuple[dict[str, object], str, Path]:
        if not self.current_path.is_file():
            raise RuntimeError(f"Persuasion live-policy pointer is missing: {self.current_path}")
        pointer = json.loads(self.current_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("schema_version") != 1 or pointer.get("contract") != PERSUASION_LIVE_POLICY_POINTER_CONTRACT or not isinstance(pointer.get("release"), str) or not isinstance(pointer.get("release_sha256"), str):
            raise RuntimeError("Persuasion live-policy pointer is invalid")
        release_path = (self.root / str(pointer["release"])).resolve()
        try:
            release_path.relative_to(self.root)
        except ValueError as error:
            raise RuntimeError("Persuasion live-policy pointer escapes its root") from error
        actual_sha256 = _file_sha256(release_path)
        if actual_sha256 != pointer["release_sha256"]:
            raise RuntimeError("Persuasion live-policy release hash differs from its pointer")
        if actual_sha256 not in self._release_cache:
            self._release_cache[actual_sha256] = self._validate_release(json.loads(release_path.read_text(encoding="utf-8")))
        return self._release_cache[actual_sha256], actual_sha256, release_path

    def policy_for_game(self, game_id: str) -> tuple[dict[str, object], bool, str | None]:
        with self._lock:
            prior = self._assignments.get(game_id)
            if prior is not None:
                policy = dict(prior["release"])
                policy["release_sha256"] = prior["release_sha256"]
                return policy, False, str(prior.get("pointer_error")) if prior.get("pointer_error") else None
            pointer_error = None
            try:
                release, release_sha256, release_path = self._load_current()
                self._last_good = release, release_sha256, release_path
            except Exception as error:
                if self._last_good is None:
                    raise
                release, release_sha256, release_path = self._last_good
                pointer_error = f"{type(error).__name__}: {error}"
            record = {
                "schema_version": 1,
                "contract": PERSUASION_LIVE_POLICY_ASSIGNMENT_CONTRACT,
                "game_id": game_id,
                "revision": release["revision"],
                "release_sha256": release_sha256,
                "release_path": str(release_path),
                "release": release,
                "pointer_error": pointer_error,
            }
            _append_fsynced(self.assignments_path, record)
            self._assignments[game_id] = record
            policy = dict(release)
            policy["release_sha256"] = release_sha256
            return policy, True, pointer_error

    def _manifest_receipt(self, release: Mapping[str, object], release_sha256: str, release_path: Path) -> dict[str, object]:
        return {
            "contract": PERSUASION_LIVE_POLICY_CONTRACT,
            "pointer_contract": PERSUASION_LIVE_POLICY_POINTER_CONTRACT,
            "root": str(self.root),
            "current_revision_at_startup": release["revision"],
            "current_release_sha256_at_startup": release_sha256,
            "current_release_path_at_startup": str(release_path),
            "assignment_contract": PERSUASION_LIVE_POLICY_ASSIGNMENT_CONTRACT,
            "assignments_path": str(self.assignments_path),
            "promotion_scope": "new games only under the stable v1 bounded-control contract",
        }

    def manifest_receipt(self) -> dict[str, object]:
        release, release_sha256, release_path = self._load_current()
        return self._manifest_receipt(release, release_sha256, release_path)

    def status(self) -> dict[str, object]:
        pointer_error = None
        try:
            release, release_sha256, release_path = self._load_current()
            self._last_good = release, release_sha256, release_path
        except Exception as error:
            if self._last_good is None:
                raise
            release, release_sha256, release_path = self._last_good
            pointer_error = f"{type(error).__name__}: {error}"
        counts = Counter(str(record["revision"]) for record in self._assignments.values())
        return {
            **self._manifest_receipt(release, release_sha256, release_path),
            "current_revision": release["revision"],
            "current_release_sha256": release_sha256,
            "current_release_path": str(release_path),
            "current_pointer_error": pointer_error,
            "assigned_games": len(self._assignments),
            "assigned_games_by_revision": dict(sorted(counts.items())),
        }
