"""Portable feature extraction and frozen linear choice inference for GLEE selectors."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .glee_selector_backend import LocalCallableSelectorBackend, SELECTOR_BACKEND_REQUEST_CONTRACT, local_result, reference_expected_value


LOCAL_SELECTOR_FEATURE_CONTRACT = "glee-local-selector-candidate-features-v1"
LOCAL_SELECTOR_MODEL_CONTRACT = "glee-local-selector-conditional-logit-v1"
LOCAL_SELECTOR_ADMISSIBILITY_CONTRACT = "glee-local-selector-nonnegative-fallback-proxy-v1"
_TOKEN_BINS = 32
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_PROVENANCE_KEYS = frozenset({"action_sha256", "candidate_index", "candidate_set_sha256", "contract", "path", "receipt", "revision", "sha256", "status", "ts", "updated_at"})


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _normalized_text(value: object) -> str:
    return " ".join(_TOKEN_RE.findall(str(value).casefold()))[:120]


def _add_numeric(output: dict[str, float], name: str, value: object, *, probability: bool = False) -> None:
    number = _finite(value)
    if number is None:
        return
    output[name] = max(0.0, min(1.0, number)) if probability else _signed_log1p(number)


def _flatten_numeric(value: object, *, prefix: str, output: dict[str, float], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, bool):
        output[prefix] = float(value)
        return
    number = _finite(value)
    if number is not None:
        _add_numeric(output, prefix, number, probability="probab" in prefix or prefix.endswith(".confidence"))
        return
    if not isinstance(value, Mapping):
        return
    for key, child in sorted(value.items(), key=lambda item: str(item[0])):
        token = str(key)
        if token in _PROVENANCE_KEYS:
            continue
        _flatten_numeric(child, prefix=f"{prefix}.{token}" if prefix else token, output=output, depth=depth + 1)


def _message_features(message: str, output: dict[str, float]) -> None:
    tokens = _TOKEN_RE.findall(message.casefold())
    characters = max(1, len(message))
    output["message.present"] = 1.0
    output["message.characters"] = math.log1p(len(message))
    output["message.words"] = math.log1p(len(tokens))
    output["message.digit_fraction"] = sum(character.isdigit() for character in message) / characters
    output["message.upper_fraction"] = sum(character.isupper() for character in message) / characters
    output["message.question"] = float("?" in message)
    output["message.exclamation"] = float("!" in message)
    output["message.negation"] = float(any(token in {"no", "not", "never", "cannot", "can't", "won't"} for token in tokens))
    if not tokens:
        return
    scale = 1.0 / math.sqrt(len(tokens))
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % _TOKEN_BINS
        sign = 1.0 if digest[2] & 1 else -1.0
        name = f"message.token_hash.{bucket:02d}"
        output[name] = output.get(name, 0.0) + sign * scale


def _forecast_features(forecast: Mapping[str, object], output: dict[str, float]) -> None:
    labels = forecast.get("labels")
    if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
        for key, values in sorted(forecast.items()):
            if not key.endswith("probabilities") or not isinstance(values, list) or len(values) != len(labels):
                continue
            for label, value in zip(labels, values, strict=True):
                _add_numeric(output, f"forecast.{key}.{label}", value, probability=True)
            probabilities = [number for value in values if (number := _finite(value)) is not None and number > 0.0]
            if len(probabilities) == len(values):
                output[f"forecast.{key}.entropy"] = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
                output[f"forecast.{key}.maximum"] = max(probabilities, default=0.0)
    for key, value in sorted(forecast.items()):
        if key in {"labels", "response_probabilities", "sequence_probabilities", "engineered_probabilities"} or key in _PROVENANCE_KEYS:
            continue
        _flatten_numeric(value, prefix=f"forecast.{key}", output=output)
    sequence = forecast.get("sequence_probabilities")
    engineered = forecast.get("engineered_probabilities")
    if isinstance(sequence, list) and isinstance(engineered, list) and len(sequence) == len(engineered):
        pairs = [(_finite(left), _finite(right)) for left, right in zip(sequence, engineered, strict=True)]
        if all(left is not None and right is not None for left, right in pairs):
            output["forecast.sequence_engineered_l1"] = sum(abs(float(left) - float(right)) for left, right in pairs) / max(1, len(pairs))


def _aligned_payload(payload: Mapping[str, object], name: str, value_name: str) -> dict[int, Mapping[str, object]]:
    surface = payload.get(name)
    rows = surface.get("rows") if isinstance(surface, Mapping) else None
    aligned: dict[int, Mapping[str, object]] = {}
    if not isinstance(rows, list):
        return aligned
    for row in rows:
        if not isinstance(row, Mapping) or isinstance(row.get("candidate_index"), bool) or not isinstance(row.get("candidate_index"), int) or not isinstance(row.get(value_name), Mapping):
            continue
        aligned[int(row["candidate_index"])] = row[value_name]
    return aligned


def _candidate_rows(wire: Mapping[str, object]) -> tuple[str, Mapping[str, object], list[Mapping[str, object]], list[str], str]:
    if wire.get("contract") != SELECTOR_BACKEND_REQUEST_CONTRACT:
        raise ValueError("local selector received the wrong request contract")
    family = str(wire.get("family") or "")
    payload = _mapping(wire.get("selector_payload"), name="selector payload")
    candidate_set = _mapping(payload.get("candidate_set"), name="candidate set")
    raw_candidates = candidate_set.get("candidates")
    candidate_ids = wire.get("candidate_ids")
    fallback_id = wire.get("fallback_candidate_id")
    if family not in {"bargaining", "negotiation", "persuasion"} or not isinstance(raw_candidates, list) or not raw_candidates or not isinstance(candidate_ids, list) or not isinstance(fallback_id, str):
        raise ValueError("local selector request has an invalid family or candidate frontier")
    candidates = sorted((_mapping(candidate, name="candidate") for candidate in raw_candidates), key=lambda candidate: int(candidate.get("candidate_index", -1)))
    if [candidate.get("candidate_index") for candidate in candidates] != list(range(len(candidates))):
        raise ValueError("local selector candidates must have contiguous zero-based indexes")
    ids = [str(candidate.get("action_sha256") or "") for candidate in candidates]
    if ids != [str(value) for value in candidate_ids] or len(set(ids)) != len(ids) or fallback_id not in ids:
        raise ValueError("local selector candidate identifiers are misaligned")
    return family, payload, candidates, ids, fallback_id


def candidate_feature_maps(wire: Mapping[str, object]) -> list[dict[str, float]]:
    """Project one immutable selector request into aligned candidate-specific sparse features."""
    family, payload, candidates, ids, fallback_id = _candidate_rows(wire)
    forecasts = _aligned_payload(payload, "conditional_opponent_response_surface", "forecast")
    evidence = _aligned_payload(payload, "family_candidate_decision_evidence", "evidence")
    base: list[dict[str, float]] = []
    for candidate, candidate_id in zip(candidates, ids, strict=True):
        index = int(candidate["candidate_index"])
        action = _mapping(candidate.get("action"), name="candidate action")
        features: dict[str, float] = {"candidate.is_fallback": float(candidate_id == fallback_id)}
        for key, value in sorted(action.items()):
            if key == "message" and isinstance(value, str):
                _message_features(value, features)
            elif isinstance(value, bool):
                features[f"action.boolean.{key}"] = float(value)
            elif _finite(value) is not None:
                _add_numeric(features, f"action.numeric.{key}", value)
            elif isinstance(value, str):
                normalized = _normalized_text(value)
                if normalized:
                    features[f"action.category.{key}={normalized}"] = 1.0
        safeguards = candidate.get("planner_candidate_safeguards")
        if isinstance(safeguards, list):
            for safeguard in safeguards:
                normalized = _normalized_text(safeguard)
                if normalized:
                    features[f"candidate.safeguard={normalized}"] = 1.0
        if index in forecasts:
            _forecast_features(forecasts[index], features)
        if index in evidence:
            _flatten_numeric(evidence[index], prefix="evidence", output=features)
        reference, _basis = reference_expected_value(family=family, payload=payload, candidate_index=index)
        if reference is not None:
            features["reference.present"] = 1.0
            features["reference.value"] = _signed_log1p(reference)
        base.append(features)
    varying = sorted({name for features in base for name in features if len({row.get(name, 0.0) for row in base}) > 1})
    for name in varying:
        values = [features.get(name, 0.0) for features in base]
        center = sum(values) / len(values)
        scale = max(abs(value - center) for value in values)
        if scale <= 1e-12:
            continue
        for features, value in zip(base, values, strict=True):
            features[f"relative.{name}"] = (value - center) / scale
    turn = payload.get("authenticated_turn")
    state = turn.get("visible_game_state") if isinstance(turn, Mapping) and isinstance(turn.get("visible_game_state"), Mapping) else None
    round_number = _finite(state.get("round")) if isinstance(state, Mapping) else None
    total_rounds = _finite(state.get("total_rounds")) if isinstance(state, Mapping) else None
    if round_number is not None and total_rounds is not None and total_rounds > 1:
        progress = max(0.0, min(1.0, (round_number - 1.0) / (total_rounds - 1.0)))
        for features in base:
            for name, value in list(features.items()):
                if name.startswith(("reference.", "forecast.response_probabilities.", "forecast.sequence_probabilities.", "candidate.is_fallback")):
                    features[f"interaction.round_progress.{name}"] = progress * value
    return base


def admissible_candidate_ids(wire: Mapping[str, object], *, tolerance: float = 1e-12) -> tuple[set[str], dict[str, float | None]]:
    """Restrict learned choice to candidates whose bounded proxy does not trail fallback."""
    family, payload, candidates, ids, fallback_id = _candidate_rows(wire)
    scores = {candidate_id: reference_expected_value(family=family, payload=payload, candidate_index=int(candidate["candidate_index"]))[0] for candidate, candidate_id in zip(candidates, ids, strict=True)}
    fallback = scores[fallback_id]
    if fallback is None:
        return {fallback_id}, scores
    eligible = {candidate_id for candidate_id, score in scores.items() if score is not None and score + tolerance >= fallback}
    return eligible or {fallback_id}, scores


class FrozenLinearSelector:
    """Standard-library inference for one hash-pinned pooled-plus-family conditional-logit release."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        self.model_sha256 = _file_sha256(self.model_path)
        self.model = json.loads(self.model_path.read_text(encoding="utf-8"))
        if self.model.get("contract") != LOCAL_SELECTOR_MODEL_CONTRACT or self.model.get("feature_contract") != LOCAL_SELECTOR_FEATURE_CONTRACT:
            raise ValueError("unsupported local selector model release")
        self.feature_names = tuple(str(value) for value in self.model.get("feature_names", []))
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("local selector feature vocabulary is empty or duplicated")
        scaler = _mapping(self.model.get("scaler"), name="selector scaler")
        self.means = tuple(float(value) for value in scaler.get("means", []))
        self.scales = tuple(float(value) for value in scaler.get("scales", []))
        weights = _mapping(self.model.get("weights"), name="selector weights")
        self.global_weights = tuple(float(value) for value in weights.get("global", []))
        residuals = _mapping(weights.get("family_residuals"), name="selector family residuals")
        self.family_residuals = {str(family): tuple(float(value) for value in values) for family, values in residuals.items() if isinstance(values, list)}
        expected = len(self.feature_names)
        if len(self.means) != expected or len(self.scales) != expected or len(self.global_weights) != expected or set(self.family_residuals) != {"bargaining", "negotiation", "persuasion"} or any(len(values) != expected for values in self.family_residuals.values()) or any(value <= 0 or not math.isfinite(value) for value in self.scales):
            raise ValueError("local selector release dimensions are inconsistent")

    @property
    def backend_id(self) -> str:
        return str(self.model.get("release_id") or "local-linear-selector-v1")

    def _score(self, features: Mapping[str, float], family: str) -> float:
        residual = self.family_residuals[family]
        total = 0.0
        for index, name in enumerate(self.feature_names):
            value = (features.get(name, 0.0) - self.means[index]) / self.scales[index]
            total += value * (self.global_weights[index] + residual[index])
        return total

    def __call__(self, wire: Mapping[str, object]) -> dict[str, object]:
        family, _payload, _candidates, ids, fallback_id = _candidate_rows(wire)
        features = candidate_feature_maps(wire)
        eligible, reference_scores = admissible_candidate_ids(wire)
        scored = [(self._score(row, family), candidate_id) for row, candidate_id in zip(features, ids, strict=True) if candidate_id in eligible]
        if not scored:
            raise ValueError("local selector admissibility gate produced no candidate")
        scored.sort(key=lambda item: (-item[0], item[1] != fallback_id, item[1]))
        selected_id = scored[0][1]
        result = local_result(request=wire, candidate_id=selected_id)
        result["policy_receipt"] = {
            "contract": LOCAL_SELECTOR_ADMISSIBILITY_CONTRACT,
            "release_id": self.backend_id,
            "model_sha256": self.model_sha256,
            "eligible_candidate_ids": sorted(eligible),
            "reference_scores": reference_scores,
            "selected_model_score": round(scored[0][0], 9),
        }
        return result


class LocalLinearSelectorBackend(LocalCallableSelectorBackend):
    """Strict backend adapter for one frozen portable selector release."""

    def __init__(self, *, model_path: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = FrozenLinearSelector(model_path)
        super().__init__(selector=self.policy, backend_id=self.policy.backend_id, clock=clock)

    @property
    def manifest_receipt(self) -> Mapping[str, object]:
        return {
            "contract": SELECTOR_BACKEND_REQUEST_CONTRACT,
            "backend_id": self.policy.backend_id,
            "model_contract": LOCAL_SELECTOR_MODEL_CONTRACT,
            "feature_contract": LOCAL_SELECTOR_FEATURE_CONTRACT,
            "admissibility_contract": LOCAL_SELECTOR_ADMISSIBILITY_CONTRACT,
            "model_path": str(self.policy.model_path),
            "model_sha256": self.policy.model_sha256,
            "attempts_per_turn": 1,
            "failure": "caller-owned deterministic fallback",
        }
