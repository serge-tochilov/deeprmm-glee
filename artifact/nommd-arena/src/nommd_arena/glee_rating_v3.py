"""Causal structural rating-delta model for all GLEE game families."""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .glee_joint_rating_analysis import FAMILY_BASE_FEATURES, _terminal_view, eta_for_game_count, predict_display_delta
from .glee_joint_rating_v2_analysis import empirical_midrank, shrinkage_weight
from .glee_negotiation_rating_v2_4 import RidgeRatingModel


RATING_V3_CONTRACT = "glee-rating-model-v3"
RATING_V3_MODEL_VERSION = "v3.0-structural-causal-rank"
RATING_V3_SCHEMA_VERSION = 1
RATING_V3_ETA_SCHEDULE = {"name": "constant-0.002", "kind": "constant", "floor": 0.002}
RATING_V3_RECENT_WINDOWS_S = (21600, 86400)
RATING_V3_RESIDUAL_FEATURES = (
    "bias",
    "base_delta",
    "base_delta_abs",
    "base_delta_signed_square",
    "structural_percentile_centered",
    "rank_shift",
    "recent_rank_shift",
    "log_global_support",
    "log_24h_support",
    "log_6h_support",
    "pregame_rating_scaled",
    "pregame_rating_squared",
    "log_game_count",
    "opponent_rating_known",
    "opponent_rating_scaled",
    "rating_gap_scaled",
    "family_median_known",
    "family_median_scaled",
    "traffic_300_log",
    "traffic_1800_log",
    "fleet_300_log",
    "fleet_1800_log",
    "identity_known",
)


def _finite(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    return numeric if math.isfinite(numeric) else default


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("rating v3 timestamp must be finite numeric seconds")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("rating v3 timestamp must be finite numeric seconds")
    return result


@dataclass(frozen=True)
class RankBlend:
    """Selected shrinkage strengths for global and time-local exact-cell ranks."""

    global_alpha: float
    local_24h_alpha: float | None
    local_6h_alpha: float | None

    def __post_init__(self) -> None:
        values = (self.global_alpha, self.local_24h_alpha, self.local_6h_alpha)
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("rank shrinkage alphas must be positive")

    def as_dict(self) -> dict[str, float | None]:
        return {"global_alpha": self.global_alpha, "local_24h_alpha": self.local_24h_alpha, "local_6h_alpha": self.local_6h_alpha}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RankBlend:
        return cls(global_alpha=float(value["global_alpha"]), local_24h_alpha=float(value["local_24h_alpha"]) if value.get("local_24h_alpha") is not None else None, local_6h_alpha=float(value["local_6h_alpha"]) if value.get("local_6h_alpha") is not None else None)


@dataclass(frozen=True)
class SignedMagnitudeCalibrator:
    """Positive scales fitted independently by predicted sign without changing direction."""

    positive_scale: float = 1.0
    negative_scale: float = 1.0
    positive_support: int = 0
    negative_support: int = 0

    def __post_init__(self) -> None:
        if self.positive_scale <= 0 or self.negative_scale <= 0:
            raise ValueError("signed magnitude scales must be positive")
        if self.positive_support < 0 or self.negative_support < 0:
            raise ValueError("signed magnitude support cannot be negative")

    def apply(self, prediction: float) -> float:
        if prediction > 0:
            return prediction * self.positive_scale
        if prediction < 0:
            return prediction * self.negative_scale
        return 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {"positive_scale": self.positive_scale, "negative_scale": self.negative_scale, "positive_support": self.positive_support, "negative_support": self.negative_support}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SignedMagnitudeCalibrator:
        return cls(positive_scale=float(value["positive_scale"]), negative_scale=float(value["negative_scale"]), positive_support=int(value.get("positive_support") or 0), negative_support=int(value.get("negative_support") or 0))

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, object]], *, lower: float = 0.25, upper: float = 2.0) -> SignedMagnitudeCalibrator:
        if not 0 < lower <= 1 <= upper:
            raise ValueError("signed calibration bounds must straddle one")

        def scale(sign: int) -> tuple[float, int]:
            selected = [row for row in rows if (float(row["prediction"]) > 0) == (sign > 0) and float(row["prediction"]) != 0]
            numerator = sum(float(row["prediction"]) * float(row["actual"]) for row in selected)
            denominator = sum(float(row["prediction"]) ** 2 for row in selected)
            fitted = numerator / denominator if denominator > 0 else 1.0
            return min(upper, max(lower, fitted if fitted > 0 else 1.0)), len(selected)

        positive, positive_support = scale(1)
        negative, negative_support = scale(-1)
        return cls(positive_scale=positive, negative_scale=negative, positive_support=positive_support, negative_support=negative_support)


class CausalPayoffIndex:
    """Exact-cell payoff references that expose only observations earlier than a query."""

    def __init__(self, *, global_references: Mapping[str, Sequence[float]] | None = None, recent_observations: Mapping[str, Sequence[Sequence[float]]] | None = None) -> None:
        self.global_references = {str(key): sorted(float(number) for number in values) for key, values in dict(global_references or {}).items()}
        self.recent_observations = {str(key): sorted((float(pair[0]), float(pair[1])) for pair in values) for key, values in dict(recent_observations or {}).items()}

    def observe(self, configuration: str, payoff: float, observed_at: float) -> None:
        key = str(configuration)
        numeric_payoff = float(payoff)
        numeric_at = _timestamp(observed_at)
        bisect.insort(self.global_references.setdefault(key, []), numeric_payoff)
        recent = self.recent_observations.setdefault(key, [])
        if recent and numeric_at < recent[-1][0]:
            bisect.insort(recent, (numeric_at, numeric_payoff))
        else:
            recent.append((numeric_at, numeric_payoff))

    def ranks(self, configuration: str, payoff: float, observed_at: float) -> dict[str, float | int | None]:
        key = str(configuration)
        timestamp = _timestamp(observed_at)
        global_rank, global_support = empirical_midrank(float(payoff), self.global_references.get(key, ())) if key in self.global_references else (None, 0)
        output: dict[str, float | int | None] = {"global_rank": global_rank, "global_support": global_support}
        recent = self.recent_observations.get(key, ())
        for window_s, label in ((86400, "24h"), (21600, "6h")):
            values = sorted(value for at, value in recent if timestamp - window_s < at < timestamp)
            rank, support = empirical_midrank(float(payoff), values) if values else (None, 0)
            output[f"rank_{label}"] = rank
            output[f"support_{label}"] = support
        return output

    def compact(self, *, cutoff: float, maximum_window_s: int = max(RATING_V3_RECENT_WINDOWS_S)) -> None:
        threshold = _timestamp(cutoff) - maximum_window_s
        self.recent_observations = {key: [(at, payoff) for at, payoff in values if at > threshold] for key, values in self.recent_observations.items() if any(at > threshold for at, _payoff in values)}

    def as_dict(self) -> dict[str, object]:
        return {"global_references": self.global_references, "recent_observations": {key: [[at, payoff] for at, payoff in values] for key, values in self.recent_observations.items()}}


def blend_rank_percentile(structural_percentile: float, ranks: Mapping[str, object], blend: RankBlend) -> dict[str, float]:
    percentile = float(structural_percentile)
    global_rank = ranks.get("global_rank")
    global_support = int(ranks.get("global_support") or 0)
    if global_rank is not None:
        weight = shrinkage_weight(global_support, blend.global_alpha)
        percentile = (1.0 - weight) * percentile + weight * float(global_rank)
    after_global = percentile
    for label, alpha in (("24h", blend.local_24h_alpha), ("6h", blend.local_6h_alpha)):
        rank = ranks.get(f"rank_{label}")
        support = int(ranks.get(f"support_{label}") or 0)
        if alpha is not None and rank is not None:
            weight = shrinkage_weight(support, alpha)
            percentile = (1.0 - weight) * percentile + weight * float(rank)
    return {"structural_percentile": float(structural_percentile), "global_percentile": after_global, "blended_percentile": percentile, "rank_shift": after_global - float(structural_percentile), "recent_rank_shift": percentile - after_global}


def residual_features(*, base_delta: float, rank_values: Mapping[str, float], ranks: Mapping[str, object], target_rating: float, target_games: int, other_rating: float | None, context: Mapping[str, object]) -> dict[str, float]:
    pregame_scaled = (float(target_rating) - 2000.0) / 1000.0
    opponent_known = other_rating is not None or bool(context.get("opponent_rating_observed"))
    opponent_rating = float(other_rating) if other_rating is not None else _finite(context.get("opponent_rating"), 2000.0)
    median_known = context.get("family_rating_median") is not None
    median = _finite(context.get("family_rating_median"), 2000.0)
    traffic_300 = max(0.0, _finite(context.get("traffic_300")))
    traffic_1800 = max(0.0, _finite(context.get("traffic_1800")))
    fleet_300 = max(0.0, _finite(context.get("fleet_300")))
    fleet_1800 = max(0.0, _finite(context.get("fleet_1800")))
    return {
        "bias": 1.0,
        "base_delta": float(base_delta),
        "base_delta_abs": abs(float(base_delta)),
        "base_delta_signed_square": math.copysign(float(base_delta) ** 2, float(base_delta)) if base_delta else 0.0,
        "structural_percentile_centered": float(rank_values["structural_percentile"]) - 0.5,
        "rank_shift": float(rank_values["rank_shift"]),
        "recent_rank_shift": float(rank_values["recent_rank_shift"]),
        "log_global_support": math.log1p(int(ranks.get("global_support") or 0)),
        "log_24h_support": math.log1p(int(ranks.get("support_24h") or 0)),
        "log_6h_support": math.log1p(int(ranks.get("support_6h") or 0)),
        "pregame_rating_scaled": pregame_scaled,
        "pregame_rating_squared": pregame_scaled * pregame_scaled,
        "log_game_count": math.log1p(max(0, int(target_games))) / 10.0,
        "opponent_rating_known": float(opponent_known),
        "opponent_rating_scaled": (opponent_rating - 2000.0) / 1000.0 if opponent_known else 0.0,
        "rating_gap_scaled": (float(target_rating) - opponent_rating) / 1000.0 if opponent_known else 0.0,
        "family_median_known": float(median_known),
        "family_median_scaled": (median - 2000.0) / 1000.0 if median_known else 0.0,
        "traffic_300_log": math.log1p(traffic_300) / 5.0,
        "traffic_1800_log": math.log1p(traffic_1800) / 7.0,
        "fleet_300_log": math.log1p(fleet_300) / 4.0,
        "fleet_1800_log": math.log1p(fleet_1800) / 5.0,
        "identity_known": float(str(context.get("identity_scope") or "") == "known"),
    }


def sign_preserving_residual(base_delta: float, residual: float) -> float:
    """Apply a residual only when it preserves the rank model's nonzero direction."""
    candidate = float(base_delta) + float(residual)
    if base_delta > 0 and candidate <= 0:
        return float(base_delta)
    if base_delta < 0 and candidate >= 0:
        return float(base_delta)
    return candidate


@dataclass(frozen=True)
class RatingV3FamilyModel:
    """One family-specific structural, rank, residual, and calibration stack."""

    family: str
    structural_model: RidgeRatingModel
    rank_blend: RankBlend
    payoff_index: CausalPayoffIndex
    residual_model: RidgeRatingModel | None
    calibrator: SignedMagnitudeCalibrator
    intervals: dict[str, float]
    epoch_origin: float
    training_count: int
    cutoff: float
    validation: dict[str, object]

    def __post_init__(self) -> None:
        if self.family not in FAMILY_BASE_FEATURES:
            raise ValueError(f"unsupported rating v3 family: {self.family}")
        if self.training_count < 1:
            raise ValueError("rating v3 family model requires training data")

    def predict_terminal(self, terminal: Mapping[str, object], *, target_player: str, target_rating: float, target_games: int, terminal_at: float, other_rating: float | None = None, context: Mapping[str, object] | None = None) -> dict[str, object]:
        dynamic_context = dict(context or {})
        if _timestamp(terminal_at) <= self.cutoff:
            raise ValueError("rating v3 prospective prediction must be later than the model cutoff")
        view = _terminal_view(terminal, target_player)
        if str(view["family"]) != self.family:
            raise ValueError(f"rating v3 family mismatch: expected {self.family}, got {view['family']}")
        features = {name: float(view["base_features"][name]) for name in FAMILY_BASE_FEATURES[self.family] if name in view["base_features"]}
        features["completion_epoch_days"] = (_timestamp(terminal_at) - self.epoch_origin) / 86400.0
        resolved_other = other_rating
        if resolved_other is None and dynamic_context.get("opponent_rating_observed"):
            resolved_other = _finite(dynamic_context.get("opponent_rating"), 2000.0)
        features["opponent_pregame_rating_known"] = float(resolved_other is not None)
        features["opponent_pregame_rating_scaled"] = (float(resolved_other) - 2000.0) / 1000.0 if resolved_other is not None else 0.0
        structural_percentile = self.structural_model.predict(features)
        configuration = str(view["observed_configuration_sha256"])
        ranks = self.payoff_index.ranks(configuration, float(view["own_payoff"]), _timestamp(terminal_at))
        rank_values = blend_rank_percentile(structural_percentile, ranks, self.rank_blend)
        eta = eta_for_game_count(RATING_V3_ETA_SCHEDULE, int(target_games))
        base_delta = predict_display_delta(float(target_rating), int(target_games), rank_values["blended_percentile"], eta)
        residual_input = residual_features(base_delta=base_delta, rank_values=rank_values, ranks=ranks, target_rating=target_rating, target_games=target_games, other_rating=resolved_other, context=dynamic_context)
        residual = self.residual_model.predict(residual_input) if self.residual_model is not None else 0.0
        uncalibrated = sign_preserving_residual(base_delta, residual)
        predicted = self.calibrator.apply(uncalibrated)
        return {
            "contract": RATING_V3_CONTRACT,
            "model_version": RATING_V3_MODEL_VERSION,
            "status": "available",
            "family": self.family,
            "target_player": target_player,
            "target_rating": float(target_rating),
            "target_games": int(target_games),
            "other_rating_known": resolved_other is not None,
            "configuration_sha256": configuration,
            "own_payoff": float(view["own_payoff"]),
            "opponent_payoff": float(view["opponent_payoff"]),
            **rank_values,
            **ranks,
            "base_delta": base_delta,
            "residual_correction": residual,
            "uncalibrated_delta": uncalibrated,
            "calibration_scale": self.calibrator.positive_scale if uncalibrated > 0 else self.calibrator.negative_scale if uncalibrated < 0 else 1.0,
            "predicted_delta": predicted,
            "interval_80": [predicted + self.intervals.get("lower_80", 0.0), predicted + self.intervals.get("upper_80", 0.0)],
            "interval_95": [predicted + self.intervals.get("lower_95", 0.0), predicted + self.intervals.get("upper_95", 0.0)],
            "eta": eta,
            "boundary": "Shadow estimate only; deterministic legality, timeout, and catastrophic-loss guards remain authoritative.",
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "structural_model": self.structural_model.as_dict(),
            "rank_blend": self.rank_blend.as_dict(),
            "payoff_index": self.payoff_index.as_dict(),
            "residual_model": self.residual_model.as_dict() if self.residual_model is not None else None,
            "calibrator": self.calibrator.as_dict(),
            "intervals": copy.deepcopy(self.intervals),
            "epoch_origin": self.epoch_origin,
            "training_count": self.training_count,
            "cutoff": self.cutoff,
            "validation": copy.deepcopy(self.validation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RatingV3FamilyModel:
        residual = value.get("residual_model")
        payoff = value.get("payoff_index")
        if not isinstance(payoff, Mapping):
            raise ValueError("rating v3 model lacks its payoff index")
        return cls(
            family=str(value["family"]),
            structural_model=RidgeRatingModel.from_dict(value["structural_model"]),
            rank_blend=RankBlend.from_dict(value["rank_blend"]),
            payoff_index=CausalPayoffIndex(global_references=payoff.get("global_references"), recent_observations=payoff.get("recent_observations")),
            residual_model=RidgeRatingModel.from_dict(residual) if isinstance(residual, Mapping) else None,
            calibrator=SignedMagnitudeCalibrator.from_dict(value["calibrator"]),
            intervals={str(key): float(number) for key, number in dict(value["intervals"]).items()},
            epoch_origin=float(value["epoch_origin"]),
            training_count=int(value["training_count"]),
            cutoff=float(value["cutoff"]),
            validation=copy.deepcopy(dict(value.get("validation") or {})),
        )


@dataclass(frozen=True)
class GleeRatingV3Model:
    """Portable all-family v3 artifact with an explicit shadow-only boundary."""

    families: dict[str, RatingV3FamilyModel]
    source: dict[str, object]

    def predict_terminal(self, terminal: Mapping[str, object], **kwargs: object) -> dict[str, object]:
        family = str(terminal.get("game_family") or "")
        model = self.families.get(family)
        if model is None:
            raise ValueError(f"rating v3 artifact has no family model: {family}")
        return model.predict_terminal(terminal, **kwargs)

    def as_dict(self) -> dict[str, object]:
        return {"contract": RATING_V3_CONTRACT, "schema_version": RATING_V3_SCHEMA_VERSION, "model_version": RATING_V3_MODEL_VERSION, "families": {family: model.as_dict() for family, model in sorted(self.families.items())}, "source": copy.deepcopy(self.source), "boundary": "Offline and prospective shadow only; no action, prompt, matchmaking, timing, identity, or public-reporting authority."}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GleeRatingV3Model:
        if value.get("contract") != RATING_V3_CONTRACT or value.get("schema_version") != RATING_V3_SCHEMA_VERSION or value.get("model_version") != RATING_V3_MODEL_VERSION:
            raise ValueError("unsupported GLEE rating v3 artifact")
        families = value.get("families")
        if not isinstance(families, Mapping):
            raise ValueError("rating v3 artifact lacks family models")
        return cls(families={str(family): RatingV3FamilyModel.from_dict(model) for family, model in families.items() if isinstance(model, Mapping)}, source=copy.deepcopy(dict(value.get("source") or {})))


def load_rating_v3_release(root: Path) -> tuple[GleeRatingV3Model, dict[str, object]]:
    """Resolve and verify the canonical release pointer without activating live authority."""
    release_root = root.resolve()
    pointer = json.loads((release_root / "current.json").read_text(encoding="utf-8"))
    if pointer.get("contract") != "glee-rating-model-release-pointer-v1":
        raise ValueError("unsupported rating-model release pointer")
    manifest_path = release_root / str(pointer["release_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "glee-rating-model-release-v1" or manifest.get("release") != pointer.get("release"):
        raise ValueError("rating-model release manifest differs from its pointer")
    release_dir = manifest_path.parent
    for name, receipt in dict(manifest.get("artifacts") or {}).items():
        path = release_dir / str(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != int(receipt["bytes"]) or digest != receipt["sha256"]:
            raise RuntimeError(f"rating-model release artifact mismatch: {name}")
    model_path = release_dir / "rating-model.json"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest.get("model_sha256"):
        raise RuntimeError("rating-model release model hash mismatch")
    model = GleeRatingV3Model.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
    return model, {"release": str(manifest["release"]), "model_sha256": str(manifest["model_sha256"]), "manifest_path": str(manifest_path), "status": str(manifest["status"])}
