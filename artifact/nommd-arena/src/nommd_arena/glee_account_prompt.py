"""Compile and serve a narrow account-level prompt context for hidden Bargaining opponents."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .glee_account_identity_analysis import UNKNOWN_ACCOUNT, load_account_map
from .glee_activity_eda import GLEE_FAMILIES, _file_digest
from .glee_behavior_channel_analysis import CHANNELS, UNKNOWN_ID, FingerprintProfile, channel_features, chronological_game_splits, fit_fingerprint_profile
from .glee_behavior_corpus import extract_behavior_moves
from .glee_behavior_fusion_analysis import ConditionalStacker, FusionExample, fit_conditional_stacker
from .glee_causal_identity_routing import _first_move_record


ACCOUNT_PROMPT_CONTRACT = "glee-account-prompt-context-v1"
MODEL_VERSION = "account-first-move-v1"
MINIMUM_GALLERY_GAMES = 3
MINIMUM_VALIDATION_PREDICTIONS = 3
MINIMUM_VALIDATION_PRECISION = 0.8
SUPPORTED_LIVE_FAMILY = "bargaining"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _gzip_json(value: object) -> bytes:
    return gzip.compress((_canonical(value) + "\n").encode("utf-8"), compresslevel=9, mtime=0)


def _gzip_jsonl(values: Iterable[Mapping[str, object]]) -> bytes:
    return gzip.compress("".join(_canonical(value) + "\n" for value in values).encode("utf-8"), compresslevel=9, mtime=0)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _serialize_profile(profile: FingerprintProfile) -> dict[str, object]:
    return {
        "alpha": profile.alpha,
        "candidates": list(profile.candidates),
        "global_probability": dict(sorted(profile.global_probability.items())),
        "identity_counts": {identity: dict(sorted(counts.items())) for identity, counts in sorted(profile.identity_counts.items())},
        "identity_totals": dict(sorted(profile.identity_totals.items())),
        "vocabulary": sorted(profile.vocabulary),
    }


def _deserialize_profile(value: Mapping[str, object]) -> FingerprintProfile:
    return FingerprintProfile(
        candidates=tuple(str(item) for item in value["candidates"]),
        vocabulary=frozenset(str(item) for item in value["vocabulary"]),
        global_probability={str(token): float(probability) for token, probability in dict(value["global_probability"]).items()},
        identity_counts={str(identity): {str(token): float(count) for token, count in dict(counts).items()} for identity, counts in dict(value["identity_counts"]).items()},
        identity_totals={str(identity): float(total) for identity, total in dict(value["identity_totals"]).items()},
        alpha=float(value["alpha"]),
    )


def _serialize_stacker(stacker: ConditionalStacker) -> dict[str, object]:
    return {
        "feature_names": list(stacker.feature_names),
        "scales": dict(sorted(stacker.scales.items())),
        "unknown_bias": stacker.unknown_bias,
        "weights": dict(sorted(stacker.weights.items())),
    }


def _deserialize_stacker(value: Mapping[str, object]) -> ConditionalStacker:
    return ConditionalStacker(
        feature_names=tuple(str(item) for item in value["feature_names"]),
        scales={str(name): float(scale) for name, scale in dict(value["scales"]).items()},
        weights={str(name): float(weight) for name, weight in dict(value["weights"]).items()},
        unknown_bias=float(value["unknown_bias"]),
    )


def _opponent_role(record: Mapping[str, object]) -> str:
    for move in record.get("moves", []):
        context = move.get("context") if isinstance(move, Mapping) and isinstance(move.get("context"), Mapping) else {}
        if context.get("opponent_role"):
            return str(context["opponent_role"]).casefold()
    return "none"


def _account_records(records: Sequence[Mapping[str, object]], account_for_id: Mapping[str, str]) -> list[dict[str, object]]:
    return [dict(record) | {"public_player_id": account_for_id.get(str(record["public_player_id"]), UNKNOWN_ACCOUNT)} for record in records]


def _galleries(records: Sequence[Mapping[str, object]], assignments: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for family in GLEE_FAMILIES:
        support = Counter(
            str(record["public_player_id"])
            for record in records
            if record["family"] == family and assignments.get(str(record["game_id"])) == "train" and record["public_player_id"] != UNKNOWN_ACCOUNT
        )
        output[family] = tuple(sorted(account for account, count in support.items() if count >= MINIMUM_GALLERY_GAMES))
    return output


def _vectors(records: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], Counter[str]]:
    output: dict[tuple[str, str], Counter[str]] = {}
    for record in records:
        snapshot = _first_move_record(record)
        for channel in CHANNELS:
            vector = channel_features(snapshot, channel)
            if channel == "action":
                vector = Counter({token: count for token, count in vector.items() if not token.startswith("terminal-style|")})
            output[(str(record["game_id"]), channel)] = vector
    return output


def _profiles(
    records: Sequence[Mapping[str, object]],
    assignments: Mapping[str, str],
    galleries: Mapping[str, Sequence[str]],
    vectors: Mapping[tuple[str, str], Counter[str]],
    channel_summary: Mapping[str, object],
) -> tuple[dict[tuple[str, str, str], FingerprintProfile], dict[tuple[str, str], float]]:
    profiles: dict[tuple[str, str, str], FingerprintProfile] = {}
    temperatures: dict[tuple[str, str], float] = {}
    for family in GLEE_FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        train = [record for record in family_records if assignments.get(str(record["game_id"])) == "train"]
        calibration = [record for record in family_records if assignments.get(str(record["game_id"])) == "calibration"]
        for channel in CHANNELS:
            selected = channel_summary["families"][family][channel].get("selected_hyperparameters")
            if not selected:
                continue
            local_vectors = {str(record["game_id"]): vectors[(str(record["game_id"]), channel)] for record in family_records}
            temperatures[(family, channel)] = float(selected["temperature"])
            profiles[(family, channel, "calibration")] = fit_fingerprint_profile(train, local_vectors, candidates=galleries[family], alpha=float(selected["alpha"]), channel=channel)
            profiles[(family, channel, "test")] = fit_fingerprint_profile(train + calibration, local_vectors, candidates=galleries[family], alpha=float(selected["alpha"]), channel=channel)
    return profiles, temperatures


def _examples(
    original_records: Sequence[Mapping[str, object]],
    account_records: Sequence[Mapping[str, object]],
    assignments: Mapping[str, str],
    galleries: Mapping[str, tuple[str, ...]],
    profiles: Mapping[tuple[str, str, str], FingerprintProfile],
    vectors: Mapping[tuple[str, str], Counter[str]],
    temperatures: Mapping[tuple[str, str], float],
) -> list[FusionExample]:
    original_by_game = {str(record["game_id"]): record for record in original_records}
    output: list[FusionExample] = []
    for record in account_records:
        game_id = str(record["game_id"])
        split = assignments.get(game_id)
        if split not in {"calibration", "test"}:
            continue
        family = str(record["family"])
        gallery = galleries[family]
        candidates = (*gallery, UNKNOWN_ID)
        features = {candidate: {} for candidate in candidates}
        evidence: dict[str, int] = {}
        for channel in CHANNELS:
            profile = profiles.get((family, channel, split))
            if profile is None:
                scores, count = ({candidate: 0.0 for candidate in gallery}, 0)
                temperature = 1.0
            else:
                scores, count = profile.scores(vectors[(game_id, channel)])
                temperature = temperatures[(family, channel)]
            evidence[channel] = count
            for candidate in candidates:
                features[candidate][channel] = float(scores.get(candidate, 0.0)) / temperature
        account = str(record["public_player_id"])
        target = account if account in gallery else UNKNOWN_ID
        original = original_by_game[game_id]
        output.append(
            FusionExample(
                game_id=game_id,
                family=family,
                role=_opponent_role(original),
                split=split,
                started_at=str(record["started_at"]),
                candidates=candidates,
                target=target,
                true_public_player_id=str(original["public_player_id"]),
                features=features,
                raw_probabilities={},
                evidence=evidence,
            )
        )
    return sorted(output, key=lambda example: (example.started_at, example.game_id))


def _prediction(example: FusionExample, stacker: ConditionalStacker) -> dict[str, object]:
    probabilities = stacker.probabilities(example)
    ranking = sorted(probabilities, key=lambda label: (-probabilities[label], label))
    target = UNKNOWN_ACCOUNT if example.target == UNKNOWN_ID else example.target
    translated = {UNKNOWN_ACCOUNT if label == UNKNOWN_ID else label: probability for label, probability in probabilities.items()}
    translated_ranking = [UNKNOWN_ACCOUNT if label == UNKNOWN_ID else label for label in ranking]
    predicted = translated_ranking[0]
    return {
        "game_id": example.game_id,
        "family": example.family,
        "role": example.role,
        "started_at": example.started_at,
        "true_public_player_id": example.true_public_player_id,
        "true_account": target,
        "predicted_account": predicted,
        "correct": predicted == target,
        "confidence": translated[predicted],
        "unknown_probability": translated[UNKNOWN_ACCOUNT],
        "true_rank": translated_ranking.index(target) + 1,
        "evidence_by_channel": dict(example.evidence),
        "account_probabilities": dict(sorted(translated.items())),
    }


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"games": 0}
    named = [row for row in rows if row["predicted_account"] != UNKNOWN_ACCOUNT]
    return {
        "games": len(rows),
        "top_one_accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "top_3_accuracy": sum(str(row["true_account"]) in sorted(dict(row["account_probabilities"]), key=lambda label: (-float(dict(row["account_probabilities"])[label]), label))[:3] for row in rows) / len(rows),
        "mean_reciprocal_rank": sum(1.0 / int(row["true_rank"]) for row in rows) / len(rows),
        "named_predictions": len(named),
        "named_prediction_coverage": len(named) / len(rows),
        "named_prediction_precision": sum(bool(row["correct"]) for row in named) / len(named) if named else None,
    }


def _named_validation(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    values: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        predicted = str(row["predicted_account"])
        if predicted == UNKNOWN_ACCOUNT:
            continue
        values[predicted][0] += 1
        values[predicted][1] += int(bool(row["correct"]))
    return {
        account: {
            "correct": counts[1],
            "named_predictions": counts[0],
            "precision": counts[1] / counts[0],
            "validated_for_prompt": counts[0] >= MINIMUM_VALIDATION_PREDICTIONS and counts[1] / counts[0] >= MINIMUM_VALIDATION_PRECISION,
        }
        for account, counts in sorted(values.items())
    }


def _account_metadata(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, dict[str, object]] = {}
    for group in payload["groups"]:
        output[str(group["key"])] = {
            "linkage_confidence": str(group.get("confidence") or "unknown"),
            "member_labels": sorted(str(label) for label in group["members"]),
            "member_public_ids": sorted(str(public_id) for public_id in group["members"].values()),
        }
    return output


class OpponentAccountPromptModelCompiler:
    """Freeze a direct account classifier and its untouched validation gate."""

    def __init__(self, *, behavior_dir: Path, channel_dir: Path, account_groups: Path, output_root: Path) -> None:
        self.behavior_dir = behavior_dir.resolve()
        self.channel_dir = channel_dir.resolve()
        self.account_groups = account_groups.resolve()
        self.output_root = output_root.resolve()

    def run(self) -> dict[str, object]:
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise FileExistsError(f"account prompt-model output root is not empty: {self.output_root}")
        behavior_path = self.behavior_dir / "behavior-games.jsonl"
        channel_path = self.channel_dir / "summary.json"
        records = _load_jsonl(behavior_path)
        channel_summary = json.loads(channel_path.read_text(encoding="utf-8"))
        account_for_id, confidence_for_account, members_by_account = load_account_map(self.account_groups)
        account_metadata = _account_metadata(self.account_groups)
        train_fraction = float(channel_summary["design"]["train_fraction"])
        calibration_fraction = float(channel_summary["design"]["calibration_fraction"])
        assignments, boundaries = chronological_game_splits(records, train_fraction=train_fraction, calibration_fraction=calibration_fraction)
        relabeled = _account_records(records, account_for_id)
        galleries = _galleries(relabeled, assignments)
        vectors = _vectors(records)
        profiles, temperatures = _profiles(relabeled, assignments, galleries, vectors, channel_summary)
        examples = _examples(records, relabeled, assignments, galleries, profiles, vectors, temperatures)
        calibration = [example for example in examples if example.split == "calibration"]
        test = [example for example in examples if example.split == "test"]
        stacker = fit_conditional_stacker(calibration, feature_names=CHANNELS)
        test_rows = [_prediction(example, stacker) for example in test]
        validation_by_family: dict[str, dict[str, dict[str, object]]] = {}
        for family in GLEE_FAMILIES:
            validation_by_family[family] = _named_validation([row for row in test_rows if row["family"] == family])
        validated_accounts = {
            family: sorted(account for account, validation in validation_by_family[family].items() if validation["validated_for_prompt"])
            for family in GLEE_FAMILIES
        }
        model = {
            "schema_version": 1,
            "contract": ACCOUNT_PROMPT_CONTRACT,
            "model_version": MODEL_VERSION,
            "unknown_label": UNKNOWN_ACCOUNT,
            "supported_live_family": SUPPORTED_LIVE_FAMILY,
            "feature_scope": "exactly the first visible opponent move; no live-presence input",
            "stacker": _serialize_stacker(stacker),
            "families": {
                family: {
                    "candidates": list(galleries[family]),
                    "temperatures": {channel: temperatures[(family, channel)] for channel in CHANNELS if (family, channel) in temperatures},
                    "profiles": {channel: _serialize_profile(profiles[(family, channel, "test")]) for channel in CHANNELS if (family, channel, "test") in profiles},
                    "validated_accounts": {
                        account: validation_by_family[family][account] | account_metadata.get(account, {"linkage_confidence": confidence_for_account.get(account, "unknown"), "member_labels": [], "member_public_ids": list(members_by_account.get(account, ()))})
                        for account in validated_accounts[family]
                    },
                }
                for family in GLEE_FAMILIES
            },
        }
        evaluation = {
            "schema_version": 1,
            "contract": ACCOUNT_PROMPT_CONTRACT,
            "model_version": MODEL_VERSION,
            "design": {
                "account_profiles": "direct pooled-sibling profiles",
                "calibration": "chronological middle block",
                "gate_selection": "post-hoc account whitelist from untouched final block",
                "minimum_gallery_games": MINIMUM_GALLERY_GAMES,
                "minimum_validation_predictions": MINIMUM_VALIDATION_PREDICTIONS,
                "minimum_validation_precision": MINIMUM_VALIDATION_PRECISION,
                "online_authority": "advisory-only prompt context",
            },
            "boundaries": boundaries,
            "candidate_accounts": {family: len(galleries[family]) for family in GLEE_FAMILIES},
            "calibration_games": len(calibration),
            "test": {
                "pooled": _metrics(test_rows),
                "by_family": {family: _metrics([row for row in test_rows if row["family"] == family]) for family in GLEE_FAMILIES},
                "named_validation_by_family": validation_by_family,
                "validated_accounts_by_family": validated_accounts,
            },
            "stacker": _serialize_stacker(stacker),
        }
        release_name = "v1-frontier-18108-account-map-36499"
        release = self.output_root / "releases" / release_name
        model_bytes = _gzip_json(model)
        predictions_bytes = _gzip_jsonl(test_rows)
        _atomic_bytes(release / "model.json.gz", model_bytes)
        _atomic_bytes(release / "test-predictions.jsonl.gz", predictions_bytes)
        _atomic_json(release / "evaluation.json", evaluation)
        artifacts = {
            "model.json.gz": {"bytes": len(model_bytes), "sha256": _digest_bytes(model_bytes)},
            "test-predictions.jsonl.gz": {"bytes": len(predictions_bytes), "sha256": _digest_bytes(predictions_bytes)},
            "evaluation.json": {"bytes": (release / "evaluation.json").stat().st_size, "sha256": _file_digest(release / "evaluation.json")},
        }
        manifest = {
            "schema_version": 1,
            "contract": ACCOUNT_PROMPT_CONTRACT,
            "model_version": MODEL_VERSION,
            "release": release_name,
            "sources": {
                "behavior_games": {"path": os.path.relpath(behavior_path, self.output_root.parent), "sha256": _file_digest(behavior_path)},
                "channel_summary": {"path": os.path.relpath(channel_path, self.output_root.parent), "sha256": _file_digest(channel_path)},
                "account_groups": {"path": os.path.relpath(self.account_groups, self.output_root.parent), "sha256": _file_digest(self.account_groups)},
            },
            "artifacts": artifacts,
            "activation": "requires a controlled supervisor restart; no live process is mutated by compilation",
        }
        _atomic_json(release / "manifest.json", manifest)
        current = {
            "schema_version": 1,
            "contract": ACCOUNT_PROMPT_CONTRACT,
            "model_version": MODEL_VERSION,
            "release": f"releases/{release_name}",
            "manifest_sha256": _file_digest(release / "manifest.json"),
        }
        _atomic_json(self.output_root / "current.json", current)
        return {"output_root": str(self.output_root), "release": release_name, "manifest_sha256": current["manifest_sha256"], "evaluation": evaluation}


@dataclass(frozen=True)
class AccountPromptAssessment:
    """One causal first-move account assessment and optional model-facing context."""

    receipt: dict[str, object]
    prompt_context: dict[str, object] | None


class OpponentAccountPromptModelReader:
    """Read one immutable account model and emit narrow HI Bargaining context."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        current_path = self.root / "current.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        if current.get("contract") != ACCOUNT_PROMPT_CONTRACT:
            raise ValueError("account prompt-model pointer has the wrong contract")
        release = (self.root / str(current["release"])).resolve()
        if self.root not in release.parents:
            raise ValueError("account prompt-model release escapes its root")
        manifest_path = release / "manifest.json"
        if _file_digest(manifest_path) != current["manifest_sha256"]:
            raise ValueError("account prompt-model manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = release / "model.json.gz"
        if _file_digest(model_path) != manifest["artifacts"]["model.json.gz"]["sha256"]:
            raise ValueError("account prompt-model artifact hash mismatch")
        model = json.loads(gzip.decompress(model_path.read_bytes()))
        if model.get("contract") != ACCOUNT_PROMPT_CONTRACT:
            raise ValueError("account prompt-model artifact has the wrong contract")
        self.current = current
        self.manifest = manifest
        self.model = model
        self.stacker = _deserialize_stacker(model["stacker"])
        self.profiles = {
            (family, channel): _deserialize_profile(profile)
            for family, family_model in model["families"].items()
            for channel, profile in family_model["profiles"].items()
        }

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "contract": ACCOUNT_PROMPT_CONTRACT,
            "model_version": self.model["model_version"],
            "release": self.current["release"],
            "manifest_sha256": self.current["manifest_sha256"],
            "supported_live_family": self.model["supported_live_family"],
        }

    def assess(self, game: Mapping[str, object]) -> AccountPromptAssessment:
        family = str(game.get("game_family") or "")
        opponent = game.get("opponent") if isinstance(game.get("opponent"), Mapping) else {}
        base = {"contract": ACCOUNT_PROMPT_CONTRACT, "model_version": self.model["model_version"], "family": family, "game_id": game.get("game_id"), "authority": "advisory-only"}
        if family != self.model["supported_live_family"]:
            return AccountPromptAssessment(base | {"status": "unsupported-family"}, None)
        if opponent.get("type") != "hidden":
            return AccountPromptAssessment(base | {"status": "known-identity-not-routed"}, None)
        moves = extract_behavior_moves(game)
        if not moves:
            return AccountPromptAssessment(base | {"status": "awaiting-first-opponent-move"}, None)
        family_model = self.model["families"][family]
        candidates = tuple(str(candidate) for candidate in family_model["candidates"])
        labels = (*candidates, UNKNOWN_ID)
        snapshot = {"moves": moves[:1]}
        features = {label: {} for label in labels}
        evidence: dict[str, int] = {}
        for channel in CHANNELS:
            vector = channel_features(snapshot, channel)
            if channel == "action":
                vector = Counter({token: count for token, count in vector.items() if not token.startswith("terminal-style|")})
            profile = self.profiles[(family, channel)]
            scores, count = profile.scores(vector)
            evidence[channel] = count
            temperature = float(family_model["temperatures"][channel])
            for label in labels:
                features[label][channel] = float(scores.get(label, 0.0)) / temperature
        example = FusionExample(str(game.get("game_id") or "live"), family, "none", "live", "", labels, UNKNOWN_ID, UNKNOWN_ID, features, {}, evidence)
        probabilities = self.stacker.probabilities(example)
        ranking = sorted(probabilities, key=lambda label: (-probabilities[label], label))
        predicted = ranking[0]
        translated = {UNKNOWN_ACCOUNT if label == UNKNOWN_ID else label: probability for label, probability in probabilities.items()}
        predicted_account = UNKNOWN_ACCOUNT if predicted == UNKNOWN_ID else predicted
        receipt = base | {
            "status": "abstained-unknown" if predicted == UNKNOWN_ID else "abstained-unvalidated-account",
            "predicted_account": predicted_account,
            "posterior_probability": translated[predicted_account],
            "unknown_probability": translated[UNKNOWN_ACCOUNT],
            "evidence_by_channel": evidence,
        }
        validated = family_model["validated_accounts"]
        if predicted == UNKNOWN_ID or predicted not in validated:
            return AccountPromptAssessment(receipt, None)
        metadata = validated[predicted]
        runner_up = ranking[1]
        runner_up_account = UNKNOWN_ACCOUNT if runner_up == UNKNOWN_ID else runner_up
        context = {
            "contract": ACCOUNT_PROMPT_CONTRACT,
            "status": "validated-candidate",
            "candidate_account": predicted,
            "candidate_member_labels": metadata["member_labels"],
            "posterior_probability": round(float(translated[predicted]), 6),
            "unknown_probability": round(float(translated[UNKNOWN_ACCOUNT]), 6),
            "runner_up": {"account": runner_up_account, "probability": round(float(translated[runner_up_account]), 6)},
            "linkage_confidence": metadata["linkage_confidence"],
            "masked_ki_validation": {"named_predictions": metadata["named_predictions"], "correct": metadata["correct"], "precision": metadata["precision"]},
            "evidence_by_channel": evidence,
            "evidence_scope": "first visible opponent move only; pooled sibling behavior; no live-presence input",
            "authority": "advisory-only",
        }
        return AccountPromptAssessment(receipt | {"status": "admitted"}, context)
