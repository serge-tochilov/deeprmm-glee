"""Live candidate substitution for the frozen Persuasion buyer-continuation heads."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .corpus import _event, canonical_json, classify_persuasion_signal, extract_events, file_sha256, object_sha256
from .data import CorpusVocabs, SequenceCollator, Vocabulary, move_batch
from .live_shadow import _static_live_game
from .model import BoundedMessageFusion, ModelConfig, _compact_message_sequence
from .persuasion_buyer_continuation import BUYER_ACTIONS, BUYER_CONTINUATION_EXPERIMENT_CONTRACT, BUYER_CONTINUATION_LABELS, BUYER_CONTINUATION_RELEASE_CONTRACT


BUYER_CONTINUATION_LIVE_CONTRACT = "glee-persuasion-buyer-continuation-live-v1"
BUYER_CONTINUATION_REGISTRY_CONTRACT = "glee-persuasion-buyer-continuation-prospective-registry-v1"
BUYER_CONTINUATION_AUTHORITY = "advisory-next-seller-signal-evidence-only"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player: {player!r}")


class _LiveReversedPersuasionHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.action_fusion = BoundedMessageFusion(config.hidden_dim, config.message_hidden_dim, config.message_gate_max)
        self.action = nn.Linear(config.hidden_dim, len(BUYER_CONTINUATION_LABELS))

    def forward(self, mechanics: torch.Tensor, message: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        hidden, _gate = self.action_fusion(mechanics, message, available)
        return self.action(hidden)


class BuyerContinuationProspectiveRegistry:
    """Append one immutable 2-candidate forecast before its next-round seller signal exists."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions (
                release_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                target_round INTEGER NOT NULL,
                source_turn_id TEXT NOT NULL,
                candidate_set_sha256 TEXT NOT NULL,
                prefix_sha256 TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                prediction_sha256 TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                PRIMARY KEY (release_id, game_id, target_round)
            );
            """
        )
        self.connection.execute("INSERT OR IGNORE INTO registry_metadata(key, value) VALUES (?, ?)", ("contract", BUYER_CONTINUATION_REGISTRY_CONTRACT))
        contract = self.connection.execute("SELECT value FROM registry_metadata WHERE key = ?", ("contract",)).fetchone()
        if contract != (BUYER_CONTINUATION_REGISTRY_CONTRACT,):
            raise ValueError("buyer-continuation registry contract mismatch")

    def close(self) -> None:
        self.connection.close()

    def register(self, prediction: Mapping[str, object]) -> dict[str, object]:
        if prediction.get("contract") != BUYER_CONTINUATION_LIVE_CONTRACT or prediction.get("authority") != BUYER_CONTINUATION_AUTHORITY:
            raise ValueError("buyer-continuation prediction has the wrong contract or authority")
        if any(str(key).startswith("actual_") or str(key).startswith("outcome") for key in prediction):
            raise ValueError("buyer-continuation prediction contains a post-decision outcome")
        required = ("release_id", "game_id", "target_round", "source_turn_id", "candidate_set_sha256", "prefix_sha256")
        if any(key not in prediction for key in required):
            raise ValueError("buyer-continuation prediction lacks an immutable identity")
        key = (str(prediction["release_id"]), str(prediction["game_id"]), int(prediction["target_round"]))
        payload = canonical_json(dict(prediction))
        prediction_sha256 = object_sha256(dict(prediction))
        identity = (str(prediction["source_turn_id"]), str(prediction["candidate_set_sha256"]), str(prediction["prefix_sha256"]), prediction_sha256)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT source_turn_id, candidate_set_sha256, prefix_sha256, prediction_sha256 FROM predictions WHERE release_id = ? AND game_id = ? AND target_round = ?", key).fetchone()
            if existing is not None:
                if existing != identity:
                    raise ValueError("buyer-continuation prediction key already has different immutable evidence")
                self.connection.execute("COMMIT")
                return {"status": "already-registered", "prediction_sha256": prediction_sha256, "key": key}
            self.connection.execute(
                "INSERT INTO predictions(release_id, game_id, target_round, source_turn_id, candidate_set_sha256, prefix_sha256, registered_at, prediction_sha256, prediction_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*key, *identity[:3], _utc_now(), prediction_sha256, payload),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return {"status": "registered", "prediction_sha256": prediction_sha256, "key": key}

    def summary(self) -> dict[str, object]:
        count = int(self.connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        return {"contract": BUYER_CONTINUATION_REGISTRY_CONTRACT, "predictions": count, "path": str(self.path)}


class PersuasionBuyerContinuationRelease:
    """Attach frozen reversed heads to the already loaded population sequence backbones."""

    def __init__(self, release_dir: Path, *, sequence: Any, registry_path: Path | None = None) -> None:
        self.release_dir = release_dir.resolve()
        self.manifest_path = self.release_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("contract") != BUYER_CONTINUATION_RELEASE_CONTRACT or self.manifest.get("status") != "frozen-offline-candidate":
            raise ValueError("unsupported or unfrozen Persuasion buyer-continuation release")
        if tuple(self.manifest.get("target_labels") or ()) != BUYER_CONTINUATION_LABELS:
            raise ValueError("buyer-continuation target labels changed")
        if self.manifest.get("selected_arm") != "deeprmm-only" or self.manifest.get("train_sources") != ["deeprmm"]:
            raise ValueError("live promotion requires the validation-selected DeepRMM-only arm")
        if dict(_require_mapping(self.manifest.get("model"), name="buyer-continuation model")) != sequence.config.receipt():
            raise ValueError("buyer-continuation head and sequence backbone configurations differ")
        source_sequence = _require_mapping(self.manifest.get("source_sequence_release"), name="source sequence release")
        if source_sequence.get("candidate_id") != sequence.candidate_id or source_sequence.get("manifest_sha256") != file_sha256(sequence.manifest_path):
            raise ValueError("buyer-continuation head identifies a different sequence release")
        self.sequence = sequence
        self.device = sequence.device
        self.vocabs = replace(sequence.vocabs, target_labels={**sequence.vocabs.target_labels, "persuasion": Vocabulary(BUYER_CONTINUATION_LABELS)})
        source_components = {int(receipt["seed"]): receipt for receipt in sequence.manifest["components"]}
        raw_components = self.manifest.get("components")
        if not isinstance(raw_components, list) or len(raw_components) != len(sequence.models):
            raise ValueError("buyer-continuation release has incomplete component coverage")
        self.heads: list[tuple[int, _LiveReversedPersuasionHead]] = []
        for receipt in raw_components:
            component = _require_mapping(receipt, name="buyer-continuation component")
            seed = int(component["seed"])
            source = _require_mapping(source_components.get(seed), name=f"sequence component {seed}")
            source_path = sequence.release_dir / str(source["path"])
            head_path = self.release_dir / str(component["path"])
            if file_sha256(head_path) != component.get("sha256"):
                raise ValueError(f"buyer-continuation component hash mismatch: {head_path}")
            payload = torch.load(head_path, map_location=self.device, weights_only=False)
            if payload.get("contract") != BUYER_CONTINUATION_EXPERIMENT_CONTRACT or payload.get("component_seed") != seed or payload.get("source_component_sha256") != file_sha256(source_path) or tuple(payload.get("target_labels") or ()) != BUYER_CONTINUATION_LABELS:
                raise ValueError(f"buyer-continuation component source contract mismatch: {head_path}")
            head = _LiveReversedPersuasionHead(sequence.config).to(self.device)
            head.load_state_dict(payload["head"])
            head.eval()
            self.heads.append((seed, head))
        if [seed for seed, _model in sequence.models] != [seed for seed, _head in self.heads]:
            raise ValueError("buyer-continuation heads and sequence backbones are not seed-aligned")
        inventory = _require_mapping(_require_mapping(self.manifest.get("source_corpus"), name="source corpus").get("inventory"), name="source inventory")
        deep = _require_mapping(inventory.get("deeprmm"), name="DeepRMM buyer-continuation inventory")
        raw_transitions = _require_mapping(deep.get("transitions"), name="DeepRMM transition support")
        self.transition_support = {str(key): int(value) for key, value in raw_transitions.items()}
        self.registry = BuyerContinuationProspectiveRegistry(registry_path) if registry_path is not None else None

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    def close(self) -> None:
        if self.registry is not None:
            self.registry.close()

    def _sample(self, *, game: Mapping[str, Any], turn_id: str, buyer_action: str) -> tuple[dict[str, object], str, int, str]:
        if buyer_action not in BUYER_ACTIONS:
            raise ValueError(f"unsupported buyer action: {buyer_action!r}")
        static_game = _static_live_game(game)
        if static_game["family"] != "persuasion" or static_game["our_role"] != "buyer":
            raise ValueError("buyer-continuation inference requires DeepRMM to be the Persuasion buyer")
        state = _require_mapping(game.get("game_state"), name="Persuasion game state")
        action_type = str(_require_mapping(game.get("valid_actions"), name="valid actions").get("type") or game.get("phase") or "")
        round_number = state.get("round")
        total_rounds = state.get("total_rounds")
        if action_type != "buyer_decision" or isinstance(round_number, bool) or not isinstance(round_number, int) or isinstance(total_rounds, bool) or not isinstance(total_rounds, int) or not 1 <= round_number < total_rounds:
            raise ValueError("buyer-continuation inference requires a nonterminal Persuasion buyer decision")
        events = extract_events(game)
        completed_rounds = [int(event["round_number"]) for event in events if event.get("actor") == "self" and event.get("kind") == "response"]
        if completed_rounds and max(completed_rounds) >= round_number:
            raise ValueError("current Persuasion buyer decision is already represented in history")
        game_id = str(game.get("game_id") or "")
        channel = str(state.get("seller_message_type") or "text").casefold()
        message = state.get("seller_message")
        polarity, family_act, _fingerprint = classify_persuasion_signal(message, channel=channel)
        signal_label = f"signal_{polarity}"
        signal_value = 1.0 if polarity == "positive" else -1.0 if polarity == "negative" else 0.0
        events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="opponent", kind="signal", action_label=signal_label, action_value=signal_value, message=message, family_act=family_act))
        events.append(_event(game_id=game_id, event_index=len(events), round_number=round_number, state=state, actor="self", kind="response", action_label=buyer_action, action_value=1.0 if buyer_action == "buy" else 0.0, response_time_ms=None))
        target_round = round_number + 1
        sample_id = hashlib.sha256(f"buyer-continuation-live:{game_id}:{round_number}:{buyer_action}:{turn_id}".encode("utf-8")).hexdigest()
        target = {
            "sample_id": sample_id,
            "game_id": game_id,
            "source_type": "prospective-live",
            "target_event_index": len(events),
            "prefix_length": len(events),
            "target_kind": "signal",
            "target_label": BUYER_CONTINUATION_LABELS[0],
            "target_value": None,
            "target_value_present": False,
            "target_message_act": None,
            "target_message_present": False,
            "target_delay_log_ms": None,
            "target_delay_present": False,
            "chronological_split": "prospective",
            "identity_scope": static_game["identity_scope"],
            "account_key": None,
            "account_confidence": None,
            "account_fold": -1,
            "mask_last_prefix_response_time": True,
        }
        prefix_sha256 = object_sha256({"static_game": static_game, "events_before_candidate": events[:-1], "current_signal": signal_label})
        return {"game": static_game, "events": events, "target": target}, signal_label, target_round, prefix_sha256

    @torch.inference_mode()
    def predict(self, *, game: Mapping[str, Any], turn_id: str, candidate_set_sha256: str, candidates: Sequence[Mapping[str, object]], register: bool = True) -> dict[str, object]:
        if not turn_id.strip() or len(candidate_set_sha256) != 64 or len(candidates) != 2:
            raise ValueError("buyer-continuation inference requires one identified 2-candidate set")
        receipts: list[tuple[int, str, str]] = []
        samples: list[dict[str, object]] = []
        preceding_signals: list[str] = []
        target_rounds: list[int] = []
        prefix_hashes: list[str] = []
        for expected_index, row in enumerate(candidates):
            if not isinstance(row, Mapping) or row.get("candidate_index") != expected_index:
                raise ValueError("buyer-continuation candidates are not in immutable index order")
            action = _require_mapping(row.get("action"), name="buyer candidate action")
            expected_action = "buy" if action == {"decision": "yes"} else "pass" if action == {"decision": "no"} else ""
            action_sha256 = str(row.get("action_sha256") or "")
            if not expected_action or object_sha256(dict(action)) != action_sha256:
                raise ValueError("buyer-continuation candidate is not an exact buy/pass action")
            sample, preceding_signal, target_round, prefix_sha256 = self._sample(game=game, turn_id=turn_id, buyer_action=expected_action)
            receipts.append((expected_index, action_sha256, expected_action))
            samples.append(sample)
            preceding_signals.append(preceding_signal)
            target_rounds.append(target_round)
            prefix_hashes.append(prefix_sha256)
        if {action for _index, _digest, action in receipts} != set(BUYER_ACTIONS) or len(set(target_rounds)) != 1 or len(set(prefix_hashes)) != 1:
            raise ValueError("buyer-continuation candidates do not form one aligned buy/pass counterfactual")
        requested_rows = len(samples)
        padded = list(samples)
        if self.device.type == "cuda" and len(padded) < 8:
            padded.extend([copy.deepcopy(padded[-1]) for _index in range(8 - len(padded))])
        batch = move_batch(SequenceCollator(self.vocabs, "persuasion")(padded), self.device)
        tensor_batch = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        component_probabilities: list[torch.Tensor] = []
        for (model_seed, model), (head_seed, head) in zip(self.sequence.models, self.heads, strict=True):
            if model_seed != head_seed:
                raise RuntimeError("buyer-continuation component seed alignment changed")
            events = model.event_encoder(tensor_batch)
            static = model.static_encoder(tensor_batch)
            account, _effective = model.account_context("persuasion", tensor_batch["accounts"], force_population=True)
            context = model.context_mix(torch.cat((static, account, static * account), dim=-1))
            mechanics = model.cores["persuasion"](events + context.unsqueeze(1), tensor_batch["lengths"])
            if model.message_encoder is None or model.message_context_mix is None or model.message_cores is None:
                raise ValueError("buyer-continuation runtime requires the frozen separate message stream")
            message_context = model.message_context_mix(torch.cat((static, account, static * account), dim=-1))
            message_events = model.message_encoder(tensor_batch) + message_context.unsqueeze(1)
            message_present = tensor_batch["event_numeric"][..., 7] > 0.5
            message_sequence, message_lengths, message_available = _compact_message_sequence(message_events, message_present, tensor_batch["lengths"])
            message = model.message_cores["persuasion"](message_sequence, message_lengths)
            component_probabilities.append(F.softmax(head(mechanics, message, message_available), dim=-1)[:requested_rows].cpu())
        ensemble = torch.stack(component_probabilities).mean(dim=0)
        rows: list[dict[str, object]] = []
        for row_index, (candidate_index, action_sha256, buyer_action) in enumerate(receipts):
            probabilities = [float(value) for value in ensemble[row_index].tolist()]
            if any(not math.isfinite(value) or value < 0.0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError("buyer-continuation probability distribution is invalid")
            support = sum(value for key, value in self.transition_support.items() if key.startswith(f"{preceding_signals[row_index]}|{buyer_action}->"))
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "action_sha256": action_sha256,
                    "forecast": {
                        "contract": BUYER_CONTINUATION_LIVE_CONTRACT,
                        "release_id": self.release_id,
                        "release_manifest_sha256": file_sha256(self.manifest_path),
                        "authority": BUYER_CONTINUATION_AUTHORITY,
                        "labels": list(BUYER_CONTINUATION_LABELS),
                        "response_probabilities": probabilities,
                        "predicted_response": BUYER_CONTINUATION_LABELS[max(range(len(probabilities)), key=probabilities.__getitem__)],
                        "component_probabilities": {str(seed): [float(value) for value in values[row_index].tolist()] for (seed, _head), values in zip(self.heads, component_probabilities, strict=True)},
                        "buyer_action": buyer_action,
                        "preceding_signal": preceding_signals[row_index],
                        "retrospective_cell_support": support,
                        "support_warning": "sparse-action-history-cell" if support < 200 else None,
                    },
                }
            )
        total_variation = 0.5 * sum(abs(float(ensemble[0, index]) - float(ensemble[1, index])) for index in range(len(BUYER_CONTINUATION_LABELS)))
        prediction = {
            "contract": BUYER_CONTINUATION_LIVE_CONTRACT,
            "release_id": self.release_id,
            "release_manifest_sha256": file_sha256(self.manifest_path),
            "authority": BUYER_CONTINUATION_AUTHORITY,
            "source_turn_id": turn_id,
            "game_id": str(game.get("game_id") or ""),
            "target_round": target_rounds[0],
            "candidate_set_sha256": candidate_set_sha256,
            "prefix_sha256": prefix_hashes[0],
            "rows": rows,
            "action_sensitivity": {"total_variation_distance": round(total_variation, 8), "interpretation": "predictive observed-policy sensitivity, not an identified intervention effect"},
            "causal_boundary": "The candidate buy/pass action is present; response time and any buy-caused quality reveal are absent; the target is the opponent seller's next-round signal.",
        }
        registry_receipt = self.registry.register(prediction) if register and self.registry is not None else {"status": "not-registered"}
        return {**prediction, "status": "predicted", "registry": registry_receipt, "execution_device": str(self.device)}

    def status(self) -> dict[str, object]:
        return {
            "contract": BUYER_CONTINUATION_LIVE_CONTRACT,
            "release_id": self.release_id,
            "release_manifest_sha256": file_sha256(self.manifest_path),
            "authority": BUYER_CONTINUATION_AUTHORITY,
            "device": str(self.device),
            "components": len(self.heads),
            "registry": self.registry.summary() if self.registry is not None else None,
        }
