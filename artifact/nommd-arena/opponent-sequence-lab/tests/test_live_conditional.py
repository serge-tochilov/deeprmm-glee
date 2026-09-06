from __future__ import annotations

import json

from glee_sequence_lab.corpus import file_sha256, object_sha256
from glee_sequence_lab.live_conditional import LIVE_CONDITIONAL_ACTIVATION_CONTRACT, LIVE_CONDITIONAL_IPC_CONTRACT, LiveConditionalAdapter


def _game() -> dict[str, object]:
    return {"game_id": "live-conditional", "game_family": "negotiation", "your_player": "player_1", "phase": "offer", "opponent": {"type": "hidden", "name": None}, "valid_actions": {"type": "offer", "fields": {}}, "game_state": {"history": [], "round": 1, "complete_information": False, "horizon_known": False, "messages_allowed": True, "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 20.0, "player_2_value": 80.0}}


class _Release:
    candidate_id = "test-candidate"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict_candidates(self, **values: object) -> list[dict[str, object]]:
        self.calls.append(values)
        labels = ["buy", "pass"] if values["family"] == "persuasion" else ["accept", "reject", "walkaway"]
        probabilities = [0.6, 0.4] if values["family"] == "persuasion" else [0.6, 0.35, 0.05]
        return [{"authority": "prospective-shadow-only", "labels": labels, "response_probabilities": probabilities, "projection": candidate} for candidate in values["candidates"]]


def test_live_conditional_adapter_batches_unique_projections_and_realigns_candidates(tmp_path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"candidate_id": "test-candidate"}), encoding="utf-8")
    activation_path = tmp_path / "activation.json"
    activation_path.write_text(json.dumps({"contract": LIVE_CONDITIONAL_ACTIVATION_CONTRACT, "status": "active-controlled-online-evaluation", "activation_id": "test-live", "controller": "test-controller", "decision_authority": "advisory-candidate-response-evidence-only", "conditional_twin": {"candidate_id": "test-candidate", "manifest_sha256": file_sha256(manifest_path)}}), encoding="utf-8")
    release = _Release()
    adapter = LiveConditionalAdapter(release_dir=release_dir, activation_path=activation_path, release=release)
    game = _game()
    actions = [{"product_price": 50.0, "message": "same"}, {"product_price": 50.0, "message": "same"}, {"product_price": 60.0, "message": "other"}]
    candidates = [{"candidate_index": index, "action_sha256": object_sha256(action), "action": action} for index, action in enumerate(actions)]
    response = adapter.forecast_candidates(game=game, turn_id="turn-1", synthetic_features={"contract": "glee-terra-synthetic-feature-bundle-v1", "game_family": "negotiation", "phase": "offer", "producer_keys": [], "features": {}}, candidate_set_sha256="a" * 64, candidates=candidates)
    assert response["contract"] == LIVE_CONDITIONAL_IPC_CONTRACT
    assert response["candidate_count"] == 3
    assert response["unique_projection_count"] == 2
    assert response["projection_index_by_candidate"] == [0, 0, 1]
    assert len(release.calls) == 1
    assert len(release.calls[0]["candidates"]) == 2
    assert [row["action_sha256"] for row in response["rows"]] == [object_sha256(action) for action in actions]


def test_live_conditional_warmup_covers_all_family_heads_without_counting_live_requests(tmp_path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"candidate_id": "test-candidate"}), encoding="utf-8")
    activation_path = tmp_path / "activation.json"
    activation_path.write_text(json.dumps({"contract": LIVE_CONDITIONAL_ACTIVATION_CONTRACT, "status": "active-controlled-online-evaluation", "activation_id": "test-live", "controller": "test-controller", "decision_authority": "advisory-candidate-response-evidence-only", "conditional_twin": {"candidate_id": "test-candidate", "manifest_sha256": file_sha256(manifest_path)}}), encoding="utf-8")
    release = _Release()
    adapter = LiveConditionalAdapter(release_dir=release_dir, activation_path=activation_path, release=release)
    receipt = adapter.warmup()
    assert receipt["path_count"] == 3
    assert [call["family"] for call in release.calls] == ["bargaining", "negotiation", "persuasion"]
    assert adapter.status()["request_count"] == 0
    assert adapter.status()["warmup_count"] == 3
