from __future__ import annotations

import pytest

from glee_sequence_lab.shadow import ProspectiveShadowRegistry


def _prediction() -> dict[str, object]:
    return {
        "contract": "glee-sequence-shadow-candidate-v1",
        "candidate_id": "candidate-a",
        "family": "bargaining",
        "game_id": "game-a",
        "target_event_index": 4,
        "target_kind": "response",
        "labels": ["accept", "reject"],
        "action_probabilities": [0.4, 0.6],
        "predicted_action": "reject",
        "authority": "prospective-shadow-only",
    }


def test_prospective_registry_is_idempotent_but_never_overwrites(tmp_path) -> None:
    registry_path = tmp_path / "shadow.sqlite3"
    prefix_sha256 = "a" * 64
    with ProspectiveShadowRegistry(registry_path) as registry:
        first = registry.register(_prediction(), prefix_event_count=4, prefix_sha256=prefix_sha256, registered_at="2026-08-16T00:00:00Z")
        second = registry.register(_prediction(), prefix_event_count=4, prefix_sha256=prefix_sha256, registered_at="2026-08-16T00:00:01Z")
        assert first["status"] == "registered"
        assert second["status"] == "already-registered"
        changed = {**_prediction(), "action_probabilities": [0.5, 0.5]}
        with pytest.raises(ValueError, match="different immutable evidence"):
            registry.register(changed, prefix_event_count=4, prefix_sha256=prefix_sha256)
        outcome = registry.record_outcome(candidate_id="candidate-a", game_id="game-a", target_event_index=4, target_kind="response", outcome={"action": "reject"}, observed_at="2026-08-16T00:01:00Z")
        repeated = registry.record_outcome(candidate_id="candidate-a", game_id="game-a", target_event_index=4, target_kind="response", outcome={"action": "reject"})
        assert outcome["status"] == "recorded"
        assert repeated["status"] == "already-recorded"
        with pytest.raises(ValueError, match="different immutable outcome"):
            registry.record_outcome(candidate_id="candidate-a", game_id="game-a", target_event_index=4, target_kind="response", outcome={"action": "accept"})


def test_prospective_registry_rejects_outcome_leakage_and_late_prefix(tmp_path) -> None:
    with ProspectiveShadowRegistry(tmp_path / "shadow.sqlite3") as registry:
        with pytest.raises(ValueError, match="immediately before"):
            registry.register(_prediction(), prefix_event_count=3, prefix_sha256="b" * 64)
        leaked = {**_prediction(), "actual_action": "reject"}
        with pytest.raises(ValueError, match="observed outcome"):
            registry.register(leaked, prefix_event_count=4, prefix_sha256="b" * 64)
        with pytest.raises(ValueError, match="without a preregistered prediction"):
            registry.record_outcome(candidate_id="candidate-a", game_id="missing", target_event_index=4, target_kind="response", outcome={"action": "reject"})
