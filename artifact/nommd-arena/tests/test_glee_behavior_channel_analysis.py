from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from nommd_arena.glee_behavior_channel_analysis import GleeBehaviorChannelAnalysis, action_features, chronological_game_splits, discourse_features, fit_fingerprint_profile, lexical_features, timing_features


def _time(index: int) -> str:
    return (datetime(2026, 8, 12, tzinfo=timezone.utc) + timedelta(minutes=index)).isoformat()


def _language(token: str, *, present: bool = True) -> dict[str, object]:
    return {
        "present": present,
        "family_act": "urgency" if present else "none",
        "message_sha256": "forbidden-exact-message-hash",
        "style": {"chars": 20, "words": 4, "sentences": 1, "uppercase_ratio": 0.1, "digit_ratio": 0.1, "question_marks": 0, "exclamation_marks": 1, "commas": 1, "semicolons": 0, "currency_marks": 1, "percent_marks": 0, "decimal_numbers": 0, "contractions": 0, "opening_sha256": f"opening-{token}", "ending_sha256": f"ending-{token}"},
        "discourse_acts": ["explicit-request", "urgency"] if present else ["silence"],
        "hashed_lexemes": {token: 2} if present else {},
    }


def _record(index: int, identity: str, *, completion_offset: int = 1, action_value: float = 0.5) -> dict[str, object]:
    return {
        "game_id": f"g{index}",
        "family": "bargaining",
        "public_player_id": identity,
        "started_at": _time(index),
        "completed_at": _time(index + completion_offset),
        "moves": [
            {"kind": "proposal", "action_value": action_value, "decision": None, "response_time_ms": None, "context": {"round_phase": 0.0, "complete_information": True, "horizon_known": True, "messages_allowed": True}, "language": _language(f"lexeme-{identity}")},
            {"kind": "response", "action_value": action_value, "decision": "accept", "response_time_ms": 4_000 + index * 100, "context": {"round_phase": 0.2, "complete_information": True, "horizon_known": True, "messages_allowed": True}, "language": None},
        ],
    }


def test_channels_keep_timing_action_lexical_and_discourse_evidence_separate() -> None:
    game = _record(0, "alpha")

    timing = timing_features(game)
    action = action_features(game)
    lexical = lexical_features(game)
    discourse = discourse_features(game)

    assert timing and all(token.startswith("timing") for token in timing)
    assert action and not any("lexeme-alpha" in token or "urgency" in token for token in action)
    assert any("lexeme-alpha" in token for token in lexical)
    assert not any("forbidden-exact-message-hash" in token for token in lexical)
    assert any("urgency" in token for token in discourse)
    assert not any("lexeme-alpha" in token for token in discourse)


def test_chronological_split_purges_a_game_crossing_the_calibration_boundary() -> None:
    records = [_record(index, "alpha" if index % 2 == 0 else "beta") for index in range(10)]
    records[5]["completed_at"] = _time(7)

    assignments, boundaries = chronological_game_splits(records)

    assert [assignments[f"g{index}"] for index in range(5)] == ["train"] * 5
    assert assignments["g5"] == "purged-train-calibration-overlap"
    assert assignments["g6"] == "calibration"
    assert assignments["g7"] == "purged-calibration-test-overlap"
    assert [assignments[f"g{index}"] for index in range(8, 10)] == ["test"] * 2
    assert boundaries["bargaining"]["counts"]["purged-train-calibration-overlap"] == 1
    assert boundaries["bargaining"]["counts"]["purged-calibration-test-overlap"] == 1


def test_lexical_profile_requires_candidate_specific_cross_game_recurrence() -> None:
    records = [_record(index, "alpha" if index < 3 else "beta") for index in range(6)]
    vectors = {
        "g0": {"lexeme|shared-once-each": 1, "lexeme|alpha-recurring": 1},
        "g1": {"lexeme|alpha-recurring": 1},
        "g2": {"style|words=1": 1},
        "g3": {"lexeme|shared-once-each": 1},
        "g4": {"style|words=1": 1},
        "g5": {"style|words=1": 1},
    }

    profile = fit_fingerprint_profile(records, vectors, candidates=("alpha", "beta"), alpha=20.0, channel="lexical")

    assert "lexeme|shared-once-each" in profile.vocabulary
    assert "lexeme|shared-once-each" not in profile.identity_counts["alpha"]
    assert "lexeme|shared-once-each" not in profile.identity_counts["beta"]
    assert profile.identity_counts["alpha"]["lexeme|alpha-recurring"] == 2.0


def test_end_to_end_channel_evaluation_writes_shadow_receipts(tmp_path) -> None:
    behavior_dir = tmp_path / "behavior"
    output_dir = tmp_path / "evaluation"
    behavior_dir.mkdir()
    records = [_record(index, "alpha" if index % 2 == 0 else "beta", action_value=0.3 if index % 2 == 0 else 0.7) for index in range(15)]
    corpus_path = behavior_dir / "behavior-games.jsonl"
    corpus_path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (behavior_dir / "manifest.json").write_text(json.dumps({"frontier_sequence": 42, "artifacts": {"behavior-games.jsonl": {"sha256": digest}}}), encoding="utf-8")

    result = GleeBehaviorChannelAnalysis(behavior_dir=behavior_dir, output_dir=output_dir, minimum_profile_games=2).run()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert result["promotion"]["identity_routing_authority"] is False
    assert summary["families"]["bargaining"]["action"]["status"] == "offline-shadow-only"
    assert summary["families"]["bargaining"]["action"]["test"]["games"] == 3
    assert (output_dir / "test-predictions.jsonl").stat().st_size > 0
    assert (output_dir / "manifest.json").is_file()
