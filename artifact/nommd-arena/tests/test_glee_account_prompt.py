from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from nommd_arena.glee_account_prompt import ACCOUNT_PROMPT_CONTRACT, MODEL_VERSION, OpponentAccountPromptModelReader


def _write_model(root: Path) -> None:
    profile = {
        "alpha": 1.0,
        "candidates": ["alpha", "beta"],
        "global_probability": {},
        "identity_counts": {"alpha": {}, "beta": {}},
        "identity_totals": {"alpha": 0.0, "beta": 0.0},
        "vocabulary": [],
    }
    model = {
        "schema_version": 1,
        "contract": ACCOUNT_PROMPT_CONTRACT,
        "model_version": MODEL_VERSION,
        "unknown_label": "__unlinked_or_unknown__",
        "supported_live_family": "bargaining",
        "stacker": {"feature_names": ["timing", "action", "lexical", "discourse"], "scales": {"timing": 1.0, "action": 1.0, "lexical": 1.0, "discourse": 1.0}, "weights": {"timing": 0.0, "action": 0.0, "lexical": 0.0, "discourse": 0.0}, "unknown_bias": -10.0},
        "families": {
            "bargaining": {
                "candidates": ["alpha", "beta"],
                "temperatures": {"timing": 1.0, "action": 1.0, "lexical": 1.0, "discourse": 1.0},
                "profiles": {"timing": profile, "action": profile, "lexical": profile, "discourse": profile},
                "validated_accounts": {"alpha": {"correct": 4, "named_predictions": 4, "precision": 1.0, "linkage_confidence": "high", "member_labels": ["Alpha-1", "Alpha-2"], "member_public_ids": ["a-1", "a-2"]}},
            }
        },
    }
    release = root / "releases" / "test"
    release.mkdir(parents=True)
    model_bytes = gzip.compress((json.dumps(model, separators=(",", ":"), sort_keys=True) + "\n").encode(), mtime=0)
    (release / "model.json.gz").write_bytes(model_bytes)
    manifest = {"schema_version": 1, "contract": ACCOUNT_PROMPT_CONTRACT, "model_version": MODEL_VERSION, "artifacts": {"model.json.gz": {"sha256": hashlib.sha256(model_bytes).hexdigest()}}}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (release / "manifest.json").write_bytes(manifest_bytes)
    current = {"schema_version": 1, "contract": ACCOUNT_PROMPT_CONTRACT, "model_version": MODEL_VERSION, "release": "releases/test", "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    (root / "current.json").write_text(json.dumps(current), encoding="utf-8")


def _hidden_bargaining_game() -> dict[str, object]:
    return {
        "game_id": "game-1",
        "game_family": "bargaining",
        "your_player": "player_2",
        "opponent": {"type": "hidden", "name": None},
        "game_state": {
            "money_to_divide": 100,
            "round": 1,
            "history": [{"round": 1, "proposer": "player_1", "offer": {"round": 1, "proposer": "player_1", "player_1_gain": 60, "player_2_gain": 40, "message": "A workable split."}}],
            "complete_information": False,
            "horizon_known": False,
            "messages_allowed": True,
        },
    }


def test_reader_emits_only_validated_hidden_bargaining_context(tmp_path: Path) -> None:
    _write_model(tmp_path)
    reader = OpponentAccountPromptModelReader(tmp_path)

    assessment = reader.assess(_hidden_bargaining_game())

    assert assessment.receipt["status"] == "admitted"
    assert assessment.prompt_context is not None
    assert assessment.prompt_context["candidate_account"] == "alpha"
    assert assessment.prompt_context["candidate_member_labels"] == ["Alpha-1", "Alpha-2"]
    assert assessment.prompt_context["authority"] == "advisory-only"


def test_reader_abstains_before_behavior_and_for_known_identity(tmp_path: Path) -> None:
    _write_model(tmp_path)
    reader = OpponentAccountPromptModelReader(tmp_path)
    game = _hidden_bargaining_game()
    game["game_state"]["history"] = []

    awaiting = reader.assess(game)
    assert awaiting.receipt["status"] == "awaiting-first-opponent-move"
    assert awaiting.prompt_context is None

    game["opponent"] = {"type": "agent", "name": "Alpha-1"}
    known = reader.assess(game)
    assert known.receipt["status"] == "known-identity-not-routed"
    assert known.prompt_context is None
