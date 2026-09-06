from __future__ import annotations

from pathlib import Path

import polars as pl

from glee_sequence_lab.meta_controller_replay_v15 import collapse_candidate_projections, select_replay_cases, submitted_action_from_event


def _row(*, family: str, identity: str, role: str, target: str, index: int, complete: bool = False, horizon: bool = False) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    game_id = f"{family}-{index}"
    sample_id = f"sample-{family}-{index}"
    game = {"game_id": game_id, "family": family, "identity_scope": identity, "our_player": role if family == "bargaining" else "player_1", "our_role": role, "complete_information": complete, "horizon_known": horizon}
    target_row = {"sample_id": sample_id, "game_id": game_id, "chronological_split": "test", "identity_scope": identity, "target_label": target}
    feature = {"sample_id": sample_id, "game_id": game_id, "turn_id": f"turn-{index}", "family": family, "phase": "seller_message" if family == "persuasion" else "offer", "round_number": index, "source_run": "/tmp/source", "source_llm_call_line": index, "source_call_ts": f"2026-08-16T00:{index:02d}:00+00:00", "source_user_sha256": f"{index:064x}"}
    return game, target_row, feature


def test_balanced_replay_selection_covers_12_frozen_strata_in_chronological_order(tmp_path: Path) -> None:
    triples = [
        _row(family="bargaining", identity="hidden", role="player_1", target="reject", index=1),
        _row(family="bargaining", identity="hidden", role="player_2", target="reject", index=2),
        _row(family="bargaining", identity="known", role="player_1", target="accept", index=3),
        _row(family="bargaining", identity="known", role="player_2", target="reject", index=4),
        _row(family="negotiation", identity="hidden", role="seller", target="accept", index=5),
        _row(family="negotiation", identity="hidden", role="seller", target="reject", index=6, horizon=True),
        _row(family="negotiation", identity="known", role="seller", target="reject", index=7, complete=True),
        _row(family="negotiation", identity="known", role="seller", target="reject", index=8, complete=False),
        _row(family="persuasion", identity="hidden", role="seller", target="buy", index=9),
        _row(family="persuasion", identity="hidden", role="seller", target="pass", index=10),
        _row(family="persuasion", identity="known", role="seller", target="buy", index=11),
        _row(family="persuasion", identity="known", role="seller", target="pass", index=12),
    ]
    pl.DataFrame([triple[0] for triple in triples], infer_schema_length=None).write_parquet(tmp_path / "games.parquet")
    pl.DataFrame([triple[1] for triple in triples], infer_schema_length=None).write_parquet(tmp_path / "targets.parquet")
    pl.DataFrame([triple[2] for triple in triples], infer_schema_length=None).write_parquet(tmp_path / "features.parquet")
    cases = select_replay_cases(tmp_path)
    assert len(cases) == 12
    assert [case.ordinal for case in cases] == list(range(1, 13))
    assert [case.source_call_ts for case in cases] == sorted(case.source_call_ts for case in cases)
    assert {case.stratum for case in cases} == {"hidden-player-1", "hidden-player-2", "known-player-1", "known-player-2", "hidden-accepted", "hidden-rejected-known-horizon", "known-rejected-complete-information", "known-rejected-incomplete-information", "hidden-buy", "hidden-pass", "known-buy", "known-pass"}


def test_submitted_action_reader_accepts_current_and_legacy_receipts() -> None:
    action = {"kind": "accept"}
    assert submitted_action_from_event({"kind": "move_submitted", "action": action}) == action
    assert submitted_action_from_event({"kind": "move_submitted", "decision": {"action": action}}) == action
    assert submitted_action_from_event({"kind": "move_submitted"}) is None


def test_predictor_equivalent_candidates_are_collapsed_with_full_alignment() -> None:
    candidates = [
        {"family": "persuasion", "phase": "seller_message", "kind": "signal", "action_label": "signal_positive", "action_value": 1.0, "action_aux_value": None, "round_number": 20, "round_phase": 1.0, "visible_quality": "high"},
        {"family": "persuasion", "phase": "seller_message", "kind": "signal", "action_label": "signal_positive", "action_value": 1.0, "action_aux_value": None, "round_number": 20, "round_phase": 1.0, "visible_quality": "high"},
        {"family": "persuasion", "phase": "seller_message", "kind": "signal", "action_label": "signal_negative", "action_value": -1.0, "action_aux_value": None, "round_number": 20, "round_phase": 1.0, "visible_quality": "high"},
    ]
    unique, indices = collapse_candidate_projections(candidates=candidates, family="persuasion", phase="seller_message")
    assert len(unique) == 2
    assert indices == [0, 0, 1]
