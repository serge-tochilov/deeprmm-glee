import json
from pathlib import Path
from typing import Any

from nommd_arena.glee_named_dossier import (
    NamedOpponentCorpusStore,
    freeze_named_dossier_snapshot,
    named_dossier_view,
    named_opponent_id,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _source_run(root: Path) -> Path:
    run = root / "sealed-run"
    game_id = "game-1"
    turn_id = f"{game_id}:r1:offer:abcd"
    user = {
        "objective": "Maximize payoff.",
        "turn_receipt": {"turn_id": turn_id, "state_hash": "state", "snapshot_id": "snapshot"},
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "opponent": {"type": "agent", "name": "Aster"},
        "official_prompt": "Offer a split.",
        "visible_game_state": {"round": 1, "history": []},
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number"}},
        "tetrad_memory": {"activated_records": [{"content": "unrelated global Self claim"}]},
    }
    hidden_user = {**user, "turn_receipt": {"turn_id": "hidden:r1:offer:abcd"}, "opponent": {"type": "hidden", "name": None}}

    def call(call_id: str, effort: str, response: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ts": "2026-08-08T10:00:00+00:00",
            "backend": "codex-cli",
            "call_id": call_id,
            "attempt": 1,
            "role": "glee_nommd_bargaining",
            "model": "gpt-5.6-luna",
            "effort": effort,
            "thinking": True,
            "elapsed_s": 1.0,
            "ok": True,
            "transient": False,
            "prompt_version": "glee_nommd_bargaining@test",
            "request": {
                "system": "SYSTEM CONTRACT",
                "system_sha256": "system-hash",
                "user": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                "user_sha256": f"user-{call_id}",
            },
            "response": {"raw": response, "sha256": f"response-{call_id}"},
            "provider": {
                "event_stream": f"event-stream-{call_id}",
                "reasoning_items": [{"type": "reasoning", "text": f"reasoning-{call_id}"}],
                "stderr": f"stderr-{call_id}",
                "tokens_in": 100,
                "tokens_out": 20,
            },
        }

    calls = [
        call("xhigh-call", "xhigh", "XHigh exact output", user),
        call("max-call", "max", "Max exact output", user),
        call("hidden-call", "max", "Hidden output", hidden_user),
    ]
    game = {
        "game_id": game_id,
        "game_family": "bargaining",
        "status": "completed",
        "your_player": "player_1",
        "opponent": {"type": "agent", "name": "Aster"},
        "game_state": {"round": 1, "history": [{"offer": {"player_1_gain": 40, "player_2_gain": 60}}]},
        "result": {"player_1_payoff": 40, "player_2_payoff": 60},
    }
    events = [
        {"schema_version": 1, "kind": "turn_observed", "turn_id": turn_id, "game_id": game_id, "game": game},
        {
            "schema_version": 1,
            "kind": "worker_finished",
            "turn_id": turn_id,
            "decision": {
                "action": {"alice_gain": 40, "bob_gain": 60},
                "selection_branch": "max",
                "call_metadata": {"call_id": "max-call"},
                "branch_receipts": [
                    {"branch": "xhigh", "call_metadata": {"call_id": "xhigh-call"}},
                    {"branch": "max", "call_metadata": {"call_id": "max-call"}},
                ],
            },
        },
        {"schema_version": 1, "kind": "game_completed", "game_id": game_id, "result": game["result"]},
    ]
    _write_jsonl(run / "llm_calls.jsonl", calls)
    _write_jsonl(run / "events.jsonl", events)
    _write_json(run / "manifest.json", {"schema_version": 1, "mode": "test"})
    _write_json(run / "complete.json", {"schema_version": 1, "completed": True})
    _write_json(run / "games" / f"bargaining-{game_id}.json", game)
    return run


def test_exact_named_outputs_are_factored_into_immutable_shard(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    store = NamedOpponentCorpusStore(tmp_path / "dossiers")
    first = store.extract_run(source)
    second = store.extract_run(source)
    assert first["source_sha256"] == second["source_sha256"]
    opponent = first["opponents"][0]
    assert opponent["name"] == "Aster"
    assert opponent["call_count"] == 2
    shard = tmp_path / "dossiers" / opponent["path"]
    records = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    prompts = [record for record in records if record["kind"] == "system_prompt"]
    calls = [record for record in records if record["kind"] == "model_call"]
    assert len(prompts) == 1
    assert prompts[0]["system"] == "SYSTEM CONTRACT"
    assert [record["call"]["response"]["raw"] for record in calls] == ["XHigh exact output", "Max exact output"]
    assert [record["call"]["provider"]["event_stream"] for record in calls] == ["event-stream-xhigh-call", "event-stream-max-call"]
    assert [record["selected"] for record in calls] == [False, True]
    assert all("system" not in record["call"]["request"] for record in calls)
    assert "Hidden output" not in shard.read_text(encoding="utf-8")
    index = json.loads((tmp_path / "dossiers" / "index.json").read_text(encoding="utf-8"))
    source_entry = index["sources"][first["source_sha256"]]
    assert source_entry["named_call_count"] == 2
    assert source_entry["complete"] is True


def test_legacy_smoke_call_is_joined_through_move_receipt_call_id(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    call_path = source / "llm_calls.jsonl"
    calls = [json.loads(line) for line in call_path.read_text(encoding="utf-8").splitlines()]
    named_call = calls[1]
    payload = json.loads(named_call["request"]["user"])
    payload.pop("turn_receipt")
    named_call["request"]["user"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    _write_jsonl(call_path, [named_call])
    event_path = source / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    observation = events[0]
    observation.pop("turn_id")
    observation["turn"] = 1
    move = {
        "schema_version": 1,
        "kind": "move_submitted",
        "game_id": "game-1",
        "decision": {
            "action": {"alice_gain": 40, "bob_gain": 60},
            "call_metadata": {"call_id": "max-call"},
        },
        "result": {"valid": True, "game_over": True},
    }
    _write_jsonl(event_path, [observation, move, events[-1]])
    store = NamedOpponentCorpusStore(tmp_path / "legacy-dossiers")
    result = store.extract_run(source)
    assert result["opponents"][0]["game_ids"] == ["game-1"]
    shard = tmp_path / "legacy-dossiers" / result["opponents"][0]["path"]
    model_call = next(json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if json.loads(line)["kind"] == "model_call")
    assert model_call["game_id"] == "game-1"
    assert model_call["selected"] is True


def test_sol_synthesis_produces_3_provenance_bounded_family_synopses(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path)
    root = tmp_path / "dossiers"
    store = NamedOpponentCorpusStore(root)
    store.extract_run(source)
    captured: dict[str, object] = {"calls": []}

    class FakeRunner:
        def __init__(self, **settings: object) -> None:
            captured["settings"] = settings

        def call_structured(self, role: str, body: str, model_cls: type[Any], **settings: object) -> tuple[Any, dict[str, object]]:
            captured["calls"].append({"role": role, "body": body, "settings": settings})
            if role == "glee_named_opponent_digest":
                value = {
                    "chunk_summary": "The complete chunk contains one bargaining turn and both model branches.",
                    "authenticated_observations": ["A bargaining agreement paid DeepRMM-01 40."],
                    "model_hypotheses_and_errors": ["Both contemplated branch outputs were retained."],
                    "opponent_model_of_us": [],
                    "uncertainties": ["One game is insufficient for a stable tendency."],
                    "family_evidence": [
                        {"game_family": "bargaining", "evidence_notes": ["One direct game."]},
                        {"game_family": "negotiation", "evidence_notes": []},
                        {"game_family": "persuasion", "evidence_notes": []},
                    ],
                }
                return model_cls.model_validate(value), {"call_id": "digest-call"}
            value = {
                "common_evidence_summary": "Aster made one directly observed bargaining response.",
                "opponent_model_of_us": "There is not yet enough evidence to infer Aster's model of DeepRMM-01.",
                "recurring_tendencies": ["No recurring tendency is established from one game."],
                "uncertainties": ["Only one bargaining game is available."],
                "family_synopses": [
                    {"game_family": "bargaining", "confidence": 55, "prompt_synopsis": "Direct bargaining evidence: one encounter; use its result cautiously."},
                    {"game_family": "negotiation", "confidence": 10, "prompt_synopsis": "No direct negotiation evidence; bargaining behavior may not transfer."},
                    {"game_family": "persuasion", "confidence": 10, "prompt_synopsis": "No direct persuasion evidence; bargaining behavior may not transfer."},
                ],
            }
            return model_cls.model_validate(value), {"call_id": "sol-call"}

    monkeypatch.setattr("nommd_arena.glee_named_dossier.ArenaCodexRunner", FakeRunner)
    monkeypatch.setattr("nommd_arena.glee_named_dossier._MAX_DIRECT_SYNTHESIS_CHARS", 1)
    result = store.synthesize(prompts_dir=Path("prompts"))
    assert result["model"] == "gpt-5.6-sol"
    assert result["effort"] == "max"
    assert [call["role"] for call in captured["calls"]] == ["glee_named_opponent_digest", "glee_named_opponent_synthesis"]
    assert "XHigh exact output" in captured["calls"][0]["body"]
    assert "Max exact output" in captured["calls"][0]["body"]
    assert "unrelated global Self claim" not in captured["calls"][0]["body"]
    summary_path = root / result["opponents"][0]["path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    families = {item["game_family"]: item for item in summary["family_synopses"]}
    assert summary["synthesis"]["mode"] == "chronological-complete-coverage-digests"
    assert summary["synthesis"]["digest_count"] == 1
    assert families["bargaining"]["evidence_basis"] == "direct"
    assert families["bargaining"]["direct_game_count"] == 1
    assert families["negotiation"]["evidence_basis"] == "cross-family-only"
    assert families["persuasion"]["evidence_basis"] == "cross-family-only"
    frozen = freeze_named_dossier_snapshot(root, tmp_path / "frozen.json")
    view = named_dossier_view(frozen, "aster", "negotiation")
    assert view is not None
    assert view["provenance"]["model"] == "gpt-5.6-sol"
    assert view["family_synopsis"]["game_family"] == "negotiation"
    assert named_opponent_id(" Aster ") == named_opponent_id("aster")
