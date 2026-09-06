import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from nommd_arena.glee_incremental_dossier import DEFAULT_FAMILY_MODELS, DEFAULT_MAX_BATCH_GAMES, IncrementalDossierDraft, IncrementalNamedDossierReader, IncrementalNamedOpponentStore, resolve_dossier_family_models
from nommd_arena.glee_named_dossier import named_opponent_id
from nommd_arena.glee_synopsis import live_dossier_projection, normalize_bargaining_live_text


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for value in values), encoding="utf-8")


def _draft(name: str, game_family: str, game_ids: list[str]) -> dict[str, object]:
    latest = game_ids[-1]
    sibling = next(family for family in ("bargaining", "negotiation", "persuasion") if family != game_family)
    return {
        "game_family": game_family,
        "confidence": 60,
        "confidence_profile": {"within_observed_context": 70, "cross_game_generalization": 35, "cross_family_transfer": 20},
        "executive_model": f"Detailed {game_family} model of {name} through {latest}.",
        "latest_evidence_update": f"The new direct {game_family} evidence comprises {', '.join(game_ids)}.",
        "latest_game_updates": [{"game_id": game_id, "game_family": game_family, "evidence_update": f"Distinct evidence from {game_id}."} for game_id in game_ids],
        "direct_evidence_model": f"Only authenticated {game_family} games directly support this model of {name}.",
        "stable_direct_tendencies": [f"{name} has one cautiously inferred {game_family} tendency."],
        "role_conditioning": "Role transfer remains uncertain.",
        "phase_conditioning": "Phase transfer remains uncertain.",
        "response_to_pressure": "No stable pressure response is established.",
        "adaptation_and_learning": "Adaptation is not yet firmly established.",
        "communication_and_truthfulness": "Statements require comparison with authenticated outcomes.",
        "timing_and_resource_policy": "Timing evidence remains limited.",
        "opponent_model_of_us": "The opponent may model DeepRMM-01's visible concessions.",
        "exploitable_regularities": ["Test a small reversible probe."],
        "contradictions_and_counterexamples": ["Sparse evidence cannot establish invariance."],
        "sibling_context_assessment": "Sibling evidence is contextual and not direct evidence in this family.",
        "transferred_hypotheses": [{"source_family": sibling, "hypothesis": "A sibling-family tendency may transfer.", "confidence": 20, "rationale": "The mechanics differ."}],
        "uncertainties": ["Display-name identity is not authenticated."],
        "recommended_counterpolicy": "Probe before committing.",
        "prompt_synopsis": f"Current {game_family} synopsis for {name} after {latest}.",
    }


def _source_run(root: Path, opponents: dict[str, list[str | tuple[str, str]]], *, name: str = "source") -> Path:
    run = root / "runs" / name
    calls: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    order = 0
    for opponent, game_specs in opponents.items():
        for specification in game_specs:
            game_id, family = (specification, "bargaining") if isinstance(specification, str) else specification
            order += 1
            turn_id = f"{game_id}:r1:offer:state"
            call_id = f"call-{game_id}"
            state: dict[str, object] = {"round": 1, "history": [{"offer": {"player_1_gain": 50, "player_2_gain": 50}}]}
            if family == "persuasion":
                state = {
                    "round": 1,
                    "total_rounds": 20,
                    "history": [{"round": 1, "quality": "high", "seller_message": "yes", "buyer_decision": "yes", "bought": True}],
                    "product_price": 10,
                    "p": 0.8,
                    "u": 0,
                    "v": 12,
                    "is_seller_know_cv": False,
                    "seller_message_type": "binary",
                    "current_quality": "high",
                }
            game = {
                "game_id": game_id,
                "game_family": family,
                "status": "completed",
                "your_player": "player_1",
                "opponent": {"type": "agent", "name": opponent},
                "game_state": state,
                "result": {"player_1_payoff": 50, "player_2_payoff": 50},
            }
            user = {
                "objective": "Maximize payoff.",
                "turn_receipt": {"turn_id": turn_id},
                "game_family": family,
                "your_player": "player_1",
                "phase": "offer",
                "opponent": {"type": "agent", "name": opponent},
                "official_prompt": "Make an offer.",
                "visible_game_state": {"round": 1, "history": []},
                "valid_actions": {"type": "offer"},
                "tetrad_memory": {"activated_records": [{"content": "unrelated prior memory"}]},
            }
            calls.append(
                {
                    "schema_version": 1,
                    "call_id": call_id,
                    "role": f"glee_nommd_{family}",
                    "model": "gpt-5.6-luna",
                    "effort": "max",
                    "ok": True,
                    "request": {"system": "SYSTEM", "system_sha256": "system", "user": json.dumps(user), "user_sha256": f"user-{game_id}"},
                    "response": {"raw": f"Exact model output for {game_id}"},
                    "provider": {"event_stream": f"events for {game_id}", "reasoning_items": [{"text": f"reasoning for {game_id}"}]},
                }
            )
            timestamp = f"2026-08-08T00:00:{order:02d}+00:00"
            events.extend(
                [
                    {"schema_version": 1, "ts": timestamp, "kind": "turn_observed", "turn_id": turn_id, "game": game},
                    {
                        "schema_version": 1,
                        "ts": timestamp,
                        "kind": "worker_finished",
                        "turn_id": turn_id,
                        "decision": {"action": {"player_1_gain": 50, "player_2_gain": 50}, "selection_branch": "max", "call_metadata": {"call_id": call_id}, "branch_receipts": [{"branch": "max", "call_metadata": {"call_id": call_id}}]},
                    },
                    {"schema_version": 1, "ts": timestamp, "kind": "game_completed", "game_id": game_id, "result": game["result"]},
                ]
            )
            _write_json(run / "games" / f"{family}-{game_id}.json", game)
    _write_jsonl(run / "llm_calls.jsonl", calls)
    _write_jsonl(run / "events.jsonl", events)
    _write_json(run / "manifest.json", {"schema_version": 1})
    _write_json(run / "complete.json", {"schema_version": 1, "completed": True})
    return run


def test_future_synopsis_schema_rejects_visible_self_correction_and_execution_mechanics() -> None:
    correction = _draft("Aster", "bargaining", ["b-1"])
    correction["prompt_synopsis"] = "Aster accepted 38.2%? No, 38.5%."
    with pytest.raises(ValueError, match="visible_self_correction"):
        IncrementalDossierDraft.model_validate(correction)
    mechanics = _draft("Aster", "bargaining", ["b-1"])
    mechanics["prompt_synopsis"] = "The selected XHigh branch used a non-fallback action."
    with pytest.raises(ValueError, match="internal_execution_detail"):
        IncrementalDossierDraft.model_validate(mechanics)


def test_live_scale_scrubbing_covers_negotiation_and_bargaining_without_mutating_dimensionless_evidence() -> None:
    source = {
        "stable_direct_tendencies": ["Aster moved from $1.2M to $800K while preserving a 0.8 reservation-value ratio."],
        "opponent_model_of_us": "It may remember $1,200,000.",
        "uncertainties": ["The $800,000 anchor may be scale-specific."],
    }
    bargaining = live_dossier_projection({"game_family": "bargaining", **source})
    negotiation = live_dossier_projection({"game_family": "negotiation", **source})
    assert bargaining["version"] == "opponent-evidence-v3"
    assert "$" not in json.dumps(bargaining, ensure_ascii=False)
    assert bargaining["semantics"]["nominal_amounts_removed"] == 4
    assert "0.8 reservation-value ratio" in bargaining["opponent_behavior"]
    assert negotiation["version"] == "opponent-evidence-v2"
    assert negotiation["semantics"]["nominal_amounts_removed"] == 4
    assert "$" not in json.dumps(negotiation, ensure_ascii=False)
    assert "0.8 reservation-value ratio" in negotiation["opponent_behavior"]


def test_bargaining_projection_normalizes_explicit_pairs_and_omits_denominatorless_amounts() -> None:
    source = "OpponentAlpha demanded a 0/1,000,000 split and later repeated 622331.3349900001, while its 62.233% threshold remained stable."
    projected, removed, pairs = normalize_bargaining_live_text(source)
    assert "0.000% / 100.000% allocation" in projected
    assert "[prior-scale amount omitted]" in projected
    assert "62.233%" in projected
    assert removed == 1
    assert pairs == 1


def test_family_lanes_are_serial_and_concurrent_for_one_opponent(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining"), ("b-2", "bargaining"), ("p-1", "persuasion"), ("p-2", "persuasion")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    assert store.enqueue_source_run(source)["named_game_count"] == 4
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active: set[tuple[str, str]] = set()
    seen: dict[tuple[str, str], int] = {}
    same_lane_overlap: list[tuple[str, str]] = []
    maximum_active = 0
    packets: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            nonlocal maximum_active
            packet = json.loads(body)
            assert len(packet["new_game_evidence_batch"]) == 1
            name = packet["new_game_evidence_batch"][0]["opponent"]["name"]
            family = packet["transition"]["game_family"]
            lane = (name, family)
            game_id = packet["new_game_evidence_batch"][0]["game_id"]
            assert packet["new_game_evidence_batch"][0]["game_family"] == family
            assert "price_transfer" not in packet["state_contract"]
            with lock:
                if lane in active:
                    same_lane_overlap.append(lane)
                active.add(lane)
                seen[lane] = seen.get(lane, 0) + 1
                maximum_active = max(maximum_active, len(active))
                packets.append(packet)
                first = seen[lane] == 1
            if first:
                barrier.wait(timeout=2)
            with lock:
                active.remove(lane)
            assert role == "glee_named_opponent_incremental"
            return model_cls.model_validate(_draft(name, family, [game_id])), {"call_id": f"sol-{game_id}"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    result = store.process_until_idle(max_workers=2, max_batch_games=1)
    assert result["published_count"] == 4
    assert maximum_active == 2
    assert same_lane_overlap == []
    opponent_key = named_opponent_id("Aster")
    for family in ("bargaining", "persuasion"):
        pointer = json.loads((root / "opponents" / opponent_key / family / "current.json").read_text(encoding="utf-8"))
        dossier = json.loads((root / pointer["path"]).read_text(encoding="utf-8"))
        assert pointer["revision_number"] == 2
        assert dossier["revision"]["direct_game_count"] == 2
        assert all(job["game_family"] == family for job in dossier["revision"]["new_jobs"])
        revision_dirs = sorted((root / "opponents" / opponent_key / family / "revisions").iterdir())
        assert all((revision_dir / "input.ref.json").is_file() and not (revision_dir / "input.json").exists() for revision_dir in revision_dirs)
        lane_packets = sorted((packet for packet in packets if packet["transition"]["game_family"] == family), key=lambda packet: packet["transition"]["revision_number"])
        assert lane_packets[0]["current_family_dossier"] is None
        assert lane_packets[1]["current_family_dossier"]["revision"]["number"] == 1
    assert "unrelated prior memory" not in json.dumps(packets)


def test_source_enqueue_filters_family_and_terminal_outcome(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {"Aster": [("p-1", "persuasion"), ("p-2", "persuasion"), ("b-1", "bargaining")]})
    timeout_path = source / "games" / "persuasion-p-2.json"
    timeout_game = json.loads(timeout_path.read_text(encoding="utf-8"))
    timeout_game["result"] = {"outcome": "timeout", "player_1_payoff": 0, "player_2_payoff": 0}
    _write_json(timeout_path, timeout_game)
    root = tmp_path / "opponent-dossiers" / "staging"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    result = store.enqueue_source_run(source, families={"persuasion"}, exclude_outcomes={"timeout"})
    assert result["named_game_count"] == 1
    assert result["excluded_game_count"] == 2
    assert result["families"] == ["persuasion"]
    assert result["excluded_outcomes"] == ["timeout"]
    jobs = list(root.glob("jobs/*/*.json"))
    assert len(jobs) == 1
    assert json.loads(jobs[0].read_text(encoding="utf-8"))["game_id"] == "p-1"


def test_watcher_family_filter_leaves_foreign_existing_lanes_pending(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining"), ("n-1", "negotiation")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    calls: list[str] = []
    events: list[dict[str, object]] = []

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            packet = json.loads(body)
            family = packet["transition"]["game_family"]
            assert packet["state_contract"]["price_transfer"].startswith("Make every reusable price comparison")
            calls.append(family)
            game_ids = [game["game_id"] for game in packet["new_game_evidence_batch"]]
            return model_cls.model_validate(_draft("Aster", family, game_ids)), {"call_id": f"synthesis-{family}"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    store.watch(max_workers=1, families={"negotiation"}, poll_interval_s=0.01, stop_when_idle=lambda: True, event=events.append)
    opponent_key = named_opponent_id("Aster")
    assert calls == ["negotiation"]
    assert [event["game_family"] for event in events if event["kind"] == "revision_started"] == ["negotiation"]
    assert (root / "opponents" / opponent_key / "negotiation" / "current.json").is_file()
    assert not (root / "opponents" / opponent_key / "bargaining" / "current.json").exists()
    assert store.has_pending(opponent_key, "bargaining") is True


def test_timeout_jobs_remain_preserved_but_never_enter_pending_synthesis(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {"Aster": [("p-1", "persuasion"), ("p-timeout", "persuasion")]})
    timeout_path = source / "games" / "persuasion-p-timeout.json"
    timeout_game = json.loads(timeout_path.read_text(encoding="utf-8"))
    timeout_game["result"] = {"outcome": "timeout", "player_1_payoff": 0, "player_2_payoff": 0}
    _write_json(timeout_path, timeout_game)
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    result = store.enqueue_source_run(source)
    assert result["named_game_count"] == 2
    opponent_key = named_opponent_id("Aster")
    assert [job["game_id"] for job in store._ordered_jobs(opponent_key, "persuasion")] == ["p-1", "p-timeout"]
    assert [job["game_id"] for job in store.pending_jobs(opponent_key, "persuasion")] == ["p-1"]


def test_family_frequency_isolation_and_same_family_batching(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining"), ("b-2", "bargaining"), ("b-3", "bargaining"), ("b-4", "bargaining"), ("b-5", "bargaining"), ("b-6", "bargaining"), ("b-7", "bargaining"), ("p-1", "persuasion")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    captured: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            packet = json.loads(body)
            captured.append(packet)
            family = packet["transition"]["game_family"]
            game_ids = [game["game_id"] for game in packet["new_game_evidence_batch"]]
            assert {game["game_family"] for game in packet["new_game_evidence_batch"]} == {family}
            return model_cls.model_validate(_draft("Aster", family, game_ids)), {"call_id": f"sol-{family}"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    result = store.process_until_idle(max_workers=2, max_batch_chars=700000)
    assert DEFAULT_MAX_BATCH_GAMES == 5
    assert result["published_count"] == 4
    assert [revision["batch_size"] for revision in result["revisions"] if revision["game_family"] == "bargaining"] == [1, 5, 1]
    assert [revision["batch_size"] for revision in result["revisions"] if revision["game_family"] == "persuasion"] == [1]
    opponent_root = root / "opponents" / named_opponent_id("Aster")
    bargaining_pointer = json.loads((opponent_root / "bargaining" / "current.json").read_text(encoding="utf-8"))
    persuasion_pointer = json.loads((opponent_root / "persuasion" / "current.json").read_text(encoding="utf-8"))
    bargaining = json.loads((root / bargaining_pointer["path"]).read_text(encoding="utf-8"))
    persuasion = json.loads((root / persuasion_pointer["path"]).read_text(encoding="utf-8"))
    assert bargaining_pointer["revision_number"] == 3
    assert bargaining["revision"]["direct_game_count"] == 7
    assert persuasion["revision"]["direct_game_count"] == 1
    assert not (opponent_root / "negotiation" / "current.json").exists()


def test_default_model_is_allocated_by_game_family_and_recorded(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining"), ("n-1", "negotiation"), ("p-1", "persuasion")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    calls: dict[str, tuple[str, str]] = {}

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **settings: object) -> tuple[Any, dict[str, object]]:
            packet = json.loads(body)
            family = packet["transition"]["game_family"]
            game_ids = [game["game_id"] for game in packet["new_game_evidence_batch"]]
            calls[family] = (str(settings["model"]), str(settings["effort"]))
            return model_cls.model_validate(_draft("Aster", family, game_ids)), {"call_id": f"synthesis-{family}"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    result = store.process_until_idle(max_workers=3)
    assert calls == {family: (model, "max") for family, model in DEFAULT_FAMILY_MODELS.items()}
    assert {revision["game_family"]: revision["model"] for revision in result["revisions"]} == DEFAULT_FAMILY_MODELS
    opponent_root = root / "opponents" / named_opponent_id("Aster")
    for family, model in DEFAULT_FAMILY_MODELS.items():
        pointer = json.loads((opponent_root / family / "current.json").read_text(encoding="utf-8"))
        dossier = json.loads((root / pointer["path"]).read_text(encoding="utf-8"))
        assert pointer["model"] == model
        assert dossier["synthesis"]["model"] == model
    assert resolve_dossier_family_models(model="one-model") == {family: "one-model" for family in DEFAULT_FAMILY_MODELS}


def test_synthesis_projection_preserves_model_outputs_without_repeated_transport_context(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    opponent_key = named_opponent_id("Aster")
    job = store._ordered_jobs(opponent_key, "bargaining")[0]
    bundle, _path, _sha256 = store._build_game_bundle(job)
    projection = store._synthesis_projection(bundle)
    assert projection["projection_contract"]["version"] == "compact-behavioral-v3"
    assert bundle["kind"] == "named-opponent-game-evidence-compact-v1"
    assert projection["final_game"] == store.load_job_final_game(job)
    assert len(projection["model_calls"]) == 1
    projected_call = projection["model_calls"][0]["call"]
    source_call = json.loads((source / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert projected_call["response"] == source_call["response"]
    assert projected_call["request_receipt"] == {"system_sha256": "system", "user_sha256": "user-b-1"}
    assert "request" not in projected_call
    assert "provider" not in projected_call
    assert all("game" not in receipt["event"] for receipt in projection["events"])
    assert "unrelated prior memory" not in json.dumps(projection)


def test_game_bundle_streams_source_logs_instead_of_materializing_whole_jsonl_files(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    job = store._ordered_jobs(named_opponent_id("Aster"), "bargaining")[0]
    original_read_text = Path.read_text

    def reject_jsonl_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.suffix == ".jsonl":
            raise AssertionError(f"whole JSONL materialization attempted: {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_jsonl_read_text)
    bundle, _path, _sha256 = store._build_game_bundle(job)
    assert bundle["game_id"] == "b-1"
    assert len(bundle["projection"]["model_calls"]) == 1
    assert "model_calls" not in bundle


def test_persuasion_synthesis_projection_normalizes_information_semantics(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {"Aster": [("p-1", "persuasion")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    opponent_key = named_opponent_id("Aster")
    job = store._ordered_jobs(opponent_key, "persuasion")[0]
    bundle, _path, _sha256 = store._build_game_bundle(job)
    projection = store._synthesis_projection(bundle)
    visible = projection["final_game"]["game_state"]
    assert visible["seller_knows_buyer_values"] is False
    assert visible["seller_always_observes_current_quality"] is True
    assert "is_seller_know_cv" not in visible
    assert projection["game_semantics"]["version"] == "persuasion-information-v1"
    assert projection["model_calls"][0]["call"]["persuasion_semantics"]["status"] == "legacy-ambiguous-semantics"
    assert store.load_job_final_game(job)["game_state"]["is_seller_know_cv"] is False


def test_single_game_projection_must_fit_serialized_input_limit(tmp_path: Path) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)
    with pytest.raises(RuntimeError, match="single-game dossier projection exceeds max_batch_chars"):
        store.process_next(named_opponent_id("Aster"), "bargaining", max_batch_chars=1)


def test_sibling_snapshot_is_provenance_pinned_and_excludes_recursive_transfer(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    captured: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            packet = json.loads(body)
            captured.append(packet)
            family = packet["transition"]["game_family"]
            game_ids = [game["game_id"] for game in packet["new_game_evidence_batch"]]
            return model_cls.model_validate(_draft("Aster", family, game_ids)), {"call_id": f"sol-{family}"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    store.enqueue_source_run(_source_run(tmp_path, {"Aster": [("p-1", "persuasion")]}, name="persuasion-source"))
    store.process_until_idle(max_workers=1)
    opponent_key = named_opponent_id("Aster")
    persuasion_pointer = json.loads((root / "opponents" / opponent_key / "persuasion" / "current.json").read_text(encoding="utf-8"))
    store.enqueue_source_run(_source_run(tmp_path, {"Aster": [("b-1", "bargaining")]}, name="bargaining-source"))
    store.process_until_idle(max_workers=1)
    bargaining_packet = next(packet for packet in captured if packet["transition"]["game_family"] == "bargaining")
    persuasion_input = next(item for item in bargaining_packet["sibling_family_dossiers"] if item["game_family"] == "persuasion")
    assert persuasion_input["availability"] == "incremental-family-dossier"
    assert persuasion_input["provenance"]["revision_sha256"] == persuasion_pointer["sha256"]
    projection = persuasion_input["direct_dossier_projection"]
    assert projection["direct_game_ids"] == ["p-1"]
    assert projection["recursive_transfer_excluded"] is True
    assert "transferred_hypotheses" not in projection
    assert "sibling_context_assessment" not in projection
    bargaining_pointer = json.loads((root / "opponents" / opponent_key / "bargaining" / "current.json").read_text(encoding="utf-8"))
    bargaining = json.loads((root / bargaining_pointer["path"]).read_text(encoding="utf-8"))
    sibling_receipt = next(item for item in bargaining["revision"]["sibling_inputs"] if item["game_family"] == "persuasion")
    assert sibling_receipt["provenance"]["revision_sha256"] == persuasion_pointer["sha256"]


def test_clean_family_rebuild_can_read_sibling_context_from_another_root(tmp_path: Path, monkeypatch: Any) -> None:
    canonical_root = tmp_path / "opponent-dossiers" / "incremental-v3"
    canonical = IncrementalNamedOpponentStore(root=canonical_root, project_root=tmp_path)
    captured: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            packet = json.loads(body)
            captured.append(packet)
            family = packet["transition"]["game_family"]
            game_ids = [game["game_id"] for game in packet["new_game_evidence_batch"]]
            return model_cls.model_validate(_draft("Aster", family, game_ids)), {"call_id": f"sol-{family}"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    canonical.enqueue_source_run(_source_run(tmp_path, {"Aster": [("b-1", "bargaining")]}, name="canonical-bargaining"))
    canonical.process_until_idle(max_workers=1)
    bargaining_pointer = json.loads((canonical_root / "opponents" / named_opponent_id("Aster") / "bargaining" / "current.json").read_text(encoding="utf-8"))
    staging_root = tmp_path / "opponent-dossiers" / "persuasion-rebuild"
    staging = IncrementalNamedOpponentStore(root=staging_root, project_root=tmp_path, sibling_context_root=canonical_root)
    staging.enqueue_source_run(_source_run(tmp_path, {"Aster": [("p-1", "persuasion")]}, name="staged-persuasion"))
    staging.process_until_idle(max_workers=1)
    persuasion_packet = next(packet for packet in captured if packet["transition"]["game_family"] == "persuasion")
    bargaining_input = next(item for item in persuasion_packet["sibling_family_dossiers"] if item["game_family"] == "bargaining")
    assert bargaining_input["availability"] == "incremental-family-dossier"
    assert bargaining_input["provenance"]["revision_sha256"] == bargaining_pointer["sha256"]
    assert not (staging_root / "opponents" / named_opponent_id("Aster") / "bargaining").exists()


def test_first_family_dossier_ignores_obsolete_aggregate_baseline(tmp_path: Path, monkeypatch: Any) -> None:
    baseline_root = tmp_path / "opponent-dossiers"
    root = baseline_root / "incremental-v3"
    opponent_key = named_opponent_id("Aster")
    _write_json(baseline_root / "index.json", {"schema_version": 1, "opponents": {opponent_key: {"current_summary": {"path": "obsolete.json", "sha256": "unused"}}}})
    _write_json(baseline_root / "obsolete.json", {"schema_version": 1, "common_evidence_summary": "This aggregate state must not be inherited."})
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(_source_run(tmp_path, {"Aster": [("b-1", "bargaining")]}))
    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            captured.update(json.loads(body))
            return model_cls.model_validate(_draft("Aster", "bargaining", ["b-1"])), {"call_id": "sol-first"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    store.process_until_idle(max_workers=1)
    assert captured["current_family_dossier"] is None
    assert all(item["availability"] == "none" for item in captured["sibling_family_dossiers"])
    pointer = json.loads((root / "opponents" / opponent_key / "bargaining" / "current.json").read_text(encoding="utf-8"))
    dossier = json.loads((root / pointer["path"]).read_text(encoding="utf-8"))
    assert dossier["revision"]["parent_revision_sha256"] is None
    assert dossier["revision"]["initialization"] == "first-completed-game-of-this-opponent-family"
    assert "obsolete" not in json.dumps(dossier)


def test_exact_game_receipts_are_required(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, _body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            return model_cls.model_validate(_draft("Aster", "bargaining", ["wrong-game"])), {"call_id": "sol-wrong"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    with pytest.raises(RuntimeError, match="exact ordered game batch"):
        store.process_until_idle(max_workers=1)
    assert not (root / "opponents" / named_opponent_id("Aster") / "bargaining" / "current.json").exists()


def test_reader_returns_last_valid_family_revision_without_cross_family_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, _body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            return model_cls.model_validate(_draft("Aster", "bargaining", ["b-1"])), {"call_id": "sol"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    store.process_until_idle(max_workers=1)
    reader = IncrementalNamedDossierReader(root)
    first = reader.view("Aster", "bargaining")
    assert first is not None
    assert reader.view("Aster", "persuasion") is None
    pointer_path = root / "opponents" / named_opponent_id("Aster") / "bargaining" / "current.json"
    original = pointer_path.read_text(encoding="utf-8")
    pointer_path.write_text("{incomplete", encoding="utf-8")
    second = reader.view("Aster", "bargaining")
    pointer_path.write_text(original, encoding="utf-8")
    assert second == first
    assert second["provenance"]["revision_number"] == 1
    assert second["family_synopsis"]["direct_game_count"] == 1
    assert set(second) == {"provenance", "family_synopsis"}
    assert set(second["family_synopsis"]) == {"game_family", "confidence", "confidence_profile", "evidence_basis", "direct_game_count", "live_projection"}
    assert second["family_synopsis"]["live_projection"]["semantics"]["normative_policy_precedent"] is False


def test_reader_projects_prior_self_actions_only_into_opponent_model_of_self(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("b-1", "bargaining")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, _body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            return model_cls.model_validate(_draft("Aster", "bargaining", ["b-1"])), {"call_id": "synthesis"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    store.process_until_idle(max_workers=1)
    pointer_path = root / "opponents" / named_opponent_id("Aster") / "bargaining" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    dossier_path = root / pointer["path"]
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    dossier["stable_direct_tendencies"] = ["Aster accepted equality in 7 of 7 observed responses.", "DeepRMM-01 accepted Aster's lowball in 3 of 3 games."]
    dossier["opponent_model_of_us"] = "Aster may infer that DeepRMM-01 accepts repeated lowballs and try to exploit that apparent threshold."
    _write_json(dossier_path, dossier)
    pointer["sha256"] = hashlib.sha256(dossier_path.read_bytes()).hexdigest()
    _write_json(pointer_path, pointer)
    view = IncrementalNamedDossierReader(root).view("Aster", "bargaining")
    assert view is not None
    projection = view["family_synopsis"]["live_projection"]
    assert "Aster accepted equality" in projection["opponent_behavior"]
    assert "DeepRMM-01 accepted" not in projection["opponent_behavior"]
    assert "try to exploit" in projection["opponent_model_of_self"]
    assert projection["semantics"]["prior_self_actions"] == "opponent-model-evidence-only"
    assert projection["semantics"]["normative_policy_precedent"] is False
    assert projection["semantics"]["opponent_learning_risk"] is True
    assert projection["semantics"]["bargaining_allocation_transfer"].startswith("dimensionless pool shares only")


def test_reader_quarantines_legacy_persuasion_revision(tmp_path: Path, monkeypatch: Any) -> None:
    source = _source_run(tmp_path, {"Aster": [("p-1", "persuasion")]})
    root = tmp_path / "opponent-dossiers" / "incremental-v3"
    store = IncrementalNamedOpponentStore(root=root, project_root=tmp_path)
    store.enqueue_source_run(source)

    class FakeRunner:
        def __init__(self, **_settings: object) -> None:
            pass

        def call_structured(self, _role: str, body: str, model_cls: type[Any], **_settings: object) -> tuple[Any, dict[str, object]]:
            packet = json.loads(body)
            assert packet["state_contract"]["game_semantics"]["version"] == "persuasion-information-v1"
            return model_cls.model_validate(_draft("Aster", "persuasion", ["p-1"])), {"call_id": "sol"}

    monkeypatch.setattr("nommd_arena.glee_incremental_dossier.ArenaCodexRunner", FakeRunner)
    store.process_until_idle(max_workers=1)
    opponent_key = named_opponent_id("Aster")
    pointer_path = root / "opponents" / opponent_key / "persuasion" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    dossier_path = root / pointer["path"]
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    assert dossier["game_semantics_version"] == "persuasion-information-v1"
    assert IncrementalNamedDossierReader(root).view("Aster", "persuasion") is not None
    dossier.pop("game_semantics_version")
    _write_json(dossier_path, dossier)
    pointer["sha256"] = hashlib.sha256(dossier_path.read_bytes()).hexdigest()
    pointer.pop("game_semantics_version", None)
    _write_json(pointer_path, pointer)
    assert IncrementalNamedDossierReader(root).view("Aster", "persuasion") is None
