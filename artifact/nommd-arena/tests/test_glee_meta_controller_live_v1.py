from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nommd_arena.glee_dossier import DossierBroker
from nommd_arena.glee_meta_controller_v2 import freeze_fixed_candidates
from nommd_arena.glee_parallel import ParallelGleeRun
from nommd_arena.glee_persuasion_live_v2 import PersuasionTurnForecast, persuasion_decision_facts
from nommd_arena.glee_persuasion_policy_v2_4 import PersuasionSellerPolicyConfig
from nommd_arena.glee_selector_backend import LocalCallableSelectorBackend, local_result
from nommd_arena.glee_tactics import GlobalTacticLedger
from nommd_arena.glee_worker import MetaControllerV15GleeTurnWorker, TurnEnvelope, _meta15_candidate_response_surface


def _eligible_game(family: str) -> dict[str, Any]:
    common = {"game_id": f"meta15-{family}", "game_family": family, "your_player": "player_1", "opponent": {"type": "hidden", "name": None}, "prompt": f"Official {family} rules"}
    if family == "bargaining":
        return {**common, "phase": "offer", "game_state": {"current_player": "player_1", "history": [], "round": 2, "money_to_divide": 100.0, "complete_information": False, "horizon_known": False, "max_rounds": None, "messages_allowed": True}, "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number", "message": "string"}}}
    if family == "negotiation":
        return {**common, "phase": "offer", "game_state": {"current_player": "player_1", "history": [], "round": 2, "complete_information": False, "horizon_known": False, "max_rounds": None, "messages_allowed": True, "player_1_role": "seller", "player_2_role": "buyer", "player_1_value": 20.0, "player_2_value": 80.0}, "valid_actions": {"type": "offer", "fields": {"product_price": "number", "message": "string"}}}
    if family == "persuasion":
        return {**common, "phase": "seller_message", "game_state": {"current_player": "player_1", "player_1_role": "seller", "player_2_role": "buyer", "history": [], "round": 2, "total_rounds": 10, "current_quality": "high", "product_price": 10.0, "p": 0.6, "u": 0.0, "v": 20.0, "seller_message_type": "text", "is_seller_know_cv": True}, "valid_actions": {"type": "seller_message", "fields": {"message": "string"}}}
    raise ValueError(f"unsupported family: {family}")


def _candidate_actions(family: str) -> list[dict[str, object]]:
    if family == "bargaining":
        return [{"alice_gain": 55.0, "bob_gain": 45.0, "message": "A balanced settlement."}, {"alice_gain": 60.0, "bob_gain": 40.0, "message": "This is a firm workable split."}]
    if family == "negotiation":
        return [{"product_price": 50.0, "message": "A direct settlement."}, {"product_price": 60.0, "message": "This price closes the gap."}]
    return [{"message": "I recommend buying this product."}, {"message": "This is high quality and worth buying."}]


def _binary_persuasion_game() -> dict[str, Any]:
    game = _eligible_game("persuasion")
    game["opponent"] = {"type": "agent", "name": "Aster"}
    game["phase"] = "seller_recommendation"
    game["game_state"].update({"current_quality": "low", "p": 0.8, "v": 40.0, "seller_message_type": "binary"})
    game["valid_actions"] = {"type": "seller_recommendation", "fields": {"decision": ["yes", "no"]}}
    return game


def _negotiation_counteroffer_game(*, current_price: float = 100.0) -> dict[str, Any]:
    game = _eligible_game("negotiation")
    game.update({"game_id": f"meta15-negotiation-counteroffer-{current_price}", "phase": "decision"})
    game["game_state"].update(
        {
            "current_player": "player_1",
            "player_1_role": "buyer",
            "player_2_role": "seller",
            "player_1_value": 80.0,
            "last_offer": {"round": 2, "from_player": "player_2", "price": current_price, "message": "Current offer."},
            "round": 2,
            "horizon_known": True,
            "max_rounds": 10,
        }
    )
    game["valid_actions"] = {"type": "decision", "fields": {"decision": "choice", "product_price": "number", "message": "string"}}
    return game


class _ConditionalClient:
    timeout_s = 3.0
    receipt = {"contract": "glee-post-planner-live-conditional-ipc-v1", "authority": "advisory-candidate-response-evidence-only"}

    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.requests: list[dict[str, object]] = []

    def forecast_candidates(self, **values: object) -> dict[str, object]:
        self.order.append("conditional")
        self.requests.append(values)
        if self.fail:
            raise TimeoutError("conditional deadline")
        candidate_set = values["candidate_set"]
        assert [candidate.index for candidate in candidate_set.candidates] == list(range(len(candidate_set.candidates)))
        labels = ["buy", "pass"] if candidate_set.family == "persuasion" else ["accept", "reject", "walkaway"]
        probabilities = [[0.4, 0.6], [0.8, 0.2], [0.65, 0.35], [0.7, 0.3], [0.75, 0.25]] if candidate_set.family == "persuasion" else [[0.4, 0.55, 0.05], [0.75, 0.2, 0.05], [0.7, 0.25, 0.05], [0.65, 0.3, 0.05], [0.6, 0.35, 0.05]]
        rows = [{"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": {"authority": "prospective-shadow-only", "labels": labels, "response_probabilities": probabilities[index], "sequence_probabilities": probabilities[index], "engineered_probabilities": probabilities[index]}} for index, candidate in enumerate(candidate_set.candidates)]
        return {"status": "predicted", "decision_authority": "advisory-candidate-response-evidence-only", "candidate_set_sha256": candidate_set.candidate_set_sha256, "rows": rows}

    def forecast_buyer_continuation(self, **values: object) -> dict[str, object]:
        self.order.append("conditional")
        self.requests.append(values)
        if self.fail:
            raise TimeoutError("buyer-continuation deadline")
        candidate_set = values["candidate_set"]
        probabilities = [[0.72, 0.20, 0.08], [0.45, 0.42, 0.13]]
        rows = [
            {
                "candidate_index": candidate.index,
                "action_sha256": candidate.action_sha256,
                "forecast": {
                    "contract": "glee-persuasion-buyer-continuation-live-v1",
                    "release_id": "test-buyer-continuation",
                    "authority": "advisory-next-seller-signal-evidence-only",
                    "labels": ["signal_positive", "signal_negative", "signal_unknown"],
                    "response_probabilities": probabilities[index],
                    "retrospective_cell_support": 300,
                    "support_warning": None,
                },
            }
            for index, candidate in enumerate(candidate_set.candidates)
        ]
        return {
            "status": "predicted",
            "contract": "glee-persuasion-buyer-continuation-live-v1",
            "service_contract": "glee-post-planner-live-conditional-service-v1",
            "release_id": "test-buyer-continuation",
            "authority": "advisory-next-seller-signal-evidence-only",
            "candidate_set_sha256": candidate_set.candidate_set_sha256,
            "rows": rows,
        }


class _SelfMirrorClient:
    timeout_s = 2.0
    receipt = {"contract": "glee-public-self-mirror-live-ipc-v1", "authority": "bounded-public-expectedness-selector-evidence-only"}

    def __init__(self, order: list[str], *, fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.forecasts: list[dict[str, object]] = []
        self.selections: list[dict[str, object]] = []

    def forecast_candidates(self, **values: object) -> dict[str, object]:
        self.order.append("mirror")
        self.forecasts.append(values)
        if self.fail:
            raise TimeoutError("self-mirror deadline")
        candidate_set = values["candidate_set"]
        count = len(candidate_set.candidates)
        rows = [
            {
                "candidate_index": candidate.index,
                "action_sha256": candidate.action_sha256,
                "forecast": {
                    "ensemble_log_expectedness": -float(index + 1),
                    "relative_expectedness": 1.0 / count,
                    "expectedness_rank": index + 1,
                    "expectedness_percentile": 1.0 if count == 1 else 1.0 - index / (count - 1),
                    "component_log_expectedness": {"1729": -float(index + 1), "2718": -float(index + 1.1)},
                    "component_log_score_stddev": 0.05,
                    "component_top_choice_disagreement": False,
                    "population_prediction": True,
                    "account_prediction": None,
                    "message_wording_scored": False,
                },
            }
            for index, candidate in enumerate(candidate_set.candidates)
        ]
        return {
            "contract": "glee-public-self-mirror-candidate-surface-v1",
            "service_contract": "glee-public-self-mirror-live-service-v1",
            "authority": "bounded-public-expectedness-selector-evidence-only",
            "release_id": "test-self-mirror",
            "turn_id": values["turn_id"],
            "game_id": values["game"]["game_id"],
            "family": candidate_set.family,
            "candidate_set_sha256": candidate_set.candidate_set_sha256,
            "rows": rows,
            "forecast_sha256": "f" * 64,
        }

    def record_selection(self, **values: object) -> dict[str, object]:
        self.order.append("selection")
        self.selections.append(values)
        return {"status": "recorded", "selection": values}


class _StagedRunnerFactory:
    def __init__(self, family: str, order: list[str], *, fail_stage: str | None = None, candidate_actions: list[dict[str, object]] | None = None, selector_index: int = -1) -> None:
        self.family = family
        self.order = order
        self.fail_stage = fail_stage
        self.candidate_actions = candidate_actions
        self.selector_index = selector_index
        self.calls: list[dict[str, object]] = []

    def __call__(self, timeout_s: int) -> Any:
        owner = self

        class Runner:
            def call_structured(self, role: str, body: str, model_cls: type[Any], **settings: object) -> tuple[Any, dict[str, object]]:
                stage = "planner" if "planner" in role else "selector" if "selector" in role else "single"
                owner.order.append(stage)
                owner.calls.append({"stage": stage, "role": role, "payload": json.loads(body), "timeout_s": timeout_s, "settings": settings})
                if owner.fail_stage == stage:
                    raise RuntimeError(f"forced {stage} failure")
                if stage == "planner":
                    candidates = [{"action": action, "purpose": f"candidate {index}"} for index, action in enumerate(owner.candidate_actions or _candidate_actions(owner.family), start=1)]
                    return model_cls.model_validate({"candidates": candidates}), {"call_id": "planner"}
                if stage == "selector":
                    payload = json.loads(body)
                    source_actions = owner.candidate_actions or _candidate_actions(owner.family)
                    desired_action = source_actions[owner.selector_index - 1] if owner.selector_index > 0 else source_actions[-1]
                    matches = [candidate["candidate_index"] for candidate in payload["candidate_set"]["candidates"] if candidate["action"] == desired_action]
                    if len(matches) != 1:
                        raise AssertionError("selector fixture could not resolve the intended canonical action after presentation permutation")
                    index = matches[0]
                    return model_cls.model_validate({"candidate_index": index}), {"call_id": "selector"}
                return model_cls.model_validate({"action": {"decision": "WalkAway"}}), {"call_id": "single"}

        return Runner()


@pytest.mark.parametrize("family", ["bargaining", "negotiation", "persuasion"])
def test_live_meta15_runs_planner_conditional_batch_and_index_only_selector(tmp_path: Path, family: str) -> None:
    game = _eligible_game(family)
    broker = DossierBroker(root=tmp_path / family, agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory(family, order)
    conditional = _ConditionalClient(order)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=conditional, model_timeout_s=108, planner_timeout_s=48, finalization_margin_s=12, minimum_selector_budget_s=12)
    snapshot = broker.observe_turn(game)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=10**20))
    assert order == ["planner", "conditional", "selector"]
    assert decision.selection_branch == "meta15-selector"
    assert decision.proposal == _candidate_actions(family)[1]
    if family == "bargaining":
        assert decision.action["alice_gain"] == 50.0
        assert decision.action["bob_gain"] == 50.0
        assert decision.action["message"] == _candidate_actions(family)[1]["message"]
    else:
        assert decision.action == _candidate_actions(family)[1]
    planner_payload = runner_factory.calls[0]["payload"]
    selector_payload = runner_factory.calls[1]["payload"]
    assert "conditional_opponent_response_surface" not in planner_payload
    assert planner_payload["learned_model_boundary"]["new_learned_model_inputs"] == []
    assert selector_payload["conditional_opponent_response_surface"]["candidate_set_sha256"] == selector_payload["candidate_set"]["candidate_set_sha256"]
    assert selector_payload["selector_authority_contract"]["candidate_standing"] == "symmetric-after-hard-controls"
    assert selector_payload["output_boundary"].startswith("Return only candidate_index")
    selector_receipt = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "meta15-selector")
    assert selector_receipt["authority_contract"] == selector_payload["selector_authority_contract"]
    assert runner_factory.calls[0]["role"] == f"glee_meta_controller_v2_15_planner_{family}"
    assert runner_factory.calls[1]["role"] == f"glee_meta_controller_v2_15_selector_{family}"
    assert conditional.requests[0]["turn_id"] == snapshot.turn_id
    broker.close()


def test_live_meta15_freezes_candidates_before_self_mirror_and_exposes_it_only_to_selector(tmp_path: Path) -> None:
    class NegotiationAdvisor:
        @staticmethod
        def candidate_selection_evidence(action: dict[str, Any]) -> dict[str, object]:
            return {"authority": "candidate-specific-bounded-continuation-evidence", "evaluation": {"price": float(action["product_price"]), "bounded_expected_value": 10.0}}

    game = _eligible_game("negotiation")
    broker = DossierBroker(root=tmp_path / "self-mirror-order", agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("negotiation", order)
    conditional = _ConditionalClient(order)
    mirror = _SelfMirrorClient(order)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=conditional, self_mirror_client=mirror)
    snapshot = broker.observe_turn(game)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=10**20, negotiation_advisor_handle=NegotiationAdvisor()))
    assert order == ["planner", "conditional", "mirror", "selector", "selection"]
    assert mirror.forecasts[0]["candidate_set"] is conditional.requests[0]["candidate_set"]
    planner_payload = runner_factory.calls[0]["payload"]
    selector_payload = runner_factory.calls[1]["payload"]
    assert "public_self_mirror_candidate_surface" not in planner_payload
    assert "public_self_mirror_economic_admissibility" not in planner_payload
    assert selector_payload["public_self_mirror_candidate_surface"]["forecast_sha256"] == "f" * 64
    assert selector_payload["public_self_mirror_economic_admissibility"]["enabled"] is True
    assert mirror.selections[0]["turn_id"] == snapshot.turn_id
    assert mirror.selections[0]["submitted_action"] == decision.action
    broker.close()


def test_live_meta15_self_mirror_failure_is_advisory(tmp_path: Path) -> None:
    game = _eligible_game("negotiation")
    broker = DossierBroker(root=tmp_path / "self-mirror-failure", agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("negotiation", order)
    mirror = _SelfMirrorClient(order, fail=True)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=_ConditionalClient(order), self_mirror_client=mirror)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20))
    assert order == ["planner", "conditional", "mirror", "selector"]
    assert decision.selection_branch == "meta15-selector"
    assert "public_self_mirror_candidate_surface" not in runner_factory.calls[1]["payload"]
    receipt = next(value for value in decision.branch_receipts if value["branch"] == "public-self-mirror")
    assert receipt["status"] == "failed-advisory"
    broker.close()


def test_live_meta15_bridges_baseline_approved_negotiation_reject_and_counteroffer(tmp_path: Path) -> None:
    game = _negotiation_counteroffer_game()
    broker = DossierBroker(root=tmp_path / "negotiation-counteroffer", agent_name="DeepRMM-01")
    order: list[str] = []
    candidates = [
        {"decision": "WalkAway"},
        {"decision": "RejectOffer", "product_price": 70.0, "message": "A workable counteroffer."},
    ]
    runner_factory = _StagedRunnerFactory("negotiation", order, candidate_actions=candidates, selector_index=2)
    conditional = _ConditionalClient(order)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=conditional)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20))
    assert order == ["planner", "conditional", "selector"]
    assert decision.selection_branch == "meta15-selector"
    assert decision.action == candidates[1]
    frozen = conditional.requests[0]["candidate_set"]
    assert len(frozen.candidates) == 2
    assert all(candidate.action["decision"] == "RejectOffer" and "product_price" in candidate.action for candidate in frozen.candidates)
    assert all(candidate.action.get("decision") not in {"AcceptOffer", "WalkAway"} for candidate in frozen.candidates)
    planner_payload = runner_factory.calls[0]["payload"]
    assert planner_payload["conditional_frontier"]["action_type"] == "decision"
    assert planner_payload["deterministic_candidate_seeds"] == [{"decision": "AcceptOffer"}, {"decision": "WalkAway"}, {"decision": "RejectOffer", "product_price": 79.99}]
    selector_actions = [row["action"] for row in runner_factory.calls[-1]["payload"]["candidate_set"]["candidates"]]
    assert {action["decision"] for action in selector_actions} == {"WalkAway", "RejectOffer"}
    broker.close()


def test_live_meta15_projects_one_negotiation_counter_locally_without_invalid_conditional_batch() -> None:
    class NegotiationAdvisor:
        @staticmethod
        def candidate_selection_evidence(action: dict[str, Any]) -> dict[str, object]:
            assert action["decision"] == "RejectOffer"
            return {"authority": "candidate-specific-bounded-continuation-evidence", "evaluation": {"opponent_response": {"weighted": 0.73}}}

    game = _negotiation_counteroffer_game()
    candidate_set = freeze_fixed_candidates(
        game=game,
        candidate_specs=(
            {"action": {"decision": "WalkAway"}, "purpose": "terminal exit"},
            {"action": {"decision": "RejectOffer", "product_price": 70.0}, "purpose": "only counter"},
        ),
    )
    order: list[str] = []
    conditional = _ConditionalClient(order)
    envelope = TurnEnvelope(game=game, snapshot=SimpleNamespace(turn_id="single-counter-turn"), deadline_at_monotonic=10**20, negotiation_advisor_handle=NegotiationAdvisor())
    receipt, surface = _meta15_candidate_response_surface(envelope=envelope, candidate_set=candidate_set, conditional_client=conditional, worker_turn={})
    assert order == []
    assert conditional.requests == []
    assert receipt["counteroffer_forecast_receipt"]["status"] == "locally-projected-single-counter"
    counter_row = next(row for row in surface["rows"] if row["candidate_index"] == 1)
    assert counter_row["forecast"]["response_probabilities"] == [0.73, 0.27, 0.0]
    terminal_row = next(row for row in surface["rows"] if row["candidate_index"] == 0)
    assert terminal_row["forecast"]["target_status"] == "terminal-self-action"


def test_live_meta15_keeps_positive_offer_decision_outside_counteroffer_bridge(tmp_path: Path) -> None:
    class NegotiationAdvisor:
        @staticmethod
        def candidate_selection_evidence(action: dict[str, Any]) -> dict[str, object]:
            if action.get("decision") == "RejectOffer":
                return {"authority": "candidate-specific-bounded-continuation-evidence", "evaluation": {"price": float(action["product_price"]), "own_surplus_if_accepted": 20.0, "bounded_expected_value": 10.0, "opponent_response": {"weighted": 0.5}, "conditional_rejection_continuation": {"bounded_continuation_value": 0.0}}}
            return {"authority": "exact-terminal-candidate-value", "evaluation": {"decision": action.get("decision"), "terminal": True, "own_surplus_if_accepted": 30.0, "bounded_expected_value": 30.0}}

    game = _negotiation_counteroffer_game(current_price=50.0)
    broker = DossierBroker(root=tmp_path / "negotiation-accept", agent_name="DeepRMM-01")
    order: list[str] = []
    conditional = _ConditionalClient(order)
    candidates = [{"decision": "AcceptOffer"}, {"decision": "RejectOffer", "product_price": 60.0, "message": "A bounded counter."}]
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("negotiation", order, candidate_actions=candidates, selector_index=1), conditional_client=conditional)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, negotiation_advisor_handle=NegotiationAdvisor()))
    assert order == ["planner", "selector"]
    assert decision.action == {"decision": "AcceptOffer"}
    assert decision.selection_branch == "meta15-selector"
    assert conditional.requests == []
    conditional_receipt = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "conditional-twin")
    assert conditional_receipt["receipt"]["counteroffer_forecast_receipt"]["status"] == "locally-projected-single-counter"
    broker.close()


@pytest.mark.parametrize(("failure", "expected_order"), [("planner", ["planner"]), ("conditional", ["planner", "conditional"]), ("selector", ["planner", "conditional", "selector"])])
def test_live_meta15_fails_closed_to_guarded_deterministic_action(tmp_path: Path, failure: str, expected_order: list[str]) -> None:
    game = _eligible_game("negotiation")
    broker = DossierBroker(root=tmp_path / failure, agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("negotiation", order, fail_stage=failure if failure != "conditional" else None)
    conditional = _ConditionalClient(order, fail=failure == "conditional")
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=conditional)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20))
    assert order == expected_order
    assert decision.fallback
    assert decision.selection_branch == "deterministic"
    assert decision.action == {"product_price": 20.01}
    assert failure in str(decision.fallback_reason)
    broker.close()


@pytest.mark.parametrize("failure", ["unavailable", "malformed", "gpu-memory"])
def test_live_meta15_local_backend_failure_is_single_attempt_and_family_safe(tmp_path: Path, failure: str) -> None:
    game = _eligible_game("negotiation")
    broker = DossierBroker(root=tmp_path / f"local-{failure}", agent_name="DeepRMM-01")
    order: list[str] = []
    calls = 0

    def selector(wire: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        if failure == "unavailable":
            raise ConnectionError("local selector unavailable")
        if failure == "gpu-memory":
            raise MemoryError("CUDA out of memory")
        return local_result(request=wire, candidate_id="unknown")

    backend = LocalCallableSelectorBackend(selector=selector, backend_id="local-test-v1")
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("negotiation", order), conditional_client=_ConditionalClient(order), selector_backend=backend)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20))
    assert order == ["planner", "conditional"]
    assert calls == 1
    assert decision.fallback
    assert decision.selection_branch == "deterministic"
    assert decision.action == {"product_price": 20.01}
    expected = {"unavailable": "unavailable", "malformed": "does not identify", "gpu-memory": "memoryerror"}[failure]
    assert expected in str(decision.fallback_reason).casefold()
    broker.close()


def test_live_meta15_rejects_a_local_result_after_its_deadline_without_retry(tmp_path: Path) -> None:
    class Clock:
        value = 1000.0

        def __call__(self) -> float:
            return self.value

    clock = Clock()
    game = _eligible_game("negotiation")
    broker = DossierBroker(root=tmp_path / "local-slow", agent_name="DeepRMM-01")
    order: list[str] = []
    calls = 0

    def selector(wire: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        clock.value += float(wire["deadline_budget_s"]) + 0.1
        return local_result(request=wire, candidate_id=str(wire["candidate_ids"][0]))

    backend = LocalCallableSelectorBackend(selector=selector, backend_id="local-slow-v1", clock=clock)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("negotiation", order), conditional_client=_ConditionalClient(order), selector_backend=backend, clock=clock)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20))
    assert calls == 1
    assert decision.fallback
    assert "deadline budget" in str(decision.fallback_reason)
    broker.close()


def test_live_meta15_compares_both_binary_persuasion_signals_and_may_override_v27_anchor(tmp_path: Path) -> None:
    game = _binary_persuasion_game()
    facts = persuasion_decision_facts(game, ())
    assert facts["authority"]["selected_action"] == {"decision": "yes"}
    assert facts["authority"]["action_authority"] == "advisory-anchor"
    forecast = PersuasionTurnForecast(game=game, prompt_context=facts, forecast_receipt={"forecast_id": "binary-anchor"}, seed_sha256="seed", state_revision=1)
    broker = DossierBroker(root=tmp_path / "persuasion-binary", agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("persuasion", order, candidate_actions=[{"decision": "no"}, {"decision": "no"}], selector_index=1)
    conditional = _ConditionalClient(order)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=conditional)
    envelope = TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts, persuasion_advisor_handle=forecast)
    decision = worker.solve(envelope)
    assert order == ["planner", "conditional", "selector"]
    assert decision.selection_branch == "meta15-selector"
    assert decision.action == {"decision": "no"}
    assert runner_factory.calls[0]["payload"]["deterministic_candidate_seeds"] == [{"decision": "yes"}, {"decision": "no"}]
    frozen = conditional.requests[0]["candidate_set"]
    assert [dict(candidate.action) for candidate in frozen.candidates] == [{"decision": "yes"}, {"decision": "no"}]
    broker.close()


def test_live_meta15_persuasion_failure_returns_v27_advisory_anchor(tmp_path: Path) -> None:
    game = _binary_persuasion_game()
    facts = persuasion_decision_facts(game, ())
    forecast = PersuasionTurnForecast(game=game, prompt_context=facts, forecast_receipt={"forecast_id": "binary-fallback"}, seed_sha256="seed", state_revision=1)
    broker = DossierBroker(root=tmp_path / "persuasion-fallback", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order, fail_stage="planner"), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts, persuasion_advisor_handle=forecast))
    assert order == ["planner"]
    assert decision.fallback
    assert decision.action == {"decision": "yes"}
    assert "planner failed" in str(decision.fallback_reason)
    broker.close()


def test_live_meta15_uses_reversed_head_before_one_selector_for_unresolved_nonterminal_buyer_turn(tmp_path: Path) -> None:
    game = _eligible_game("persuasion")
    game.update({"game_id": "persuasion-buyer-continuation", "your_player": "player_2", "phase": "buyer_decision"})
    game["game_state"].update({"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "seller_message": "I recommend buying this product.", "seller_message_type": "text", "p": 0.5, "u": 0.0, "v": 20.0, "product_price": 10.0})
    game["valid_actions"] = {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
    facts = persuasion_decision_facts(game, ())
    assert facts["authority"]["action_authority"] == "advisory-only"
    facts["buyer_economic_control"].update({"expected_surplus": 0.0, "expected_surplus_over_price": 0.0, "buy_economically_admissible": True, "local_expected_value_action": {"decision": "no"}, "common_unit_continuation_lower_bound_over_price": 0.01})
    broker = DossierBroker(root=tmp_path / "buyer-continuation", agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("persuasion", order, candidate_actions=[{"decision": "yes"}, {"decision": "no"}])
    conditional = _ConditionalClient(order)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=conditional)
    snapshot = broker.observe_turn(game)
    decision = worker.solve(TurnEnvelope(game=game, snapshot=snapshot, deadline_at_monotonic=10**20, persuasion_advisor_context=facts))
    assert order == ["conditional", "selector"]
    assert decision.selection_branch == "persuasion-buyer-continuation-selector"
    assert decision.action == {"decision": "no"}
    assert runner_factory.calls[0]["role"] == "glee_persuasion_buyer_continuation_selector"
    payload = runner_factory.calls[0]["payload"]
    assert payload["contract"] == "glee-persuasion-buyer-continuation-selector-payload-v1"
    assert payload["learned_model_boundary"]["candidate_set_is_exhaustive"] is True
    assert payload["buyer_selector_gate"]["selector_eligible"] is True
    assert {row["action"]["decision"] for row in payload["candidate_set"]["candidates"]} == {"yes", "no"}
    assert conditional.requests[0]["turn_id"] == snapshot.turn_id
    broker.close()


@pytest.mark.parametrize("expected_surplus_ratio", [-0.01, -0.025, -0.05])
def test_live_meta15_blocks_material_negative_buyer_purchases_after_recording_local_continuation(tmp_path: Path, expected_surplus_ratio: float) -> None:
    game = _eligible_game("persuasion")
    game.update({"game_id": f"persuasion-material-loss-{expected_surplus_ratio}", "your_player": "player_2", "phase": "buyer_decision"})
    game["game_state"].update({"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "seller_message": "A neutral description.", "seller_message_type": "text", "p": 0.5, "u": 0.0, "v": 200.0 * (1.0 + expected_surplus_ratio), "product_price": 100.0})
    game["valid_actions"] = {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
    facts = persuasion_decision_facts(game, ())
    assert facts["buyer_economic_control"]["expected_surplus_over_price"] == pytest.approx(expected_surplus_ratio)
    broker = DossierBroker(root=tmp_path / f"material-loss-{abs(expected_surplus_ratio)}", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order, candidate_actions=[{"decision": "yes"}, {"decision": "no"}]), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts))
    assert order == ["conditional"]
    assert decision.action == {"decision": "no"}
    assert decision.selection_branch == "persuasion-buyer-economic-local"
    gate = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "persuasion-buyer-selector-gate")["control"]
    assert gate["materially_negative_purchase"] is True
    assert gate["selector_eligible"] is False
    broker.close()


def test_live_meta15_resolves_decisive_positive_buyer_value_locally_after_continuation_receipt(tmp_path: Path) -> None:
    game = _eligible_game("persuasion")
    game.update({"game_id": "persuasion-positive-local", "your_player": "player_2", "phase": "buyer_decision"})
    game["game_state"].update({"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "seller_message": "A neutral description.", "seller_message_type": "text", "p": 0.5, "u": 0.0, "v": 240.0, "product_price": 100.0})
    game["valid_actions"] = {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
    facts = persuasion_decision_facts(game, ())
    broker = DossierBroker(root=tmp_path / "positive-local", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts))
    assert order == ["conditional"]
    assert decision.action == {"decision": "yes"}
    assert decision.selection_branch == "persuasion-buyer-economic-local"
    broker.close()


def test_live_meta15_skips_buyer_selector_when_candidate_conditioned_signal_change_is_small(tmp_path: Path) -> None:
    class LowSensitivity(_ConditionalClient):
        def forecast_buyer_continuation(self, **values: object) -> dict[str, object]:
            receipt = super().forecast_buyer_continuation(**values)
            receipt["rows"][0]["forecast"]["response_probabilities"] = [0.5, 0.4, 0.1]
            receipt["rows"][1]["forecast"]["response_probabilities"] = [0.48, 0.42, 0.1]
            return receipt

    game = _eligible_game("persuasion")
    game.update({"game_id": "persuasion-low-sensitivity", "your_player": "player_2", "phase": "buyer_decision"})
    game["game_state"].update({"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "seller_message": "A neutral description.", "seller_message_type": "text", "p": 0.5, "u": 0.0, "v": 202.0, "product_price": 100.0})
    game["valid_actions"] = {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
    facts = persuasion_decision_facts(game, ())
    broker = DossierBroker(root=tmp_path / "low-sensitivity", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order), conditional_client=LowSensitivity(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts))
    assert order == ["conditional"]
    assert decision.action == {"decision": "no"}
    gate = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "persuasion-buyer-selector-gate")["control"]
    assert gate["continuation_total_variation"] == pytest.approx(0.02)
    assert gate["selector_eligible"] is False
    broker.close()


def test_live_meta15_passes_an_uncertain_tiny_positive_buyer_edge_without_a_common_unit_continuation_bound(tmp_path: Path) -> None:
    game = _eligible_game("persuasion")
    game.update({"game_id": "persuasion-uncertain-positive", "your_player": "player_2", "phase": "buyer_decision"})
    game["game_state"].update({"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "seller_message": "A neutral description.", "seller_message_type": "text", "p": 0.5, "u": 0.0, "v": 202.0, "product_price": 100.0})
    game["valid_actions"] = {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
    facts = persuasion_decision_facts(game, ())
    assert 0.0 < facts["buyer_economic_control"]["expected_surplus_over_price"] <= 0.05
    assert facts["buyer_economic_control"]["lower_expected_surplus_over_price"] < 0.0
    assert facts["buyer_economic_control"]["robust_uncertainty_action"] == {"decision": "no"}
    broker = DossierBroker(root=tmp_path / "uncertain-positive", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order, candidate_actions=[{"decision": "yes"}, {"decision": "no"}], selector_index=1), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts))
    assert order == ["conditional"]
    assert decision.action == {"decision": "no"}
    assert decision.selection_branch == "persuasion-buyer-economic-local"
    gate = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "persuasion-buyer-selector-gate")["control"]
    assert gate["continuation_sensitive"] is True
    assert gate["selector_eligible"] is False
    assert "no lower-bounded continuation value" in gate["reason"]
    broker.close()


def test_live_meta15_keeps_hard_and_terminal_persuasion_buyer_turns_outside_reversed_head(tmp_path: Path) -> None:
    game = _eligible_game("persuasion")
    game.update({"game_id": "persuasion-buyer-hard", "your_player": "player_2", "phase": "buyer_decision"})
    game["game_state"].update({"current_player": "player_2", "player_1_role": "seller", "player_2_role": "buyer", "seller_message": "yes", "seller_message_type": "binary", "p": 0.5, "u": 10.0, "v": 20.0, "product_price": 10.0})
    game["valid_actions"] = {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
    hard = persuasion_decision_facts(game, ())
    assert hard["authority"]["action_authority"] == "categorical"
    broker = DossierBroker(root=tmp_path / "buyer-hard", agent_name="DeepRMM-01")
    hard_order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", hard_order), conditional_client=_ConditionalClient(hard_order))
    hard_decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=hard))
    assert hard_order == []
    assert hard_decision.action == {"decision": "yes"}
    terminal = copy.deepcopy(game)
    terminal["game_id"] = "persuasion-buyer-terminal"
    terminal["game_state"].update({"round": terminal["game_state"]["total_rounds"], "u": 0.0, "v": 15.0})
    terminal_facts = persuasion_decision_facts(terminal, ())
    terminal_order: list[str] = []
    terminal_worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", terminal_order), conditional_client=_ConditionalClient(terminal_order))
    terminal_decision = terminal_worker.solve(TurnEnvelope(game=terminal, snapshot=broker.observe_turn(terminal), deadline_at_monotonic=10**20, persuasion_advisor_context=terminal_facts))
    assert terminal_order == []
    assert terminal_decision.action == {"decision": "no"}
    assert terminal_decision.selection_branch == "persuasion-buyer-terminal-expected-value"
    broker.close()


def test_live_meta15_bargaining_numeric_override_passes_only_with_aligned_multi_model_support(tmp_path: Path) -> None:
    class BargainingAdvisor:
        @staticmethod
        def submission_prediction(action: dict[str, Any]) -> dict[str, object]:
            own_share = float(action["alice_gain"]) / 100.0
            expected = 0.3 + max(0.0, own_share - 0.5)
            return {
                "opponent_acceptance_probability_v2": 0.7,
                "response_expert_probabilities": {"test": 0.7},
                "response_expert_weights": {"test": 1.0},
                "behavioral_offer_evaluation": {"accepted_value": own_share, "rejected_path_value": 0.2, "expected_value": expected},
            }

    game = _eligible_game("bargaining")
    broker = DossierBroker(root=tmp_path / "bargaining-supported", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("bargaining", order), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, bargaining_advisor_handle=BargainingAdvisor()))
    assert decision.action == _candidate_actions("bargaining")[1]
    policy = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "family-selector-policy")
    assert policy["status"] == "passed"
    assert policy["expected_value_gains_over_strongest_baseline_numeric_control"]["blended"] > 0.02
    broker.close()


def test_live_meta15_bargaining_numeric_override_does_not_borrow_wording_gain(tmp_path: Path) -> None:
    class BargainingAdvisor:
        @staticmethod
        def submission_prediction(action: dict[str, Any]) -> dict[str, object]:
            own_share = float(action["alice_gain"]) / 100.0
            selected = own_share > 0.61
            return {
                "opponent_acceptance_probability_v2": 0.31 if selected else 0.33,
                "response_expert_probabilities": {"test": 0.31 if selected else 0.33},
                "response_expert_weights": {"test": 1.0},
                "behavioral_offer_evaluation": {
                    "accepted_value": own_share,
                    "rejected_path_value": 0.338,
                    "expected_value": 0.400 if selected else 0.399,
                },
            }

    class WordingControlConditional(_ConditionalClient):
        def forecast_candidates(self, **values: object) -> dict[str, object]:
            self.order.append("conditional")
            self.requests.append(values)
            candidate_set = values["candidate_set"]
            probabilities = [
                ([0.33, 0.67, 0.0], [0.46, 0.54, 0.0], [0.21, 0.79, 0.0]),
                ([0.429, 0.571, 0.0], [0.673, 0.327, 0.0], [0.204, 0.796, 0.0]),
                ([0.429, 0.571, 0.0], [0.661, 0.339, 0.0], [0.215, 0.785, 0.0]),
            ]
            rows = [
                {
                    "candidate_index": candidate.index,
                    "action_sha256": candidate.action_sha256,
                    "forecast": {
                        "authority": "prospective-shadow-only",
                        "labels": ["accept", "reject", "walkaway"],
                        "response_probabilities": probabilities[index][0],
                        "sequence_probabilities": probabilities[index][1],
                        "engineered_probabilities": probabilities[index][2],
                    },
                }
                for index, candidate in enumerate(candidate_set.candidates)
            ]
            return {"status": "predicted", "rows": rows}

    game = _eligible_game("bargaining")
    broker = DossierBroker(root=tmp_path / "bargaining-wording-control", agent_name="DeepRMM-01")
    order: list[str] = []
    candidates = [
        {"alice_gain": 62.5, "bob_gain": 37.5, "message": "fair share today"},
        {"alice_gain": 60.0, "bob_gain": 40.0, "message": "reasonable compromise now"},
    ]
    advisor_context = {
        "model_version": "bargaining-live-advisor-v2.17",
        "status": "available",
        "behavioral_continuation": {
            "policy_guard": {
                "mode": "bounded-authoritative",
                "minimum_opponent_share": 0.0,
                "maximum_opponent_share": 0.4,
            }
        },
    }
    runner_factory = _StagedRunnerFactory("bargaining", order, candidate_actions=candidates, selector_index=1)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=WordingControlConditional(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, bargaining_advisor_context=advisor_context, bargaining_advisor_handle=BargainingAdvisor()))

    assert decision.proposal == candidates[0]
    assert decision.action == {"alice_gain": 60.0, "bob_gain": 40.0, "message": "fair share today"}
    policy = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "family-selector-policy")
    assert policy["status"] == "numeric-reverted"
    assert policy["baseline_numeric_comparator_candidate_indexes"] == [0, 2]
    assert policy["expected_value_gains_over_strongest_baseline_numeric_control"]["blended"] < 0.02
    broker.close()


def test_live_meta15_negotiation_selector_receives_candidate_specific_bounded_continuation(tmp_path: Path) -> None:
    class NegotiationAdvisor:
        @staticmethod
        def candidate_selection_evidence(action: dict[str, Any]) -> dict[str, object]:
            price = float(action["product_price"])
            return {"authority": "candidate-specific-bounded-continuation-evidence", "evaluation": {"price": price, "own_surplus_if_accepted": price - 20.0, "conditional_rejection_counterproposal": {"q50_price": 45.0}, "bounded_expected_value": price / 2}}

    game = _eligible_game("negotiation")
    broker = DossierBroker(root=tmp_path / "negotiation-evidence", agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("negotiation", order)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, negotiation_advisor_handle=NegotiationAdvisor()))
    assert decision.action == _candidate_actions("negotiation")[1]
    selector_payload = runner_factory.calls[-1]["payload"]
    rows = selector_payload["family_candidate_decision_evidence"]["rows"]
    assert len(rows) == 3
    candidates = selector_payload["candidate_set"]["candidates"]
    price_60_index = next(candidate["candidate_index"] for candidate in candidates if candidate["action"].get("product_price") == 60.0)
    assert next(row for row in rows if row["candidate_index"] == price_60_index)["evidence"]["evaluation"]["bounded_expected_value"] == 30.0
    broker.close()


def test_live_meta15_hidden_one_round_seller_preserves_markup_fallback_but_selects_from_distinct_prices(tmp_path: Path) -> None:
    class NegotiationAdvisor:
        @staticmethod
        def guard_action(action: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
            return dict(action), []

        @staticmethod
        def candidate_selection_evidence(action: dict[str, Any]) -> dict[str, object]:
            price = float(action["product_price"])
            return {"authority": "candidate-specific-bounded-continuation-evidence", "evaluation": {"price": price, "own_surplus_if_accepted": price - 80.0, "bounded_expected_value": price - 80.0}}

    game = _eligible_game("negotiation")
    game["game_id"] = "one-round-hidden-meta15"
    game["game_state"].update({"round": 1, "horizon_known": True, "max_rounds": 1, "player_1_value": 80.0})
    context = {"model_version": "negotiation-live-advisor-v2.10", "status": "available", "deterministic_decision_facts": {"one_round_seller_control": {"status": "advisory", "mode": "incomplete-information-calibrated-markup", "target_price": 100.0}}}
    candidates = [{"product_price": 90.0, "message": "A lower closing price."}, {"product_price": 110.0, "message": "A higher terminal ask."}]
    broker = DossierBroker(root=tmp_path / "one-round-hidden", agent_name="DeepRMM-01")
    order: list[str] = []
    runner_factory = _StagedRunnerFactory("negotiation", order, candidate_actions=candidates, selector_index=1)
    worker = MetaControllerV15GleeTurnWorker(runner_factory=runner_factory, conditional_client=_ConditionalClient(order))
    envelope = TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, negotiation_advisor_context=context, negotiation_advisor_handle=NegotiationAdvisor())
    fallback = worker.fallback(envelope, "test fallback")
    assert fallback.action == {"product_price": 100.0}
    decision = worker.solve(envelope)
    assert decision.action == candidates[0]
    assert runner_factory.calls[0]["payload"]["deterministic_candidate_seeds"] == [{"product_price": 100.0}]
    candidate_prices = {candidate["action"]["product_price"] for candidate in runner_factory.calls[-1]["payload"]["candidate_set"]["candidates"]}
    assert candidate_prices == {90.0, 100.0, 110.0}
    broker.close()


def test_live_meta15_persuasion_weak_opposite_polarity_is_reverted_to_anchor(tmp_path: Path) -> None:
    class WeakConditional(_ConditionalClient):
        def forecast_candidates(self, **values: object) -> dict[str, object]:
            self.order.append("conditional")
            candidate_set = values["candidate_set"]
            probabilities = [[0.4, 0.6], [0.5, 0.5]]
            rows = [{"candidate_index": candidate.index, "action_sha256": candidate.action_sha256, "forecast": {"authority": "prospective-shadow-only", "labels": ["buy", "pass"], "response_probabilities": probabilities[index], "sequence_probabilities": probabilities[index], "engineered_probabilities": probabilities[index]}} for index, candidate in enumerate(candidate_set.candidates)]
            return {"status": "predicted", "rows": rows}

    game = _binary_persuasion_game()
    facts = persuasion_decision_facts(game, ())
    forecast = PersuasionTurnForecast(game=game, prompt_context=facts, forecast_receipt={"forecast_id": "weak-polarity"}, seed_sha256="seed", state_revision=1)
    broker = DossierBroker(root=tmp_path / "persuasion-weak", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order, candidate_actions=[{"decision": "no"}, {"decision": "no"}], selector_index=1), conditional_client=WeakConditional(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts, persuasion_advisor_handle=forecast))
    assert decision.proposal == {"decision": "no"}
    assert decision.action == {"decision": "yes"}
    assert "persuasion_meta15_polarity_override_margin" in decision.deterministic_safeguards
    policy = next(receipt for receipt in decision.branch_receipts if receipt["branch"] == "family-selector-policy")
    assert policy["status"] == "polarity-reverted"
    broker.close()


def test_live_meta15_terminal_binary_persuasion_anchor_bypasses_both_terra_calls(tmp_path: Path) -> None:
    game = _binary_persuasion_game()
    game["game_state"]["round"] = game["game_state"]["total_rounds"]
    facts = persuasion_decision_facts(game, ())
    forecast = PersuasionTurnForecast(game=game, prompt_context=facts, forecast_receipt={"forecast_id": "terminal-anchor"}, seed_sha256="seed", state_revision=1)
    broker = DossierBroker(root=tmp_path / "persuasion-terminal", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts, persuasion_advisor_handle=forecast))
    assert order == []
    assert decision.selection_branch == "persuasion-v2.7-terminal-binary-anchor"
    assert decision.action == facts["authority"]["selected_action"]
    broker.close()


def test_live_meta15_no_response_persuasion_authority_bypasses_both_terra_calls(tmp_path: Path) -> None:
    game = _binary_persuasion_game()
    game["game_state"].update({"round": 5, "history": [{"round": round_number, "seller_message": "yes", "buyer_decision": "no", "bought": False, "quality": "high", "buyer_payoff": 0, "seller_payoff": 0, "response_time_ms": 1_000} for round_number in range(1, 5)]})
    facts = persuasion_decision_facts(game, (), seller_policy_config=PersuasionSellerPolicyConfig(no_response_routing="deterministic-quality-consistent"))
    assert facts["authority"]["action_authority"] == "bounded-authoritative"
    forecast = PersuasionTurnForecast(game=game, prompt_context=facts, forecast_receipt={"forecast_id": "no-response"}, seed_sha256="seed", state_revision=1)
    broker = DossierBroker(root=tmp_path / "persuasion-no-response", agent_name="DeepRMM-01")
    order: list[str] = []
    worker = MetaControllerV15GleeTurnWorker(runner_factory=_StagedRunnerFactory("persuasion", order), conditional_client=_ConditionalClient(order))
    decision = worker.solve(TurnEnvelope(game=game, snapshot=broker.observe_turn(game), deadline_at_monotonic=10**20, persuasion_advisor_context=facts, persuasion_advisor_handle=forecast))
    assert order == []
    assert decision.selection_branch == "persuasion-v2.8-no-response-authority"
    assert decision.action == {"decision": "no"}
    broker.close()


def test_parallel_meta15_wires_one_shared_conditional_client_and_records_contract(tmp_path: Path) -> None:
    class Client:
        @staticmethod
        def stats() -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

    conditional = _ConditionalClient([])
    project_root = Path(__file__).resolve().parents[1]
    run = ParallelGleeRun(project_root=project_root, run_dir=tmp_path / "run", env_file=None, model="gpt-5.6-terra", effort="high", worker_policy="meta15", model_timeout_s=108, turn_deadline_s=120, emergency_margin_s=12, max_parallel=1, max_games=1, families=("negotiation",), client=Client(), conditional_twin_client=conditional, global_tactic_ledger=GlobalTacticLedger())
    try:
        assert isinstance(run.worker, MetaControllerV15GleeTurnWorker)
        assert run.worker.conditional_client is conditional
        assert run._mode() == "glee-parallel-v36-meta15"
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["worker_policy"] == "meta15"
        assert manifest["meta_controller"]["contract"] == "glee-pre-terra-one-and-half-round-controller-v2.2"
        assert manifest["meta_controller"]["selector_authority_contract"] == "glee-symmetric-candidate-selector-authority-v1"
        assert manifest["meta_controller"]["family_selector_policy_contract"] == "glee-family-selector-policy-v1"
        assert manifest["conditional_twin"]["authority"] == "advisory-candidate-response-evidence-only"
    finally:
        run.broker.close()


def test_parallel_meta15_sol_high_uses_sol_for_selector_and_ineligible_turns(tmp_path: Path) -> None:
    class Client:
        @staticmethod
        def stats() -> dict[str, object]:
            return {"agent_id": "agent-1", "agent_name": "DeepRMM-01", "active_games": 0, "scores": {}}

    conditional = _ConditionalClient([])
    project_root = Path(__file__).resolve().parents[1]
    run = ParallelGleeRun(project_root=project_root, run_dir=tmp_path / "run", env_file=None, model="gpt-5.6-sol", effort="high", worker_policy="meta15", model_timeout_s=108, turn_deadline_s=120, emergency_margin_s=12, max_parallel=1, max_games=1, families=("negotiation",), client=Client(), conditional_twin_client=conditional, global_tactic_ledger=GlobalTacticLedger())
    try:
        assert isinstance(run.worker, MetaControllerV15GleeTurnWorker)
        assert run.worker.selector_backend.manifest_receipt["backend_id"] == "sol-cli-v1-position-permuted"
        assert run.worker.single_call_worker.manifest_chain == [{"branch": "primary", "model": "gpt-5.6-sol", "effort": "high"}]
        manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["meta_controller"]["ineligible_model_chain"] == [{"branch": "primary", "model": "gpt-5.6-sol", "effort": "high"}]
    finally:
        run.broker.close()
