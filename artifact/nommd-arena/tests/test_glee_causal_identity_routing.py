from __future__ import annotations

import math

import pytest

from nommd_arena.glee_behavior_channel_analysis import UNKNOWN_ID
from nommd_arena.glee_causal_identity_routing import ActionPackage, OOV_ACTION_CONTEXT, OOV_ACTION_TOKEN, _action_packages, _collapse_prior, _first_move_record, _mixture_nll, _move_action_observations, _move_action_tokens


def _move(kind: str, value: float, decision: str | None = None) -> dict[str, object]:
    return {"kind": kind, "action_value": value, "decision": decision, "context": {"round_phase": 0.25, "complete_information": True, "horizon_known": True, "messages_allowed": True}}


def test_gallery_collapse_preserves_known_mass_and_accumulates_natural_unknown() -> None:
    collapsed = _collapse_prior({"alpha": 0.4, "beta": 0.3, "sparse": 0.25, UNKNOWN_ID: 0.05}, ("alpha", "beta"))

    assert collapsed == pytest.approx({"alpha": 0.4, "beta": 0.3, UNKNOWN_ID: 0.3})
    assert sum(collapsed.values()) == pytest.approx(1.0)


def test_first_move_snapshot_and_action_evidence_exclude_later_and_terminal_tokens() -> None:
    record = {"game_id": "g", "moves": [_move("proposal", 0.4), _move("response", 0.4, "accept")], "channel_counts": {"moves": 2, "timed_moves": 1}}

    snapshot = _first_move_record(record)
    tokens = _move_action_tokens(snapshot["moves"][0])

    assert snapshot["moves"] == [record["moves"][0]]
    assert snapshot["channel_counts"] == {"moves": 1}
    assert tokens
    assert not any(token.startswith("terminal-style|") for token in tokens)


def test_next_action_observations_predict_choice_given_visible_context() -> None:
    proposal = _move_action_observations(_move("proposal", 0.4))
    response = _move_action_observations(_move("response", 0.4, "accept"))

    assert len(proposal) == 1
    assert proposal[0][0].startswith("action|kind=proposal|")
    assert proposal[0][1].startswith("value=")
    assert len(response) == 1
    assert response[0][0].startswith("decision|kind=response|")
    assert "|value=" in response[0][0]
    assert response[0][1] == "decision=accept"


def test_action_packages_shrink_identity_choices_toward_family_population() -> None:
    records = [
        {"game_id": "a1", "family": "bargaining", "public_player_id": "alpha", "moves": [_move("response", 0.5, "accept")]},
        {"game_id": "a2", "family": "bargaining", "public_player_id": "alpha", "moves": [_move("response", 0.5, "accept")]},
        {"game_id": "b1", "family": "bargaining", "public_player_id": "beta", "moves": [_move("response", 0.5, "reject")]},
        {"game_id": "b2", "family": "bargaining", "public_player_id": "beta", "moves": [_move("response", 0.5, "reject")]},
    ]
    assignments = {str(record["game_id"]): "train" for record in records}
    galleries = {"bargaining": ("alpha", "beta"), "negotiation": (), "persuasion": ()}
    packages = _action_packages(records, assignments, galleries, "calibration", {"bargaining": 1.0, "negotiation": 1.0, "persuasion": 1.0})
    observation = _move_action_observations(_move("response", 0.5, "accept"))[0]
    generic = packages[("bargaining", UNKNOWN_ID)]
    alpha = packages[("bargaining", "alpha")]
    beta = packages[("bargaining", "beta")]

    assert alpha.probability(*observation) > generic.probability(*observation) > beta.probability(*observation)
    assert alpha.nll([observation]) < generic.nll([observation])
    assert math.isfinite(generic.nll([("never-seen-context", "never-seen-outcome")]))


def test_soft_route_mixes_predictive_probabilities_not_losses() -> None:
    observation = [("context", "yes")]
    packages = {
        "alpha": ActionPackage({"context": {"yes": 0.9, OOV_ACTION_TOKEN: 0.1}, OOV_ACTION_CONTEXT: {"yes": 0.5, OOV_ACTION_TOKEN: 0.5}}),
        UNKNOWN_ID: ActionPackage({"context": {"yes": 0.3, OOV_ACTION_TOKEN: 0.7}, OOV_ACTION_CONTEXT: {"yes": 0.5, OOV_ACTION_TOKEN: 0.5}}),
    }

    loss = _mixture_nll(observation, {"alpha": 0.5, UNKNOWN_ID: 0.5}, packages)

    assert loss == pytest.approx(-math.log(0.6))
