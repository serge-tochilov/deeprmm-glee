from pathlib import Path
from typing import Any

from nommd_arena.glee_tactics import GlobalTacticLedger


def _ledger() -> GlobalTacticLedger:
    return GlobalTacticLedger(Path(__file__).resolve().parents[1] / "tactics" / "glee-global-tactics.json")


def _game(family: str, phase: str, state: dict[str, Any], action_type: str) -> dict[str, Any]:
    return {"game_id": "route-test", "game_family": family, "your_player": "player_1", "phase": phase, "opponent": {"type": "hidden", "name": None}, "prompt": "rules", "game_state": state, "valid_actions": {"type": action_type, "fields": {}}}


def test_extreme_share_tactic_is_defensive_only() -> None:
    ledger = _ledger()
    offer = _game("bargaining", "offer", {"current_player": "player_1", "money_to_divide": 100, "round": 1, "history": []}, "offer")
    assert ledger.view(offer) is None
    decision = _game("bargaining", "decision", {"current_player": "player_1", "money_to_divide": 100, "round": 1, "history": [], "last_offer": {"player_1_gain": 5, "player_2_gain": 95, "proposer": "player_2"}}, "decision")
    view = ledger.view(decision)
    assert view is not None
    assert [entry["id"] for entry in view["entries"]] == ["bargaining-token-share-probe-v1"]
    assert "do not originate" in view["entries"][0]["avoid"]


def test_negotiation_tactic_appears_only_after_a_price_pair_repeats() -> None:
    ledger = _ledger()
    state = {
        "history": [
            {"offer": {"from_player": "player_1", "price": 100}},
            {"offer": {"from_player": "player_2", "price": 50}},
            {"offer": {"from_player": "player_1", "price": 100}},
        ],
        "last_offer": {"from_player": "player_2", "price": 50},
    }
    view = ledger.view(_game("negotiation", "decision", state, "decision"))
    assert view is not None
    assert view["entries"] == [
        {
            "id": "negotiation-movement-topology-v1",
            "observed": "The latest 4 alternating offers repeat the same pair (100, 50).",
            "do": "Choose a deliberate bridge, justified hold with an escape condition, or walkaway; make the choice from surplus, horizon, and reciprocity.",
            "avoid": "Do not replay the same counter by default or keep conceding against a fixed opponent anchor.",
        }
    ]
    state["last_offer"] = {"from_player": "player_2", "price": 55}
    assert ledger.view(_game("negotiation", "decision", state, "decision")) is None


def test_terminal_persuasion_tactic_requires_late_two_class_compliance() -> None:
    ledger = _ledger()
    history = [
        {"seller_message": "yes", "quality": "high", "buyer_decision": "yes", "bought": True},
        {"seller_message": "I recommend this high-quality product.", "quality": "high", "buyer_decision": "yes", "bought": True},
        {"seller_message": "no", "quality": "low", "buyer_decision": "no", "bought": False},
    ]
    seller_state = {"round": 16, "total_rounds": 20, "history": history, "current_quality": "low"}
    seller_view = ledger.view(_game("persuasion", "seller_recommendation", seller_state, "seller_recommendation"))
    assert seller_view is not None
    assert seller_view["entries"][0]["id"] == "persuasion-terminal-reputation-cashout-v1"
    assert "2 followed accurate positives" in seller_view["entries"][0]["observed"]
    assert seller_view["entries"][0]["do"].startswith("Treat cash-out")
    weak_state = {**seller_state, "history": history[:2]}
    assert ledger.view(_game("persuasion", "seller_recommendation", weak_state, "seller_recommendation")) is None
    early_state = {**seller_state, "round": 15}
    assert ledger.view(_game("persuasion", "seller_recommendation", early_state, "seller_recommendation")) is None


def test_terminal_persuasion_buyer_defense_is_late_and_signal_specific() -> None:
    ledger = _ledger()
    history = [
        {"seller_message": "yes", "quality": "high", "bought": True},
        {"seller_message": "yes", "quality": "high", "bought": True},
        {"seller_message": "no", "quality": "low", "bought": False},
    ]
    state = {"round": 20, "total_rounds": 20, "history": history, "seller_message": "yes"}
    view = ledger.view(_game("persuasion", "buyer_decision", state, "buyer_decision"))
    assert view is not None
    assert view["entries"][0]["do"].startswith("Remove any reputation bonus")
    assert ledger.view(_game("persuasion", "buyer_decision", {**state, "seller_message": "no"}, "buyer_decision")) is None
