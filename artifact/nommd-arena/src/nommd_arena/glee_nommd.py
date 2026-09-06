"""NoMMD activation-ledger bridge for live GLEE turns."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .memory import ActivationTetradLedger
from .models import CognitiveTrace, TetradUpdate
from .glee_semantics import model_static_game_context, model_visible_game_state

GLEE_MAIN_DESIRE = "Maximize DeepRMM-01's final ranking by maximizing its own payoff across games while obeying the competition rules."
_LEDGER_CONTENT_LIMIT = 1200


class _NommdResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CognitiveTraceTransport(BaseModel):
    """Keep cognitive semantics non-fatal to the separately valid outward action."""

    model_config = ConfigDict(extra="forbid")
    kind: str
    disposition: str
    content: str
    strength: int
    salience: int
    mental_path: list[str]
    source_slots: list[int | str]
    tags: list[str]


class _TetradUpdateTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: list[_CognitiveTraceTransport]


@lru_cache(maxsize=None)
def nommd_action_model(action_cls: type[BaseModel]) -> type[BaseModel]:
    """Wrap one exact GLEE action schema with a sparse, non-action BDE update."""
    return create_model(
        f"Nommd{action_cls.__name__}",
        __base__=_NommdResponse,
        action=(action_cls, ...),
        update=(_TetradUpdateTransport | None, None),
    )


@lru_cache(maxsize=None)
def glee_action_model(action_cls: type[BaseModel]) -> type[BaseModel]:
    """Wrap one exact GLEE action schema without the retired tetrad carrier."""
    return create_model(
        f"Glee{action_cls.__name__}",
        __base__=_NommdResponse,
        action=(action_cls, ...),
    )


class _PlannerCandidateBase(BaseModel):
    """One exact move candidate plus a bounded statement of its strategic purpose."""

    model_config = ConfigDict(extra="forbid")
    purpose: str = Field(min_length=1, max_length=240)


@lru_cache(maxsize=None)
def nommd_candidate_plan_model(action_cls: type[BaseModel]) -> type[BaseModel]:
    """Wrap one exact GLEE action schema in a bounded candidate-only planner contract."""
    candidate_cls = create_model(
        f"Planned{action_cls.__name__}",
        __base__=_PlannerCandidateBase,
        action=(action_cls, ...),
    )
    return create_model(
        f"Nommd{action_cls.__name__}CandidatePlan",
        __base__=_NommdResponse,
        candidates=(list[candidate_cls], Field(min_length=2, max_length=5)),
    )


class _CandidateSelection(_NommdResponse):
    """Select one immutable planner candidate without reproducing or editing its action."""

    candidate_index: int = Field(ge=0, le=4)


def nommd_candidate_selection_model() -> type[BaseModel]:
    """Return the tetrad-free candidate-index selector schema."""
    return _CandidateSelection


def validate_tetrad_transport(value: _TetradUpdateTransport | None) -> tuple[TetradUpdate | None, list[str]]:
    """Discard malformed cognitive traces without discarding the paired game action."""
    if value is None:
        return None, []
    issues: list[str] = []
    valid: list[CognitiveTrace] = []
    for index, trace in enumerate(value.updates, start=1):
        try:
            valid.append(CognitiveTrace.model_validate(trace.model_dump()))
        except ValidationError as error:
            detail = error.errors(include_url=False, include_input=False)[0]
            location = ".".join(str(item) for item in detail.get("loc", ())) or "trace"
            issues.append(f"trace {index} discarded at {location}: {detail['msg']}")
    return TetradUpdate(updates=valid), issues


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _bounded_evidence(label: str, value: object) -> str:
    """Fit an authenticated projection in the ledger while preserving its receipt identity."""
    content = f"{label}: {_canonical(value)}"
    if len(content) <= _LEDGER_CONTENT_LIMIT:
        return content
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    marker = f" ... [ledger projection truncated; sha256={digest}] ... "
    remaining = _LEDGER_CONTENT_LIMIT - len(marker)
    head = remaining * 3 // 5
    return content[:head] + marker + content[-(remaining - head) :]


def _opponent_label(game: dict[str, Any]) -> str:
    opponent = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
    name = " ".join(str(opponent.get("name") or "").split())
    return name if name and len(name) <= 40 else "HIDDEN_OPPONENT"


def _static_game_context(game: dict[str, Any]) -> dict[str, object]:
    return {**model_static_game_context(game), "opponent": game.get("opponent")}


class GleeNommdMemory:
    """Publish visible GLEE events and expose one persistent mind's bounded working set."""

    def __init__(self, *, root: Path, agent_name: str, retrieval_limit: int, decay: float) -> None:
        self.agent_name = agent_name
        self.ledger = ActivationTetradLedger(
            root=root,
            participants=[agent_name],
            main_desire=GLEE_MAIN_DESIRE,
            retrieval_limit=retrieval_limit,
            decay=decay,
            known_minds=["HIDDEN_OPPONENT"],
        )

    @staticmethod
    def stage_id(game: dict[str, Any]) -> str:
        state = game["game_state"]
        return f"glee:{game['game_id']}:r{int(state.get('round', 0)):03d}:{game['valid_actions']['type']}"

    def observe(self, game: dict[str, Any]) -> str:
        game_id = str(game["game_id"])
        family = str(game["game_family"])
        state = game["game_state"]
        opponent = _opponent_label(game)
        self.ledger.register_minds([opponent])
        self.ledger.publish(
            event_id=f"{game_id}:context",
            stage_id=f"glee:{game_id}:context",
            actor="ENGINE",
            content=_bounded_evidence("Game context", _static_game_context(game)),
            viewers=[self.agent_name],
            visibility="engine",
            tags=["glee", family, "game-context", opponent],
            salience=90,
        )
        for index, entry in enumerate(state.get("history") or [], start=1):
            self.ledger.publish(
                event_id=f"{game_id}:history:{index:03d}",
                stage_id=f"glee:{game_id}:history:{index:03d}",
                actor="ENGINE",
                content=_bounded_evidence(f"Authenticated {family} history entry {index}", entry),
                viewers=[self.agent_name],
                visibility="engine",
                tags=["glee", family, "history", opponent, f"round-{index}"],
                salience=88,
            )
        dynamic = {key: value for key, value in model_visible_game_state(game).items() if key != "history"}
        stage_id = self.stage_id(game)
        self.ledger.publish(
            event_id=f"{game_id}:turn:{int(state.get('round', 0)):03d}:{game['valid_actions']['type']}",
            stage_id=stage_id,
            actor="ENGINE",
            content=_bounded_evidence(f"Current visible {family} turn", dynamic),
            viewers=[self.agent_name],
            visibility="engine",
            tags=["glee", family, "current-turn", opponent, str(game["valid_actions"]["type"]), f"round-{int(state.get('round', 0))}"],
            salience=100,
        )
        return opponent

    def context(self, game: dict[str, Any], opponent: str) -> tuple[dict[str, object], dict[str, object]]:
        state = game["game_state"]
        stage_id = self.stage_id(game)
        cues = ["glee", str(game["game_family"]), opponent, str(game["valid_actions"]["type"]), f"round-{int(state.get('round', 0))}"]
        context, metadata = self.ledger.context(self.agent_name, stage_id=stage_id, cues=cues)
        context["identity"] = self.agent_name
        context["opponent_mind_label"] = opponent
        context["main_desire"] = GLEE_MAIN_DESIRE
        return context, metadata

    def commit(self, update: TetradUpdate | None, metadata: dict[str, object], transport_issues: list[str] | None = None) -> dict[str, object]:
        return self.ledger.commit_update(self.agent_name, update, metadata, external_issues=transport_issues or [])

    def publish_action(self, game: dict[str, Any], action: dict[str, Any], result: dict[str, Any]) -> None:
        state = game["game_state"]
        action_type = str(game["valid_actions"]["type"])
        self.ledger.publish(
            event_id=f"{game['game_id']}:self:{int(state.get('round', 0)):03d}:{action_type}",
            stage_id=f"{self.stage_id(game)}:accepted",
            actor=self.agent_name,
            content=_bounded_evidence("Accepted own action", {"action": action, "server_result": result}),
            viewers=[self.agent_name],
            visibility="engine",
            tags=["glee", str(game["game_family"]), "own-action", action_type, f"round-{int(state.get('round', 0))}"],
            salience=95,
        )

    def publish_result(self, family: str, game_id: str, result: object) -> None:
        self.ledger.publish(
            event_id=f"{game_id}:result",
            stage_id=f"glee:{game_id}:result",
            actor="ENGINE",
            content=_bounded_evidence(f"Final {family} result", result),
            viewers=[self.agent_name],
            visibility="engine",
            tags=["glee", family, "game-result"],
            salience=100,
        )

    def write_summary(self) -> dict[str, object]:
        return self.ledger.write_summary()
