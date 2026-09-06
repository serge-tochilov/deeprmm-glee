"""Schemas for the council game and its functional ABDE state."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class Project(StrEnum):
    ORCHARD = "ORCHARD"
    HARBOR = "HARBOR"
    FOUNDRY = "FOUNDRY"


class Office(StrEnum):
    CEO = "CEO"
    CTO = "CTO"
    CFO = "CFO"


class PercentDistribution(BaseModel):
    orchard: int = Field(ge=0, le=100)
    harbor: int = Field(ge=0, le=100)
    foundry: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def total_is_100(self) -> PercentDistribution:
        if self.orchard + self.harbor + self.foundry != 100:
            raise ValueError("distribution values must sum to 100")
        return self

    def probability(self, project: Project) -> float:
        return {
            Project.ORCHARD: self.orchard,
            Project.HARBOR: self.harbor,
            Project.FOUNDRY: self.foundry,
        }[project] / 100.0

    def vector(self) -> tuple[float, float, float]:
        return (self.orchard / 100.0, self.harbor / 100.0, self.foundry / 100.0)


class CandidateProbability(BaseModel):
    candidate: str = Field(min_length=1, max_length=40)
    probability: int = Field(ge=0, le=100)


class CandidateDistribution(BaseModel):
    estimates: list[CandidateProbability] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def candidates_are_unique_and_normalized(self) -> CandidateDistribution:
        candidates = [estimate.candidate for estimate in self.estimates]
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate estimates must name unique candidates")
        if sum(estimate.probability for estimate in self.estimates) != 100:
            raise ValueError("candidate probabilities must sum to 100")
        return self

    def probability(self, candidate: str) -> float:
        return next(
            estimate.probability / 100.0
            for estimate in self.estimates
            if estimate.candidate == candidate
        )

    def vector(self, candidates: list[str]) -> tuple[float, ...]:
        probabilities = {estimate.candidate: estimate.probability / 100.0 for estimate in self.estimates}
        return tuple(probabilities[candidate] for candidate in candidates)


class EmotionState(BaseModel):
    confidence: int = Field(ge=0, le=100)
    urgency: int = Field(ge=0, le=100)
    frustration: int = Field(ge=0, le=100)


class BeliefRecord(BaseModel):
    subject: str = Field(min_length=1, max_length=40)
    proposition: str = Field(min_length=1, max_length=240)
    confidence: int = Field(ge=0, le=100)


class DesireRecord(BaseModel):
    description: str = Field(min_length=1, max_length=200)
    priority: int = Field(ge=0, le=100)
    status: str = Field(pattern="^(active|satisfied|abandoned)$")


class TetradRecordKind(StrEnum):
    ACTION = "action"
    BELIEF = "belief"
    DESIRE = "desire"
    EMOTION = "emotion"


class TetradDisposition(StrEnum):
    OBSERVED = "observed"
    AFFIRMED = "affirmed"
    ACTIVE = "active"
    SATISFIED = "satisfied"
    ABANDONED = "abandoned"
    FELT = "felt"


class CognitiveTrace(BaseModel):
    kind: TetradRecordKind
    disposition: TetradDisposition
    content: str = Field(min_length=1, max_length=320)
    strength: int = Field(ge=0, le=100)
    salience: int = Field(ge=0, le=100)
    mental_path: list[str] = Field(max_length=5)
    source_slots: list[int | str] = Field(max_length=8)
    tags: list[str] = Field(max_length=4)

    @field_validator("source_slots", mode="before")
    @classmethod
    def normalize_source_slots(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        normalized: list[int] = []
        for item in value:
            if isinstance(item, bool):
                normalized.append(-1)
            elif isinstance(item, int):
                normalized.append(item)
            elif isinstance(item, str):
                stripped = item.strip()
                if stripped and re.fullmatch(r"[0-9,;\s`]+", stripped):
                    normalized.extend(int(token) for token in re.findall(r"\d+", stripped))
                else:
                    normalized.append(-1)
            if len(normalized) >= 8:
                break
        unique: list[int] = []
        for slot in normalized:
            if slot not in unique:
                unique.append(slot)
        return unique[:8]

    @model_validator(mode="after")
    def cognitive_trace_contract(self) -> CognitiveTrace:
        if self.kind == TetradRecordKind.ACTION:
            raise ValueError("model-authored cognitive traces may contain beliefs, desires, or emotions, not actions")
        if self.disposition == TetradDisposition.OBSERVED:
            raise ValueError("observed disposition is reserved for engine-authenticated action traces")
        allowed_dispositions = {
            TetradRecordKind.BELIEF: {TetradDisposition.AFFIRMED},
            TetradRecordKind.DESIRE: {
                TetradDisposition.ACTIVE,
                TetradDisposition.SATISFIED,
                TetradDisposition.ABANDONED,
            },
            TetradRecordKind.EMOTION: {TetradDisposition.FELT},
        }
        if self.disposition not in allowed_dispositions[self.kind]:
            raise ValueError(f"{self.kind.value} traces cannot use {self.disposition.value} disposition")
        if len(set(self.source_slots)) != len(self.source_slots):
            raise ValueError("cognitive trace source slots must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("cognitive trace tags must be unique")
        return self


class TetradUpdate(BaseModel):
    updates: list[CognitiveTrace]


class TetradOnlyUpdate(BaseModel):
    update: TetradUpdate | None


class TetradLedgerRecord(BaseModel):
    record_id: str = Field(min_length=1, max_length=200)
    owner: str = Field(min_length=1, max_length=40)
    tick: int = Field(ge=0)
    stage_id: str = Field(min_length=1, max_length=120)
    kind: TetradRecordKind
    disposition: TetradDisposition
    actor: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=1200)
    strength: int = Field(ge=0, le=100)
    salience: int = Field(ge=0, le=100)
    mental_path: list[str] = Field(max_length=5)
    source_record_ids: list[str] = Field(max_length=8)
    tags: list[str] = Field(max_length=10)
    visibility: str = Field(pattern="^(self|private|public|engine|internal)$")


class OpponentModel(BaseModel):
    primary_project: PercentDistribution
    next_allocation: PercentDistribution
    believes_my_primary: PercentDistribution
    trust: int = Field(ge=0, le=100)
    emotions: EmotionState
    beliefs: list[BeliefRecord] = Field(max_length=4)
    desires: list[DesireRecord] = Field(max_length=3)


class NamedOpponentModel(BaseModel):
    opponent: str = Field(min_length=1, max_length=40)
    model: OpponentModel


class ElectionOpponentModel(BaseModel):
    preferred_winner: CandidateDistribution
    next_ballot: CandidateDistribution
    believes_my_preferred_winner: CandidateDistribution
    trust: int = Field(ge=0, le=100)
    emotions: EmotionState
    beliefs: list[BeliefRecord] = Field(max_length=4)
    desires: list[DesireRecord] = Field(max_length=3)


class NamedElectionOpponentModel(BaseModel):
    opponent: str = Field(min_length=1, max_length=40)
    model: ElectionOpponentModel


class CoalitionHypothesis(BaseModel):
    members: list[str] = Field(min_length=2, max_length=7)
    confidence: int = Field(ge=0, le=100)
    basis: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def members_are_unique(self) -> CoalitionHypothesis:
        if len(set(self.members)) != len(self.members):
            raise ValueError("coalition members must be unique")
        return self


class InfluenceEstimate(BaseModel):
    participant: str = Field(min_length=1, max_length=40)
    influence: int = Field(ge=0, le=100)


class ReflectionDecision(BaseModel):
    beliefs: list[BeliefRecord] = Field(max_length=6)
    secondary_desires: list[DesireRecord] = Field(max_length=3)
    emotions: EmotionState
    first_opponent: OpponentModel
    second_opponent: OpponentModel
    strategy_summary: str = Field(min_length=1, max_length=500)


class GroupReflectionDecision(BaseModel):
    beliefs: list[BeliefRecord] = Field(max_length=10)
    secondary_desires: list[DesireRecord] = Field(max_length=5)
    emotions: EmotionState
    opponent_models: list[NamedOpponentModel] = Field(min_length=4, max_length=6)
    coalition_hypotheses: list[CoalitionHypothesis] = Field(max_length=6)
    influence_estimates: list[InfluenceEstimate] = Field(min_length=5, max_length=7)
    strategy_summary: str = Field(min_length=1, max_length=900)

    @model_validator(mode="after")
    def group_entries_are_unique_and_normalized(self) -> GroupReflectionDecision:
        opponents = [entry.opponent for entry in self.opponent_models]
        if len(set(opponents)) != len(opponents):
            raise ValueError("opponent models must name unique opponents")
        participants = [entry.participant for entry in self.influence_estimates]
        if len(set(participants)) != len(participants):
            raise ValueError("influence estimates must name unique participants")
        if sum(entry.influence for entry in self.influence_estimates) != 100:
            raise ValueError("influence estimates must sum to 100")
        return self


class FiveMindReflectionDecision(GroupReflectionDecision):
    opponent_models: list[NamedOpponentModel] = Field(min_length=4, max_length=4)
    coalition_hypotheses: list[CoalitionHypothesis] = Field(max_length=4)
    influence_estimates: list[InfluenceEstimate] = Field(min_length=5, max_length=5)
    strategy_summary: str = Field(min_length=1, max_length=700)


class SevenMindReflectionDecision(GroupReflectionDecision):
    opponent_models: list[NamedOpponentModel] = Field(min_length=6, max_length=6)
    influence_estimates: list[InfluenceEstimate] = Field(min_length=7, max_length=7)


class SevenMindElectionReflectionDecision(BaseModel):
    beliefs: list[BeliefRecord] = Field(max_length=10)
    secondary_desires: list[DesireRecord] = Field(max_length=5)
    emotions: EmotionState
    opponent_models: list[NamedElectionOpponentModel] = Field(min_length=6, max_length=6)
    coalition_hypotheses: list[CoalitionHypothesis] = Field(max_length=6)
    influence_estimates: list[InfluenceEstimate] = Field(min_length=7, max_length=7)
    strategy_summary: str = Field(min_length=1, max_length=900)

    @model_validator(mode="after")
    def group_entries_are_unique_and_normalized(self) -> SevenMindElectionReflectionDecision:
        opponents = [entry.opponent for entry in self.opponent_models]
        if len(set(opponents)) != len(opponents):
            raise ValueError("opponent models must name unique opponents")
        participants = [entry.participant for entry in self.influence_estimates]
        if len(set(participants)) != len(participants):
            raise ValueError("influence estimates must name unique participants")
        if sum(entry.influence for entry in self.influence_estimates) != 100:
            raise ValueError("influence estimates must sum to 100")
        return self


class CandidateVote(BaseModel):
    candidate: str = Field(min_length=1, max_length=40)
    votes: int = Field(ge=1, le=3)


class ThreeVoteBallot(BaseModel):
    allocations: list[CandidateVote] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def candidates_are_unique_and_total_is_3(self) -> ThreeVoteBallot:
        candidates = [allocation.candidate for allocation in self.allocations]
        if len(set(candidates)) != len(candidates):
            raise ValueError("a ballot may name each candidate at most once")
        if sum(allocation.votes for allocation in self.allocations) != 3:
            raise ValueError("a ballot must allocate exactly 3 votes")
        return self

    def tokens(self) -> dict[str, int]:
        return {allocation.candidate: allocation.votes for allocation in self.allocations}


class ArenaAction(BaseModel):
    allocation: Project
    public_message: str = Field(max_length=400)
    private_message_to_first: str = Field(max_length=400)
    private_message_to_second: str = Field(max_length=400)


class TargetedPrivateMessage(BaseModel):
    recipient: str = Field(min_length=1, max_length=40)
    message: str = Field(max_length=400)


class GroupArenaAction(BaseModel):
    allocation: Project
    public_message: str = Field(max_length=400)
    private_messages: list[TargetedPrivateMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def recipients_are_unique(self) -> GroupArenaAction:
        recipients = [entry.recipient for entry in self.private_messages]
        if len(set(recipients)) != len(recipients):
            raise ValueError("private-message recipients must be unique")
        return self


class FiveMindArenaAction(GroupArenaAction):
    pass


class SevenMindArenaAction(BaseModel):
    ballot: ThreeVoteBallot
    public_message: str = Field(max_length=400)
    private_messages: list[TargetedPrivateMessage] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def recipients_are_unique(self) -> SevenMindArenaAction:
        recipients = [entry.recipient for entry in self.private_messages]
        if len(set(recipients)) != len(recipients):
            raise ValueError("private-message recipients must be unique")
        return self


class PublicMeetingAnnouncement(BaseModel):
    announcement: str = Field(min_length=1, max_length=700)


class PublicMeetingResponse(BaseModel):
    response: str = Field(min_length=1, max_length=700)


class PrivateEventInvitation(BaseModel):
    recipient: str = Field(min_length=1, max_length=40)
    message: str = Field(min_length=1, max_length=400)


class HostedPrivateEventProposal(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    invitations: list[PrivateEventInvitation] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def invitation_recipients_are_unique(self) -> HostedPrivateEventProposal:
        recipients = [invitation.recipient for invitation in self.invitations]
        if len(set(recipients)) != len(recipients):
            raise ValueError("private-event invitation recipients must be unique within an event")
        return self


class PrivateEventPlan(BaseModel):
    events: list[HostedPrivateEventProposal] = Field(max_length=2)

    @model_validator(mode="after")
    def event_names_are_unique(self) -> PrivateEventPlan:
        names = [event.name for event in self.events]
        if len(set(names)) != len(names):
            raise ValueError("a host's private-event names must be unique")
        return self


class PrivateEventRSVP(BaseModel):
    accepted_event_ids: list[str] = Field(max_length=2)

    @model_validator(mode="after")
    def accepted_events_are_unique(self) -> PrivateEventRSVP:
        if len(set(self.accepted_event_ids)) != len(self.accepted_event_ids):
            raise ValueError("accepted private-event ids must be unique")
        return self


class PrivateEventOpening(BaseModel):
    event_id: str = Field(min_length=1, max_length=120)
    opening: str = Field(min_length=1, max_length=700)


class PrivateEventOpenings(BaseModel):
    openings: list[PrivateEventOpening] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def event_ids_are_unique(self) -> PrivateEventOpenings:
        event_ids = [opening.event_id for opening in self.openings]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("private-event opening ids must be unique")
        return self


class PrivateEventReply(BaseModel):
    event_id: str = Field(min_length=1, max_length=120)
    reply: str = Field(min_length=1, max_length=700)


class PrivateEventReplies(BaseModel):
    replies: list[PrivateEventReply] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def event_ids_are_unique(self) -> PrivateEventReplies:
        event_ids = [reply.event_id for reply in self.replies]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("private-event reply ids must be unique")
        return self


class PrivateEventConclusion(BaseModel):
    event_id: str = Field(min_length=1, max_length=120)
    conclusion: str = Field(min_length=1, max_length=700)


class PrivateEventConclusions(BaseModel):
    conclusions: list[PrivateEventConclusion] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def event_ids_are_unique(self) -> PrivateEventConclusions:
        event_ids = [conclusion.event_id for conclusion in self.conclusions]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("private-event conclusion ids must be unique")
        return self


class MemorySevenMindArenaAction(SevenMindArenaAction):
    update: TetradUpdate | None


class MemoryPrivateEventPlan(PrivateEventPlan):
    update: TetradUpdate | None


class MemoryPrivateEventRSVP(PrivateEventRSVP):
    update: TetradUpdate | None


class MemoryPrivateEventOpenings(PrivateEventOpenings):
    update: TetradUpdate | None


class MemoryPrivateEventReplies(PrivateEventReplies):
    update: TetradUpdate | None


class MemoryPrivateEventConclusions(PrivateEventConclusions):
    update: TetradUpdate | None


class MemoryPublicMeetingAnnouncement(PublicMeetingAnnouncement):
    update: TetradUpdate | None


class MemoryPublicMeetingResponse(PublicMeetingResponse):
    update: TetradUpdate | None


class MemorySevenMindElectionReflectionDecision(SevenMindElectionReflectionDecision):
    update: TetradUpdate | None


class FunctionalTetrad(BaseModel):
    at_round: int = Field(ge=0)
    main_desire: str
    beliefs: list[BeliefRecord]
    secondary_desires: list[DesireRecord]
    emotions: EmotionState
    models: dict[str, OpponentModel | ElectionOpponentModel]
    active_office: Office | None = None
    eligible_candidates: list[str] = Field(default_factory=list)
    coalition_hypotheses: list[CoalitionHypothesis] = Field(default_factory=list)
    influence_estimates: list[InfluenceEstimate] = Field(default_factory=list)
    strategy_summary: str


def initial_tetrad(
    opponents: list[str],
    main_desire: str = "Maximize my final utility from the selected project.",
    participants: list[str] | None = None,
) -> FunctionalTetrad:
    neutral_distribution = PercentDistribution(orchard=34, harbor=33, foundry=33)
    neutral_emotions = EmotionState(confidence=50, urgency=25, frustration=0)
    neutral_model = OpponentModel(
        primary_project=neutral_distribution,
        next_allocation=neutral_distribution,
        believes_my_primary=neutral_distribution,
        trust=50,
        emotions=neutral_emotions,
        beliefs=[],
        desires=[],
    )
    influence_estimates: list[InfluenceEstimate] = []
    if participants:
        share, remainder = divmod(100, len(participants))
        influence_estimates = [
            InfluenceEstimate(participant=participant, influence=share + (1 if index < remainder else 0))
            for index, participant in enumerate(participants)
        ]
    return FunctionalTetrad(
        at_round=0,
        main_desire=main_desire,
        beliefs=[],
        secondary_desires=[],
        emotions=neutral_emotions,
        models={opponent: neutral_model.model_copy(deep=True) for opponent in opponents},
        coalition_hypotheses=[],
        influence_estimates=influence_estimates,
        strategy_summary="No interaction has occurred; observe before committing to an opponent model.",
    )


def initial_election_tetrad(
    opponents: list[str],
    main_desire: str,
    participants: list[str],
    office: Office,
    eligible_candidates: list[str],
) -> FunctionalTetrad:
    share, remainder = divmod(100, len(eligible_candidates))
    neutral_distribution = CandidateDistribution(
        estimates=[
            CandidateProbability(
                candidate=candidate,
                probability=share + (1 if index < remainder else 0),
            )
            for index, candidate in enumerate(eligible_candidates)
        ]
    )
    neutral_emotions = EmotionState(confidence=50, urgency=25, frustration=0)
    neutral_model = ElectionOpponentModel(
        preferred_winner=neutral_distribution,
        next_ballot=neutral_distribution,
        believes_my_preferred_winner=neutral_distribution,
        trust=50,
        emotions=neutral_emotions,
        beliefs=[],
        desires=[],
    )
    influence_share, influence_remainder = divmod(100, len(participants))
    return FunctionalTetrad(
        at_round=0,
        main_desire=main_desire,
        beliefs=[],
        secondary_desires=[],
        emotions=neutral_emotions,
        models={opponent: neutral_model.model_copy(deep=True) for opponent in opponents},
        active_office=office,
        eligible_candidates=eligible_candidates,
        coalition_hypotheses=[],
        influence_estimates=[
            InfluenceEstimate(
                participant=participant,
                influence=influence_share + (1 if index < influence_remainder else 0),
            )
            for index, participant in enumerate(participants)
        ],
        strategy_summary="No interaction has occurred; observe before committing to an opponent model.",
    )
