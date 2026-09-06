"""Build a collision-safe reference corpus for offline GLEE behavior channels."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .glee_activity_eda import _file_digest
from .glee_bargaining_twin import classify_message_act
from .glee_joint_rating_analysis import _digest
from .glee_negotiation_twin_v2 import classify_negotiation_message, opponent_surplus_share
from .glee_persuasion_twin_v2 import classify_persuasion_signal


BEHAVIOR_CORPUS_CONTRACT = "glee-collision-safe-behavior-corpus-v1"
_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?%?")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|<num>")
_DISCOURSE_PATTERNS = {
    "authority": re.compile(r"\b(?:authority|official|rule|equilibrium|rubinstein|optimal|rational)\b", re.IGNORECASE),
    "explicit-request": re.compile(r"\b(?:accept|agree|take|buy|purchase|choose|reply|counter)\b", re.IGNORECASE),
    "fairness": re.compile(r"\b(?:fair|equal|even|balanced|50\s*/\s*50|half)\b", re.IGNORECASE),
    "finality": re.compile(r"\b(?:final|last|only offer|take it or leave it|no further)\b", re.IGNORECASE),
    "price-value": re.compile(r"\b(?:price|cost|value|worth|payoff|surplus|discount)\b|[$€£]", re.IGNORECASE),
    "promise": re.compile(r"\b(?:promise|will reciprocate|next round|in return|return the favor)\b", re.IGNORECASE),
    "reciprocity": re.compile(r"\b(?:recipro|cooperat|mutual|both|together|good faith)\w*\b", re.IGNORECASE),
    "reservation": re.compile(r"\b(?:reservation|minimum|maximum|floor|ceiling|cannot go|won't go)\b", re.IGNORECASE),
    "threat": re.compile(r"\b(?:or else|punish|walk away|no deal|lose|nothing|never accept)\b", re.IGNORECASE),
    "trust": re.compile(r"\b(?:trust|honest|truth|lie|lied|deceiv|credible)\w*\b", re.IGNORECASE),
    "urgency": re.compile(r"\b(?:now|quick|hurry|immediate|before|time|deadline|shrinking|decreasing)\b", re.IGNORECASE),
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    _atomic_text(path, "".join(_canonical(value) + "\n" for value in values))


def _other_player(player: str) -> str:
    if player == "player_1":
        return "player_2"
    if player == "player_2":
        return "player_1"
    raise ValueError(f"unsupported player identity: {player}")


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _normalized_message(value: object) -> str:
    text = str(value or "").replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return _SPACE.sub(" ", text.strip())


def _feature_hash(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}\x00{value}".encode("utf-8")).hexdigest()[:16]


def _lexical_signature(message: object, *, family_act: str) -> dict[str, object]:
    """Represent language without copying recoverable message prose into the derived corpus."""
    text = _normalized_message(message)
    if not text:
        return {"present": False, "family_act": family_act, "message_sha256": hashlib.sha256(b"").hexdigest(), "style": {"chars": 0, "words": 0}, "discourse_acts": ["silence"], "hashed_lexemes": {}}
    lowered = text.casefold()
    abstracted = _NUMBER.sub("<num>", lowered)
    words = _WORD.findall(abstracted)
    lexemes: Counter[str] = Counter()
    for word in words:
        lexemes[_feature_hash("word", word)] += 1
    for left, right in zip(words, words[1:]):
        lexemes[_feature_hash("bigram", f"{left} {right}")] += 1
    compact_chars = f"^{abstracted}$"
    for width in (3, 4):
        for index in range(max(0, len(compact_chars) - width + 1)):
            lexemes[_feature_hash(f"char-{width}", compact_chars[index : index + width])] += 1
    letters = [char for char in text if char.isalpha()]
    digits = [char for char in text if char.isdigit()]
    style = {
        "chars": len(text),
        "words": len(words),
        "sentences": sum(text.count(mark) for mark in ".!?") or 1,
        "uppercase_ratio": round(sum(char.isupper() for char in letters) / max(1, len(letters)), 6),
        "digit_ratio": round(len(digits) / max(1, len(text)), 6),
        "question_marks": text.count("?"),
        "exclamation_marks": text.count("!"),
        "commas": text.count(","),
        "semicolons": text.count(";") + text.count(":"),
        "currency_marks": sum(text.count(mark) for mark in "$€£"),
        "percent_marks": text.count("%"),
        "decimal_numbers": len(re.findall(r"\d+\.\d+", text)),
        "contractions": len(re.findall(r"\b[a-z]+['’][a-z]+\b", lowered)),
        "opening_sha256": _feature_hash("opening", " ".join(words[:4])) if words else None,
        "ending_sha256": _feature_hash("ending", " ".join(words[-4:])) if words else None,
    }
    acts = sorted(name for name, pattern in _DISCOURSE_PATTERNS.items() if pattern.search(text))
    if family_act and family_act != "none" and family_act not in acts:
        acts.append(family_act)
    return {"present": True, "family_act": family_act, "message_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "style": style, "discourse_acts": sorted(acts) or ["other"], "hashed_lexemes": dict(sorted(lexemes.items()))}


def _round_phase(round_number: int, state: Mapping[str, object]) -> float:
    maximum = state.get("max_rounds")
    if state.get("horizon_known") is True and isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 1:
        return min(1.0, max(0.0, (round_number - 1) / (maximum - 1)))
    total = state.get("total_rounds")
    if isinstance(total, int) and not isinstance(total, bool) and total > 1:
        return min(1.0, max(0.0, (round_number - 1) / (total - 1)))
    return 1.0 - math.exp(-max(0, round_number - 1) / 12.0)


def _base_context(state: Mapping[str, object], *, round_number: int, move_kind: str, opponent_role: str | None = None) -> dict[str, object]:
    return {
        "round": round_number,
        "round_phase": round(_round_phase(round_number, state), 8),
        "move_kind": move_kind,
        "opponent_role": opponent_role,
        "complete_information": state.get("complete_information") is True,
        "horizon_known": state.get("horizon_known") is True,
        "messages_allowed": state.get("messages_allowed") is True or state.get("seller_message_type") in {"binary", "text"},
    }


def _bargaining_moves(game: Mapping[str, Any]) -> list[dict[str, object]]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    pool = max(1.0, abs(_number(state.get("money_to_divide"), 1.0)))
    moves: list[dict[str, object]] = []
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            continue
        offer = raw.get("offer") if isinstance(raw.get("offer"), Mapping) else {}
        proposer = str(raw.get("proposer") or offer.get("proposer") or "")
        round_number = int(raw.get("round") or offer.get("round") or index + 1)
        opponent_gain = _number(offer.get(f"{opponent_player}_gain")) / pool
        if proposer == opponent_player:
            message = str(offer.get("message") or "")
            moves.append({"move_index": len(moves), "actor": "opponent", "kind": "proposal", "action_value": opponent_gain, "decision": None, "response_time_ms": None, "context": _base_context(state, round_number=round_number, move_kind="proposal"), "language": _lexical_signature(message, family_act=classify_message_act(message, messages_allowed=state.get("messages_allowed") is True))})
        elif proposer == our_player:
            decision = str(raw.get("decision") or "").casefold()
            delay = _number(raw.get("response_time_ms"), -1.0)
            moves.append({"move_index": len(moves), "actor": "opponent", "kind": "response", "action_value": opponent_gain, "decision": decision, "response_time_ms": delay if delay >= 0 else None, "context": _base_context(state, round_number=round_number, move_kind="response"), "language": None})
    return moves


def _negotiation_moves(game: Mapping[str, Any]) -> list[dict[str, object]]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    our_role = str(state.get(f"{our_player}_role") or "")
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    our_value = _number(state.get(f"{our_player}_value"), 1.0)
    raw_opponent_value = state.get(f"{opponent_player}_value") if state.get("complete_information") is True else None
    opponent_value = _number(raw_opponent_value) if raw_opponent_value is not None else None
    moves: list[dict[str, object]] = []
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            continue
        offer = raw.get("offer") if isinstance(raw.get("offer"), Mapping) else {}
        proposer = str(offer.get("from_player") or "")
        round_number = int(raw.get("round") or offer.get("round") or index + 1)
        price = _number(offer.get("price"))
        normalized_price = math.tanh(price / max(1.0, abs(our_value)))
        surplus_share = opponent_surplus_share(price, opponent_role=opponent_role, our_role=our_role, our_value=our_value, opponent_value=opponent_value)
        context = _base_context(state, round_number=round_number, move_kind="proposal" if proposer == opponent_player else "response", opponent_role=opponent_role)
        context["opponent_value_known"] = opponent_value is not None
        if proposer == opponent_player:
            message = str(offer.get("message") or "")
            moves.append({"move_index": len(moves), "actor": "opponent", "kind": "proposal", "action_value": surplus_share if surplus_share is not None else normalized_price, "normalized_price": normalized_price, "opponent_surplus_share": surplus_share, "decision": None, "response_time_ms": None, "context": context, "language": _lexical_signature(message, family_act=classify_negotiation_message(message, messages_allowed=state.get("messages_allowed") is True))})
        elif proposer == our_player:
            decision = str(raw.get("decision") or "")
            delay = _number(raw.get("response_time_ms"), -1.0)
            moves.append({"move_index": len(moves), "actor": "opponent", "kind": "response", "action_value": surplus_share if surplus_share is not None else normalized_price, "normalized_price": normalized_price, "opponent_surplus_share": surplus_share, "decision": decision, "response_time_ms": delay if delay >= 0 else None, "context": context, "language": None})
    return moves


def _persuasion_moves(game: Mapping[str, Any]) -> list[dict[str, object]]:
    state = game.get("game_state") if isinstance(game.get("game_state"), Mapping) else {}
    history = state.get("history") if isinstance(state.get("history"), list) else []
    our_player = str(game.get("your_player") or "")
    opponent_player = _other_player(our_player)
    opponent_role = str(state.get(f"{opponent_player}_role") or "")
    channel = str(state.get("seller_message_type") or "text").casefold()
    moves: list[dict[str, object]] = []
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            continue
        round_number = int(raw.get("round") or index + 1)
        message = str(raw.get("seller_message") or "")
        polarity, act, _fingerprint = classify_persuasion_signal(message, channel=channel)
        context = _base_context(state, round_number=round_number, move_kind="signal" if opponent_role == "seller" else "response", opponent_role=opponent_role)
        context["signal_polarity"] = polarity
        if opponent_role == "seller":
            action_value = 1.0 if polarity == "positive" else -1.0 if polarity == "negative" else 0.0
            moves.append({"move_index": len(moves), "actor": "opponent", "kind": "signal", "action_value": action_value, "decision": polarity, "response_time_ms": None, "context": context, "language": _lexical_signature(message, family_act=act)})
        elif opponent_role == "buyer":
            bought = raw.get("bought") is True or str(raw.get("buyer_decision") or "").casefold() == "yes"
            delay = _number(raw.get("response_time_ms"), -1.0)
            moves.append({"move_index": len(moves), "actor": "opponent", "kind": "response", "action_value": 1.0 if bought else 0.0, "decision": "buy" if bought else "pass", "response_time_ms": delay if delay >= 0 else None, "context": context, "language": None})
    return moves


def extract_behavior_moves(game: Mapping[str, Any]) -> list[dict[str, object]]:
    """Extract scale-normalized opponent moves from one visible terminal game."""
    family = str(game.get("game_family") or "")
    if family == "bargaining":
        return _bargaining_moves(game)
    if family == "negotiation":
        return _negotiation_moves(game)
    if family == "persuasion":
        return _persuasion_moves(game)
    raise ValueError(f"unsupported GLEE behavior family: {family}")


def extract_behavior_game(envelope: Mapping[str, object], game: Mapping[str, Any]) -> dict[str, object]:
    """Extract compact opponent-attributed features under one exact temporal public identity."""
    family = str(envelope["family"])
    if str(game.get("game_id") or "") != str(envelope["game_id"]) or str(game.get("game_family") or "") != family:
        raise ValueError("identity envelope and terminal game disagree")
    public_player_id = str(envelope.get("exact_public_player_id") or "")
    if not public_player_id:
        raise ValueError("behavior corpus requires an exact public player ID")
    moves = extract_behavior_moves(game)
    message_opportunities = sum(move["language"] is not None for move in moves)
    nonempty_messages = sum(bool(move["language"] and move["language"]["present"]) for move in moves)
    timed_moves = sum(move["response_time_ms"] is not None for move in moves)
    return {
        "contract": BEHAVIOR_CORPUS_CONTRACT,
        "schema_version": 1,
        "game_id": envelope["game_id"],
        "family": family,
        "public_player_id": public_player_id,
        "display_label": envelope.get("disclosed_label"),
        "resolution_status": envelope.get("resolution_status"),
        "identity_frontiers": {"assignment": envelope.get("assignment_frontier_sequence"), "completion": envelope.get("aligned_completion_frontier_sequence")},
        "started_at": envelope.get("started_at"),
        "completed_at": envelope.get("completed_at"),
        "source": {"archive_path": envelope["source_archive_path"], "archive_sha256": envelope["source_archive_sha256"]},
        "channel_counts": {"moves": len(moves), "timed_moves": timed_moves, "message_opportunities": message_opportunities, "nonempty_messages": nonempty_messages},
        "moves": moves,
    }


class GleeBehaviorCorpusAnalysis:
    """Freeze exact-ID behavior features without mutating live timing or dossier stores."""

    def __init__(self, *, identity_dir: Path, game_archive_root: Path, output_dir: Path) -> None:
        self.identity_dir = identity_dir.resolve()
        self.game_archive_root = game_archive_root.resolve()
        self.output_dir = output_dir.resolve()

    def _load(self) -> tuple[list[dict[str, object]], Mapping[str, object]]:
        manifest_path = self.identity_dir / "manifest.json"
        envelopes_path = self.identity_dir / "game-identity-envelopes.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {}).get("game-identity-envelopes.jsonl", {}).get("sha256")
        if expected and _file_digest(envelopes_path) != expected:
            raise ValueError("identity-envelope artifact hash mismatch")
        envelopes = [json.loads(line) for line in envelopes_path.read_text(encoding="utf-8").splitlines() if line]
        return envelopes, manifest

    @staticmethod
    def _readme(summary: Mapping[str, object]) -> str:
        inventory = summary["inventory"]
        return "\n".join(
            [
                "# GLEE collision-safe behavior corpus v1",
                "",
                f"**Status:** Completed offline Stage 5 corpus through temporal-identity frontier `{summary['frontier_sequence']}`; it starts no matchmaking, sends no model call, changes no live prompt or action, and publishes no dossier or identity route.",
                "",
                "## Result",
                "",
                f"The corpus resolves {inventory['games']:,} known-identity games to exact temporal public IDs and extracts {inventory['moves']:,} opponent-attributed actions, {inventory['timed_moves']:,} exact server response delays, and {inventory['nonempty_messages']:,} nonempty opponent messages from {inventory['message_opportunities']:,} message opportunities. Raw messages and terminal game objects are not copied; language is represented by style counts, discourse acts, stable hashes, and hashed word, bigram, and character features, while every game retains a hash-verified archive reference.",
                "",
                "Bargaining and Negotiation proposals are normalized to opponent share or surplus when visible, with scale-safe price fallback under incomplete information. Responses preserve the offered normalized value, decision, and exact opponent response time. Persuasion separates seller signals from buyer decisions and does not use terminally revealed hidden quality as an online-observer feature.",
                "",
                "## Boundary",
                "",
                "Only `exact_public_player_id` envelopes enter this v1 corpus. Collision sets, probabilistic assignments, hidden games, and normalized-name timing rows remain excluded rather than collapsed into one identity. This is a feature substrate, not an identity classifier; chronological channel evaluation and calibrated `unknown` handling remain the next Stage 5 checkpoint.",
                "",
            ]
        )

    def run(self) -> dict[str, object]:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"behavior-corpus output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        envelopes, identity_manifest = self._load()
        records: list[dict[str, object]] = []
        exclusions: Counter[str] = Counter()
        for envelope in envelopes:
            if not envelope.get("exact_public_player_id"):
                exclusions["no-exact-public-player-id"] += 1
                continue
            path = self.game_archive_root / str(envelope["source_archive_path"])
            try:
                game = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                exclusions["archive-unreadable"] += 1
                continue
            if not isinstance(game, Mapping) or _digest(game) != str(envelope["source_archive_sha256"]):
                exclusions["archive-hash-mismatch"] += 1
                continue
            try:
                record = extract_behavior_game(envelope, game)
            except (KeyError, TypeError, ValueError):
                exclusions["feature-extraction-invalid"] += 1
                continue
            if not record["moves"]:
                exclusions["no-opponent-moves"] += 1
                continue
            records.append(record)
        records.sort(key=lambda record: (str(record["started_at"]), str(record["game_id"])))
        family_inventory: dict[str, object] = {}
        support = Counter((str(record["family"]), str(record["public_player_id"])) for record in records)
        for family in ("bargaining", "negotiation", "persuasion"):
            selected = [record for record in records if record["family"] == family]
            counts = [count for (candidate_family, _public_id), count in support.items() if candidate_family == family]
            family_inventory[family] = {
                "games": len(selected),
                "public_player_ids": len(counts),
                "moves": sum(int(record["channel_counts"]["moves"]) for record in selected),
                "timed_moves": sum(int(record["channel_counts"]["timed_moves"]) for record in selected),
                "message_opportunities": sum(int(record["channel_counts"]["message_opportunities"]) for record in selected),
                "nonempty_messages": sum(int(record["channel_counts"]["nonempty_messages"]) for record in selected),
                "ids_with_at_least_2_games": sum(count >= 2 for count in counts),
                "ids_with_at_least_5_games": sum(count >= 5 for count in counts),
                "ids_with_at_least_10_games": sum(count >= 10 for count in counts),
                "maximum_games_per_id": max(counts, default=0),
            }
        inventory = {
            "games": len(records),
            "public_player_ids": len({str(record["public_player_id"]) for record in records}),
            "moves": sum(int(record["channel_counts"]["moves"]) for record in records),
            "timed_moves": sum(int(record["channel_counts"]["timed_moves"]) for record in records),
            "message_opportunities": sum(int(record["channel_counts"]["message_opportunities"]) for record in records),
            "nonempty_messages": sum(int(record["channel_counts"]["nonempty_messages"]) for record in records),
            "families": family_inventory,
            "exclusions": dict(sorted(exclusions.items())),
        }
        summary = {
            "contract": BEHAVIOR_CORPUS_CONTRACT,
            "schema_version": 1,
            "status": "offline-shadow-only",
            "frontier_sequence": identity_manifest["frontier_sequence"],
            "sources": {"identity_dir": str(self.identity_dir), "identity_manifest_sha256": _file_digest(self.identity_dir / "manifest.json"), "game_archive_root": str(self.game_archive_root)},
            "inventory": inventory,
            "promotion": {"live_authority": False, "identity_routing_authority": False, "next_gate": "chronological timing, action, and language channel evaluation with calibrated unknown handling"},
        }
        _write_jsonl(self.output_dir / "behavior-games.jsonl", records)
        _write_json(self.output_dir / "summary.json", summary)
        _atomic_text(self.output_dir / "README.md", self._readme(summary))
        artifacts = ("README.md", "summary.json", "behavior-games.jsonl")
        manifest = {
            "contract": BEHAVIOR_CORPUS_CONTRACT,
            "schema_version": 1,
            "frontier_sequence": summary["frontier_sequence"],
            "identity_manifest_sha256": summary["sources"]["identity_manifest_sha256"],
            "implementation_sha256": {
                "glee_behavior_corpus.py": _file_digest(Path(__file__)),
                "glee_bargaining_twin.py": _file_digest(Path(__file__).with_name("glee_bargaining_twin.py")),
                "glee_negotiation_twin_v2.py": _file_digest(Path(__file__).with_name("glee_negotiation_twin_v2.py")),
                "glee_persuasion_twin_v2.py": _file_digest(Path(__file__).with_name("glee_persuasion_twin_v2.py")),
            },
            "artifacts": {name: {"bytes": (self.output_dir / name).stat().st_size, "sha256": _file_digest(self.output_dir / name)} for name in artifacts},
        }
        _write_json(self.output_dir / "manifest.json", manifest)
        return {"contract": BEHAVIOR_CORPUS_CONTRACT, "output_dir": str(self.output_dir), "frontier_sequence": summary["frontier_sequence"], "inventory": inventory, "promotion": summary["promotion"], "manifest_sha256": _file_digest(self.output_dir / "manifest.json")}
