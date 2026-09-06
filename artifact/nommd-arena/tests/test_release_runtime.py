import hashlib
import json
from pathlib import Path

import pytest

from nommd_arena.glee_live_policy import BargainingLivePolicyStore, NegotiationLivePolicyStore, PersuasionLivePolicyStore
from nommd_arena.glee_tactics import GlobalTacticLedger
from nommd_arena.model_runner import ArenaCodexRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


@pytest.mark.parametrize(
    ("family", "release_name", "revision", "store_type"),
    [
        ("bargaining", "v2.22-bounded-v3-low-share.json", "bargaining-v2.22-bounded-v3-low-share", BargainingLivePolicyStore),
        ("negotiation", "v2.14-buyer-surplus-and-single-counter.json", "v2.14-buyer-surplus-and-single-counter", NegotiationLivePolicyStore),
        ("persuasion", "v2.8-bounded-no-response-routing.json", "v2.8-bounded-no-response-routing", PersuasionLivePolicyStore),
    ],
)
def test_final_family_policy_pointer_resolves_to_valid_release(family: str, release_name: str, revision: str, store_type: type[object]) -> None:
    root = PROJECT_ROOT / "policies" / family
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    release_path = root / str(pointer["release"])
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert release_path.name == release_name
    assert hashlib.sha256(release_path.read_bytes()).hexdigest() == pointer["release_sha256"]
    assert release["revision"] == revision
    store_type._validate_release(release)


def test_final_prompt_composition_is_complete(tmp_path: Path) -> None:
    runner = ArenaCodexRunner(prompts_dir=PROJECT_ROOT / "prompts", log_path=tmp_path / "calls.jsonl", session_dir=tmp_path / "session")
    for family in ("bargaining", "negotiation", "persuasion"):
        planner, planner_version, _planner_digest = runner._prompt(f"glee_meta_controller_v2_15_planner_{family}")
        selector, selector_version, _selector_digest = runner._prompt(f"glee_meta_controller_v2_15_selector_{family}")
        assert "tetrad" not in (planner + selector).casefold()
        assert "Generate and commit the immutable candidate set" in planner
        assert "contains no policy-marginal prior" in selector
        assert "every guarded candidate has equal selection standing" in selector.casefold()
        assert planner_version.startswith(f"glee_meta_controller_v2_15_planner_{family}@")
        assert selector_version.startswith(f"glee_meta_controller_v2_15_selector_{family}@")


def test_final_global_tactic_ledger_matches_launch_receipt() -> None:
    manifest = json.loads((REPOSITORY_ROOT / "receipts" / "v108" / "public-launch.json").read_text(encoding="utf-8"))
    ledger = GlobalTacticLedger(PROJECT_ROOT / "tactics" / "glee-global-tactics.json")
    assert ledger.sha256 == manifest["global_tactic_ledger"]["public_derivative_sha256"]
    assert manifest["global_tactic_ledger"]["source_sha256"] == "90ce90befd3261536c752e00c4b3bcaf3b7130cb96a91e5aad5b309f6b254c32"


def test_v108_launch_receipt_identifies_the_exported_execution_path() -> None:
    manifest = json.loads((REPOSITORY_ROOT / "receipts" / "v108" / "public-launch.json").read_text(encoding="utf-8"))
    assert manifest["worker_policy"] == "meta15"
    assert manifest["model"] == "gpt-5.6-terra"
    assert manifest["effort"] == "high"
    assert manifest["max_parallel"] == 20
    assert manifest["families"] == ["bargaining", "negotiation", "persuasion"]
    assert manifest["conditional_twin"]["forecast_exposed_to_selector"] is True
    assert manifest["meta_controller"]["public_self_mirror"]["planner_visibility"] is False
