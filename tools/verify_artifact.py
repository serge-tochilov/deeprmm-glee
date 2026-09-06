"""Verify the curated DeepRMM GLEE paper artifact without network access."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "files.sha256"
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".sqlite3", ".sqlite3-shm", ".sqlite3-wal"}
PAPER_BUILD_SUFFIXES = {".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".run.xml"}
TEXT_SUFFIXES = {".cff", ".json", ".jsonl", ".md", ".py", ".tex", ".txt", ".toml", ".yaml", ".yml"}
ALLOWED_EMAILS = {"garnett@wustl.edu", "serge.tochilov@ncis.org"}
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
UUID_PATTERN = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
SECRET_PATTERNS = {
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "bearer-token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    "openai-api-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "pem-private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
}
FORBIDDEN_RELEASE_PATTERNS = {
    "excluded-provider-name": re.compile(r"\b(?:anth" + "ropic|clau" + r"de)\b", re.IGNORECASE),
    "internal-identity-or-channel-name": re.compile(
        r"\b(?:"
        + "mo"
        + "no"
        + "|fo"
        + "lio"
        + "|coul"
        + "ped"
        + "|inter"
        + "num"
        + r")\b|\bcoup"
        + "led"
        + r"[-_ ]?(?:connec"
        + "tor"
        + "|chan"
        + "nel"
        + "|sess"
        + r"ion)\b|"
        + "mo"
        + "no"
        + "fo"
        + "lio"
        + r"|\baccount[-_ ]?[ab]\b",
        re.IGNORECASE,
    ),
    "excluded-project-name": re.compile(r"\brmm[-_ ]" + r"bench\b", re.IGNORECASE),
    "removed-prompt-experiment": re.compile(r"(?:prompt[_ -]comp" + "action|glee_meta_controller_v2_15_" + "compact)", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        bool(IGNORED_PARTS.intersection(relative.parts))
        or relative.parts[:3] == ("artifact", "nommd-arena", "runs")
        or any(path.name.endswith(suffix) for suffix in IGNORED_SUFFIXES)
        or relative.parts[:1] == ("paper",) and any(path.name.endswith(suffix) for suffix in PAPER_BUILD_SUFFIXES)
        or any(part.endswith(".egg-info") for part in relative.parts)
    )


def artifact_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and path != MANIFEST and not ignored(path))


def write_content_manifest() -> None:
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in artifact_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} content hashes")


def verify_content_manifest() -> int:
    failures = 0
    listed: set[Path] = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        listed.add(path.resolve())
        if not path.is_file() or sha256(path) != expected:
            print(f"FAIL content {relative}")
            failures += 1
    actual = {path.resolve() for path in artifact_files()}
    for path in sorted(actual - listed):
        print(f"FAIL unmanifested {path.relative_to(ROOT)}")
        failures += 1
    for path in sorted(listed - actual):
        print(f"FAIL missing {path.relative_to(ROOT)}")
        failures += 1
    return failures


def local_import_closure(root: Path, entry: str) -> set[str]:
    pending = [entry]
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        path = root / f"{module}.py"
        if not path.is_file():
            raise ValueError(f"missing local module {module}")
        seen.add(module)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            candidates = [node.module.split(".")[0]] if node.module else [alias.name.split(".")[0] for alias in node.names]
            pending.extend(candidate for candidate in candidates if (root / f"{candidate}.py").is_file() and candidate not in seen)
    return seen


def verify_runtime_closure() -> int:
    root = ROOT / "artifact" / "nommd-arena" / "src" / "nommd_arena"
    expected = {path.stem for path in root.glob("*.py")} - {"__init__"}
    observed = local_import_closure(root, "glee_parallel")
    for entry in ("glee_analytics_online", "glee_selector_backend_replay", "glee_selector_model"):
        observed.update(local_import_closure(root, entry))
    if observed == expected:
        return 0
    print(f"FAIL source closure missing={sorted(observed - expected)} extra={sorted(expected - observed)}")
    return 1


def verify_policy_pointers() -> int:
    failures = 0
    root = ROOT / "artifact" / "nommd-arena" / "policies"
    for pointer_path in sorted(root.glob("*/current.json")):
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        relative = pointer.get("path") or pointer.get("release")
        expected = pointer.get("sha256") or pointer.get("release_sha256")
        target = (pointer_path.parent / str(relative or "")).resolve()
        if not relative or not expected or target.parent != (pointer_path.parent / "releases").resolve() or not target.is_file() or sha256(target) != expected:
            print(f"FAIL policy pointer {pointer_path.relative_to(ROOT)}")
            failures += 1
    return failures


def verify_public_model_receipt() -> int:
    path = ROOT / "receipts" / "models" / "frozen-model-metrics.json"
    if not path.is_file():
        print("FAIL public aggregate model receipt missing")
        return 1
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "contract": "deeprmm-glee-public-aggregate-model-evidence-v1",
        "frozen_logical_game_count": 9587,
        "status": "public-aggregate-derivative-of-private-hash-bound-receipts",
    }
    failures = 0
    for key, expected in required.items():
        if receipt.get(key) != expected:
            print(f"FAIL public aggregate model receipt {key}")
            failures += 1
    for section in ("conditional_twin", "opponent_statistical_packages", "persuasion_buyer_continuation", "rating_model", "self_mirror"):
        if not isinstance(receipt.get(section), dict):
            print(f"FAIL public aggregate model receipt section {section}")
            failures += 1
    twin = receipt.get("conditional_twin", {}).get("held_out_comparison", {})
    if twin.get("responses") != 2928 or twin.get("selected_convex_stack_equal_family_negative_log_likelihood") != 0.3259653652706446:
        print("FAIL public aggregate model receipt conditional comparison")
        failures += 1
    continuation = receipt.get("persuasion_buyer_continuation", {})
    if continuation.get("selected_minus_baseline_game_macro_negative_log_likelihood") != -0.088015:
        print("FAIL public aggregate model receipt persuasion baseline")
        failures += 1
    return failures


def verify_public_execution_receipt() -> int:
    path = ROOT / "receipts" / "analysis" / "all-history-execution-summary.json"
    if not path.is_file():
        print("FAIL public all-history execution receipt missing")
        return 1
    receipt = json.loads(path.read_text(encoding="utf-8"))
    summary = receipt.get("execution_summary", {})
    required = {
        "contract": "deeprmm-glee-public-all-history-execution-summary-v1",
        "private_source_receipt_sha256": "36aa5963d411f624c620ac21160636188bd05e2f0c22246bab6f60a9d0a5a413",
        "status": "public-aggregate-derivative-of-private-hash-bound-receipt",
    }
    failures = 0
    for key, expected in required.items():
        if receipt.get(key) != expected:
            print(f"FAIL public all-history execution receipt {key}")
            failures += 1
    for key, expected in {
        "distinct_terminal_games": 13125,
        "worker_decisions": 97581,
        "move_submissions": 97473,
        "invalid_move_submissions": 0,
    }.items():
        if summary.get(key) != expected:
            print(f"FAIL public all-history execution receipt {key}")
            failures += 1
    if receipt.get("final_v108_prefix_comparison", {}).get("status") != "matched":
        print("FAIL public all-history execution receipt v108 comparison")
        failures += 1
    return failures


def verify_metric_reproduction_receipt() -> int:
    path = ROOT / "receipts" / "analysis" / "reproduced-paper-metrics.json"
    expected_receipt_sha256 = "b16b9938fba3f49dbcc94d8147bd8895ec833bf00daea698c5e6a1e7630a4683"
    if not path.is_file() or sha256(path) != expected_receipt_sha256:
        print("FAIL metric-level reproduction receipt")
        return 1
    receipt = json.loads(path.read_text(encoding="utf-8"))
    failures = 0
    if receipt.get("contract") != "deeprmm-glee-deidentified-paper-metric-reproduction-v1" or receipt.get("schema_version") != 1:
        print("FAIL metric-level reproduction receipt contract")
        failures += 1
    source_files = receipt.get("source_files")
    expected_names = {
        "conditional-twin-test.parquet",
        "dossier-comparison.parquet",
        "execution-decisions.parquet",
        "execution-submissions.parquet",
        "execution-terminal-games.parquet",
        "persuasion-continuation-test.parquet",
        "rating-model-heldout.parquet",
        "self-mirror-test.parquet",
    }
    if not isinstance(source_files, dict) or set(source_files) != expected_names:
        print("FAIL metric-level reproduction source inventory")
        failures += 1
    else:
        for name, expected in source_files.items():
            source = ROOT / "data" / "reproduction" / name
            if Path(name).name != name or not source.is_file() or sha256(source) != expected:
                print(f"FAIL metric-level reproduction source {name}")
                failures += 1
    metrics = receipt.get("metrics", {})
    twin = metrics.get("conditional_twin", {}).get("arms", {}).get("convex_stack", {})
    persuasion = metrics.get("persuasion_buyer_continuation", {}).get("paired_game_bootstrap", {})
    execution = metrics.get("execution", {}).get("all_history", {})
    if twin.get("equal_family_negative_log_likelihood") != 0.325965365271 or twin.get("accuracy") != 0.886953551913:
        print("FAIL metric-level reproduction conditional twin")
        failures += 1
    if persuasion.get("selected_minus_markov_game_macro_negative_log_likelihood") != -0.08801602003 or persuasion.get("bootstrap_95_percent_interval") != [-0.140925926453, -0.033548329932]:
        print("FAIL metric-level reproduction persuasion")
        failures += 1
    if execution.get("distinct_terminal_games") != 13125 or execution.get("worker_decisions") != 97581 or execution.get("invalid_move_submissions") != 0:
        print("FAIL metric-level reproduction execution")
        failures += 1
    dossier = metrics.get("dossier_comparison", {}).get("uncertainty", {}).get("direct_context_minus_prose_dossier", {})
    if set(dossier) != {"proposal_mae_delta", "proposal_nll_delta", "proposal_rmse_delta", "response_brier_delta", "response_nll_delta"} or not all(value.get("interval_crosses_zero") is True for value in dossier.values()):
        print("FAIL metric-level reproduction dossier comparison")
        failures += 1
    rating = metrics.get("rating_model", {}).get("families", {})
    expected_rating_rows = {"bargaining": 679, "negotiation": 442, "persuasion": 317}
    if {family: values.get("held_out_rows") for family, values in rating.items()} != expected_rating_rows:
        print("FAIL metric-level reproduction rating model")
        failures += 1
    return failures


def verify_checklist_answers() -> int:
    path = ROOT / "paper" / "checklist.tex"
    answers = re.findall(r"^\s*\\item\[\] Answer:\s*\\(answerYes|answerNA|answerNo)\{\}\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if len(answers) != 16:
        print(f"FAIL checklist answer count {len(answers)}")
        return 1
    if "answerNo" in answers:
        print("FAIL checklist contains a No answer")
        return 1
    return 0


def verify_evidence_ledger() -> int:
    failures = 0
    ledger = ROOT / "paper" / "evidence-sources.md"
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "`" not in line or "SHA-256" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3 or not cells[1].startswith("`") or not cells[2].startswith("`"):
            continue
        relative = cells[1].strip("`")
        expected = cells[2].strip("`")
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            print(f"FAIL evidence {relative}")
            failures += 1
    return failures


def verify_exclusions() -> int:
    forbidden_names = {"auth.json", "cookies.sqlite", "cookies.sqlite3", "glee.env", "events.jsonl", "llm_calls.jsonl"}
    failures = 0
    for relative in (Path("artifact/nommd-arena/models"), Path("artifact/nommd-arena/seeds")):
        if (ROOT / relative).exists():
            print(f"FAIL private-derived directory {relative}")
            failures += 1
    for path in ROOT.rglob("*"):
        relative_text = path.relative_to(ROOT).as_posix()
        for label, pattern in FORBIDDEN_RELEASE_PATTERNS.items():
            if pattern.search(relative_text):
                print(f"FAIL {label} path {relative_text}")
                failures += 1
        if path.is_file() and path.name in forbidden_names:
            print(f"FAIL forbidden artifact {path.relative_to(ROOT)}")
            failures += 1
        local_home_marker = str(Path("/home") / "airoot")
        if path.is_file() and path.suffix in TEXT_SUFFIXES and not ignored(path):
            text = path.read_text(encoding="utf-8", errors="replace")
            if local_home_marker in text:
                print(f"FAIL local path {path.relative_to(ROOT)}")
                failures += 1
            if UUID_PATTERN.search(text):
                print(f"FAIL UUID {path.relative_to(ROOT)}")
                failures += 1
            unexpected_emails = set(EMAIL_PATTERN.findall(text)) - ALLOWED_EMAILS
            if unexpected_emails:
                print(f"FAIL unexpected email {path.relative_to(ROOT)}")
                failures += 1
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    print(f"FAIL {label} {path.relative_to(ROOT)}")
                    failures += 1
            for label, pattern in FORBIDDEN_RELEASE_PATTERNS.items():
                if pattern.search(text):
                    print(f"FAIL {label} {path.relative_to(ROOT)}")
                    failures += 1
    return failures


def main() -> int:
    failures = verify_content_manifest() + verify_runtime_closure() + verify_policy_pointers() + verify_public_model_receipt() + verify_public_execution_receipt() + verify_metric_reproduction_receipt() + verify_checklist_answers() + verify_evidence_ledger() + verify_exclusions()
    if failures:
        print(f"artifact verification failed: {failures} issue(s)")
        return 1
    print("artifact verification passed")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-manifest"]:
        write_content_manifest()
        raise SystemExit(0)
    if sys.argv[1:]:
        raise SystemExit("usage: verify_artifact.py [--write-manifest]")
    raise SystemExit(main())
