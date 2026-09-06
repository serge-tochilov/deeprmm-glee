# Artifact scope and source authority

## Final runtime

The final live deployment is `glee-v108-deeprmm-terra-terminal-race-compiled-bargaining-meta15-balanced-self-mirror-shared-w20-g48-1800-20260827T140153Z`. Its immutable private `snapshot/nommd-arena` tree remains authoritative for the historical runtime, prompts, and protocols. The launch recorded source commit `81bc0c53964d7ed3bc89f180c0acb0aca71d0a4c` and exactly 5 dirty files; the private archive preserves those bytes, while this public tree applies the bounded transformations recorded below.

The exported online source is the recursive local-import closure of `nommd_arena.glee_parallel`, plus package initialization. This keeps the final shared family pool, deterministic family engines, statistical overlay, message and timing policies, rating advisory, Terra selector integration, conditional-twin client, public-self-mirror client, transport containment, and their directly required analysis helpers. A second 4-module closure—`glee_analytics_online`, `glee_analytics_incremental`, `glee_selector_backend_replay`, and `glee_selector_model`—is included because the sequence-lab training and evaluation package imports it. Modules that belonged only to obsolete family runners, Fieldglass, browser history synchronization, control-plane daemons, or later paper tooling are absent.

The original `nommd-arena` console-script declaration was removed from the copied `pyproject.toml` because its broad historical `cli.py` is deliberately outside this runtime closure. The public deployment boundary is retained through `paper/evidence.json`, `receipts/v108/public-launch.json`, `receipts/v108/activity-scheduler.json`, and private-source hashes recorded in `provenance/source-authorities.json`; the private archive preserves the exact launch manifest and snapshot inventories.

## Local model services

The conditional-twin and public-self-mirror services remained separate CPU processes during `v108`. Their private source authority is the byte-identical `snapshot/lab` tree shared by `glee-v21-conditional-twin-final-deeprmm-9587-cpu-20260821T003042Z` and `glee-v21-public-self-mirror-final-deeprmm-9587-cpu-20260821T003057Z`. The public package retains the model architectures, training and inference code, and tests, while its replay adapter and dependency lock are regenerated against the self-contained public runtime.

## Public transport transformation

The final deployment selected the Codex CLI branch. The public derivative extracts that branch and its provider-neutral validation and receipt logic into `nommd_arena.codex_transport`, removes unselected transport backends and their historical package boundary, and preserves the live Codex command, schema validation, retry classification, immutable request logging, model settings, and timeout behavior. The final prompt composition is exposed under canonical filenames with no runtime experiment selector; prompt text and composition remain the final deployed instructions.

## Immutable releases

The public tree includes the policy releases pinned by the final launch: Bargaining `v2.22`, Negotiation `v2.14`, Persuasion `v2.8`, message-style `v2`, meta-controller `v1.4`, self-mirror `v1`, and the canonical global tactic ledger loaded by the run. It also includes the exact model-training and CPU-inference source closure. The private archive retains the final 9,587-game conditional and reversed-Persuasion weights, self-mirror ensemble, promoted CPU model releases, statistical package, Bargaining account map, and rating model `v3.0`; the public aggregate evidence receipt binds their source receipts by SHA-256, while deidentified target-and-prediction tables permit independent recomputation of the reported evaluation metrics.

## Metric-level data transformation

`tools/build_deidentified_reproduction.py` records the source-side projection that produced 8 compact Parquet tables from frozen private evidence. Conditional-twin and self-mirror tables retain held-out targets and frozen outputs; the Persuasion table was regenerated once from hash-bound weights and corpus because its original row-level output was not retained; rating, dossier, and execution tables are direct projections. Stable identifiers and chronology were discarded, and required grouping structure was replaced by table-local labels that cannot be linked across tables. `tools/reproduce_paper_metrics.py` validates those schemas and label namespaces, recomputes the estimators, and compares the complete result with `receipts/analysis/reproduced-paper-metrics.json`.

## Evidence boundary

The preserved private repository includes aggregate paper evidence, content hashes, final manifests, execution logs, source-state receipts, selected analytical reports, model evaluation outputs, advisor seeds, and 3 sealed corpora. The public scope replaces the raw-data dependencies of the reported estimators with `data/glee-v104-ki-hi-eligible.jsonl` and the 8 tables under `data/reproduction/`. It excludes participant-authored messages, names, stable participant and game identifiers, inferred account links, exact timestamps, identity-bearing model vocabularies, learned weights, and raw model corpora. It also excludes the 424 MB final event journal, the 53 MB cloud-call journal, earlier runs, mutable databases, authentication material, and private browser state. Private final-v108 journal prefixes remain bound by exact size and SHA-256 in `paper/evidence.json`; campaign-wide execution aggregates are separately bound to the private all-history reconstruction by `receipts/analysis/all-history-execution-summary.json`.

The final launch used the live `glee-opponent-timing-v1` implementation and its shared SQLite store. The implementation, tests, and protocol are included, but the database is not: the launch manifest binds only its mutable path, not a content hash, and a later copy cannot be represented as the exact launch state. Timing remained a shadow identity channel rather than an authoritative action router.
