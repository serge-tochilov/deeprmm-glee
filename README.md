# DeepRMM GLEE 2026 artifact

This curated release contains the paper, final live implementation path, policy releases, model-training and inference code, deidentified analysis data, and aggregate source-bound receipts for the DeepRMM-01 entry in the 2026 GLEE Competition. It is an inspectable and testable source closure, not a turnkey reproduction of the historical live agent.

The repository is a curated artifact with a disclosure-audited release history. The executable core is a documented Codex-only public derivative of the immutable final `v108` deployment snapshot; the conditional-twin and public-self-mirror code is derived from their final CPU-service snapshots. Historical runners, alternate model transports, superseded policy releases, Fieldglass collector code, prose dossiers, browser automation, credentials, mutable run state, and full private event or model-call journals are excluded.

## Contents

- `artifact/nommd-arena/`: the final runtime module closure, canonical Terra prompts, local Codex transport, final policies, model interfaces, and focused tests.
- `artifact/nommd-arena/opponent-sequence-lab/`: the exact model-training and CPU-inference package frozen by the promoted conditional-twin and self-mirror services.
- `paper/`: LaTeX source, references, compiled review PDF, quantitative evidence receipt, and evidence ledger.
- `data/`: deidentified sufficient inputs for the reported KI/HI inference, predictor comparisons, dossier ablation, rating estimates, and execution telemetry.
- `receipts/`: final-deployment and all-history execution summaries, chronological repair evidence, and aggregate learned-model evidence.
- `docs/analysis/`: the small set of contemporaneous analyses cited by the paper.
- `provenance/`: source-boundary documentation and a repository-wide content manifest.

## Verification

Run `uv run --no-project python tools/verify_artifact.py` to verify the committed content manifest, immutable policy pointers, public aggregate receipts, and paper evidence ledger.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv run --project artifact/nommd-arena --locked --no-sync pytest artifact/nommd-arena/tests` for the focused runtime tests after dependencies are synchronized.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-lab-uv-cache uv run --project artifact/nommd-arena/opponent-sequence-lab --locked --no-sync pytest artifact/nommd-arena/opponent-sequence-lab/tests` for the sequence-model tests after dependencies are synchronized.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv run --no-project python tools/glee_ki_hi_inference.py --eligible-input data/glee-v104-ki-hi-eligible.jsonl --output /tmp/glee-v104-ki-hi-results.json` to reproduce the KI/HI block-bootstrap analysis. The result should match `receipts/analysis/glee-v104-ki-hi-public-reproduction.json` byte for byte.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-lab-uv-cache uv run --project artifact/nommd-arena/opponent-sequence-lab --locked --no-sync python tools/reproduce_paper_metrics.py --output /tmp/deeprmm-glee-paper-metrics.json --expected receipts/analysis/reproduced-paper-metrics.json` to recompute the remaining principal quantitative results from `data/reproduction/` and verify the frozen expected receipt.

See `REPRODUCIBILITY.md` for the claim boundary, exact environments, compute disclosure, and expected outputs.

## Data boundary

The released 752-row KI/HI table contains only family, chronology index, identity mode, and authenticated rating delta. The 8 Parquet tables under `data/reproduction/` retain deidentified targets, frozen predictions, anonymous within-table cluster labels, call costs, and execution fields sufficient for the other reported estimators. They contain no messages, prompts, names, stable participant or game identifiers, exact timestamps, account links, credentials, or private paths.

Raw advisor corpora, participant-derived statistical packages and account maps, identity-bearing vocabularies, learned weights, and full event and cloud-call journals remain only in the preserved private archive. The released prediction tables permit metric recomputation without permitting exact retraining, identity reconstruction, or online replay. Private-source aggregate metrics and declared baselines remain bound by SHA-256 in `receipts/models/frozen-model-metrics.json`.

Original software is released under `Apache-2.0`; the paper, project-authored documentation, deidentified analysis data, and public receipts use `CC-BY-4.0`. See `LICENSE.md`, `CREDITS.md`, and `THIRD_PARTY.md` for scope, attribution, exclusions, and upstream terms.
