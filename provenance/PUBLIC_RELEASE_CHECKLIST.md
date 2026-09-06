# Public-release checklist

## Completed release preparation

- [x] Add a scoped software license (`Apache-2.0`) and a paper, documentation, deidentified-analysis, and aggregate-receipt license (`CC-BY-4.0`).
- [x] Add author and AI-assistance credits, citation metadata, and a direct third-party inventory with upstream terms.
- [x] Add a 752-row deidentified sufficient input and an exact command and receipt for reproducing the KI/HI analysis without private raw journals.
- [x] Add 8 deidentified metric-level Parquet tables, a schema-validating reproduction command, and a frozen expected receipt for the learned-model comparisons, dossier ablation, rating estimates, and execution telemetry.
- [x] Record the analysis input and output SHA-256 values, fixed bootstrap design, measured wall time, and peak resident memory.
- [x] Preserve regenerated public runtime and model-code `uv` locks, aggregate model evidence, test commands, deployment receipts, and the repository-wide content manifest.
- [x] Keep credentials, browser profiles, mutable databases, full event journals, cloud-call journals, and superseded mutable runs outside the curated repository.

## Clean-stage privacy boundary

- [x] Build the clean stage without the entire advisor-seed tree, including all 3 raw corpora and derived seed state.
- [x] Withhold raw model corpora, learned weights, statistical packages, account maps, stable identifiers, messages, and identity-bearing vocabularies; release only the deidentified targets, frozen outputs, table-local grouping labels, and resource fields required by the reported estimators.
- [x] Exclude participant-authored messages and identifiers, private journals, credentials, and the unlicensed GLEE SDK while retaining the selected Codex implementation path and deidentified sufficient analysis data.

## Final attended release

- [x] Commit the approved tree as exactly one root commit, then run secret, identifier, message, local-path, email, and account-ID scans across its full reachable history.
- [x] Run the content-manifest verifier, both public reproduction commands, focused tests, LaTeX build, and link checks from a clean clone.
- [x] Confirm that `LICENSE.md`, `CREDITS.md`, `THIRD_PARTY.md`, `CITATION.cff`, `REPRODUCIBILITY.md`, and `data/README.md` remain present and internally consistent.
- [ ] Obtain Serge's explicit confirmation after the full-history disclosure audit, change visibility, create the submission tag, and record the public commit and tag in the manuscript and OpenReview entry.

Private remote staging is authorized. Public visibility, a submission tag, and any OpenReview metadata update remain separate attended actions.
