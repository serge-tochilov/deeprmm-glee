# Reproducibility

## Claim boundary

The main reproducibility claim is analytical: released deidentified sufficient inputs, fixed estimators, resampling designs, exact environments, and frozen expected receipts reproduce the KI/HI inference, held-out learned-model comparisons, dossier ablation, rating estimates, and execution telemetry at the precision reported in the paper. The artifact also supports inspection and testing of the final architecture, deterministic policies, prompt composition, model-training and inference code, public-self-mirror interface, rating advisory, and transport guards.

Reproducibility does not mean retraining from the withheld private corpus or replaying the live competition. A closed-source language model, a changing opponent population, matchmaking, network timing, and rating shrinkage make the historical leaderboard trajectory non-repeatable. The paper therefore makes no claim that a new online run will recreate the final rating or rank.

## Exact KI/HI analysis

The released input is `data/glee-v104-ki-hi-eligible.jsonl`: 752 chronologically indexed records containing only family, KI/HI mode, and authenticated displayed-rating delta. Its SHA-256 is `0e750482c09113bdc223ac8cf1e309c11ce7f1b9b2256598d47a1ce7d4c74913`.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv run --no-project python tools/glee_ki_hi_inference.py --eligible-input data/glee-v104-ki-hi-eligible.jsonl --output /tmp/glee-v104-ki-hi-results.json` from the repository root. Compare the output with `receipts/analysis/glee-v104-ki-hi-public-reproduction.json`; byte equality is expected, and the reference receipt's SHA-256 is `ffe0fc8f3b98ec21c30e308fa96f078919880279da316b7dc19309b94dda89e1`.

The analysis performs 50,000 circular moving-block resamples at the primary 20-game block length and 20,000 resamples at each 10- and 40-game sensitivity length, using seed `1729`. On the documented AMD WSL host, this command used at most 27,264 KiB resident memory and completed in 31.55 seconds.

## Exact metric-level reproduction

Eight Parquet tables under `data/reproduction/` contain deidentified sufficient inputs for the conditional twin, reversed Persuasion head, public self-mirror, rating model, dossier comparison, and execution summaries. Run `UV_CACHE_DIR=/tmp/deeprmm-glee-lab-uv-cache uv run --project artifact/nommd-arena/opponent-sequence-lab --locked --no-sync python tools/reproduce_paper_metrics.py --output /tmp/deeprmm-glee-paper-metrics.json --expected receipts/analysis/reproduced-paper-metrics.json` from the repository root. The command validates the released schemas and anonymous label namespaces, recomputes the metrics and cluster bootstraps, and fails unless the complete output equals the frozen expected receipt.

On the documented AMD WSL host, the metric-level command completed in 1.66 seconds with 135,664 KiB peak resident memory. The input tables occupy approximately 1.0 MB. Their fields and deidentification boundary are documented in `data/reproduction/README.md`.

## Python environments

The runtime and sequence-model packages retain separate `uv` projects because their dependency sets differ substantially. The public locks are regenerated from the curated Codex-only source tree; the private source-authority record preserves the hashes of the exact promoted environments.

Synchronize the runtime environment with `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv sync --project artifact/nommd-arena --locked` and run its focused tests with `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv run --project artifact/nommd-arena --locked --no-sync pytest artifact/nommd-arena/tests`. The explicit test path prevents the nested sequence-model project from being discovered under the runtime root. The packaged Codex-only runtime suite passes 260 tests.

Synchronize the model environment with `UV_CACHE_DIR=/tmp/deeprmm-glee-lab-uv-cache uv sync --project artifact/nommd-arena/opponent-sequence-lab --locked` and run its tests with `UV_CACHE_DIR=/tmp/deeprmm-glee-lab-uv-cache uv run --project artifact/nommd-arena/opponent-sequence-lab --locked --no-sync pytest artifact/nommd-arena/opponent-sequence-lab/tests`. The packaged public sequence-model suite passes 58 tests and skips one optional GPU test on the CPU validation path.

## Learned-model evidence

`receipts/models/frozen-model-metrics.json` preserves private-source aggregate dimensions, fixed seeds, held-out metrics, declared baselines, and source-receipt hashes. `receipts/analysis/reproduced-paper-metrics.json` is independently derived from the released sufficient tables. On 2,928 held-out direct responses, the validation-selected convex stack attained equal-family NLL 0.325965, compared with 0.329210 for its sequence-only component and 0.452166 for its engineered-only component. Its accuracy was 0.886954, compared with 0.899249 for the sequence-only component. On 3,021 held-out Persuasion transitions, the continuation head reduced game-macro NLL from 0.702333 for the preceding-signal-and-action Markov baseline to 0.614317, with a paired difference of -0.088015 and 95% interval [-0.140926, -0.033548]. Its lower raw and balanced accuracies are recorded in both receipts. Self-mirror accuracies are descriptive because no simple baseline was frozen.

The action-conditioned sequence expert has 344,059 parameters per seed, and the public self-mirror has 270,275 parameters per seed. Public code preserves the architectures and evaluation machinery. The deidentified prediction tables support exact recomputation at paper precision, while raw participant-derived corpora and learned weights remain withheld; exact retraining is outside the public claim.

The original training used PyTorch 2.10.0 with CUDA 13.0 on one NVIDIA RTX 2060 with 6 GB VRAM, a 6-core/12-thread AMD CPU, and 32 GB host RAM. The 2 conditional sequence seeds took 231.51 and 257.68 seconds; the 2 public-self-mirror seeds took 2,384.72 and 2,109.12 seconds; and the reversed Persuasion experiment took 76.11 seconds. Promoted inference used one CPU thread per service. The 292-request dossier comparison used 8 concurrent cloud workers and ran for 2 hours 10 minutes 14.63 seconds. The final deployment receipt reports a 508.6-second cold compiled-state build, a 2.58-second warm restore, and decision latency of 0.043 seconds median, 7.69 seconds mean, and 26.05 seconds p95 across the frozen prefix.

`receipts/analysis/all-history-execution-summary.json` is a public aggregate derivative of the SHA-256-bound private all-history reconstruction. It reports 13,125 distinct journal-terminal games, 97,581 worker decisions, 97,473 submitted moves, zero invalid submissions, 1,074 fallbacks, and 11.01/17.70/77.97/108.52-second median/mean/p95/maximum decision latency through the clean retirement cutoff. Its nested v108 fields match `paper/evidence.json`. The dashboard's 13,031 displayed games are a separate scoreboard observation and are not expected to equal the journal terminal-ID count.

Historical wall times for every exploratory or failed training run, the rating-model fit, and total provider-side cloud compute were not preserved. These quantities are unnecessary for running the released estimators, and their absence is disclosed without an invented estimate.

## Final online invocation

`paper/evidence.json`, `receipts/v108/public-launch.json`, `receipts/v108/activity-scheduler.json`, and `provenance/source-authorities.json` preserve the public deployment boundary without exposing the private launch path, agent identifier, credential-file location, or participant-derived model locations. The public launch receipt binds the withheld private manifest by SHA-256 and records the deidentifying tactic-ledger transformation. The operational supervisor and private credential file are intentionally excluded: the competition endpoint is external, credentials are not research artifacts, and the supported results concern the preserved historical run.

The final online policy used OpenAI's GPT-5.6 Terra High as both candidate planner and selector, with GPT-5.6 Luna High as capacity fallback. The planner committed candidate actions before receiving local action-conditioned forecasts; the selector then received the frozen candidate set, conditional opponent-response evidence, bounded public-self-mirror evidence, family-policy evidence, and rating estimates before returning one candidate index. The public runtime preserves only this selected Codex execution path and records that bounded source transformation in `provenance/SCOPE.md`.

The opponent-timing SQLite store is intentionally absent. Its live implementation and tests are reproducible, while its launch-time contents were mutable and were not independently frozen by a hash receipt.

## Evidence and privacy boundary

`paper/evidence.json` records byte limits, line counts, final sequences, and SHA-256 values for the exact event and sensor prefixes used by the manuscript. Those raw prefixes remain in a private evidence archive because they contain participant-derived interactions and are much larger than the curated artifact. The receipt detects substitution, truncation, or later append when checked against that archive.

The deidentified KI/HI table is sufficient for the paper's identity-mode inference. The metric-level Parquet tables preserve the additional targets, frozen predictions, anonymous within-table cluster structure, call-cost fields, and execution measurements needed by the remaining reported estimators. The clean public history excludes raw corpora, advisor seeds, participant identifiers and messages, inferred account linkages, exact timestamps, identity-bearing model vocabularies, learned weights, and full event or cloud-call journals. The preserved private archive remains the source authority for the SHA-256-bound source receipts.
