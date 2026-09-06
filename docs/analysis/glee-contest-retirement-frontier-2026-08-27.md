# GLEE contest retirement frontier

**Decision time:** 2026-08-27 23:51 EDT

**Status:** Live GLEE operation is retired. Every canonical game and infrastructure module reports `stopped`, no GLEE worker or service process remains, and no new game is admitted. This is an operational retirement, not deletion of the implementation or evidence.

## Decision

GLEE began as an ecological test of persistent recursive mental modeling in repeated social games. In practice, its low-dimensional mechanics, very high repetition counts, nonstationary opponent population, and rating incentives increasingly rewarded throughput, specialized family heuristics, and rapid empirical adaptation. These are legitimate agent-engineering problems, but they no longer isolate the central question of whether deeper RMM improves social reasoning.

Continuing would spend cloud quota, engineering time, and attention mainly on leaderboard maintenance and a capacity race. That opportunity cost now exceeds the expected scientific value, especially because the NoMMD and RMM papers need focused work. The live system is therefore stopped without claiming that RMM failed: the competition became a poor instrument for separating RMM from scale, control engineering, and repeated-game optimization.

No GLEE game runner, collector, sensor, history synchronizer, reporter, analytics service, conditional twin, or self-mirror should be restarted for ranking defense or additional opportunistic data. A future restart requires an explicit new research protocol rather than continuation by inertia.

## Clean operational boundary

The canonical `stop all` request was issued at `2026-08-28T03:51:02Z`, corresponding to 2026-08-27 23:51 EDT. The last fully supported live sensor frontier was recorded at `2026-08-28T03:51:02.158455Z`, event sequence `134914`, with prefix SHA-256 `d6b2e6fb40f995daaa69e62abbe5855923c4bc0dd4b44a2e8fbac3209941b918` and 17 active games.

| Family | Displayed games | Displayed rating | Change during v108 before stop |
| --- | ---: | ---: | ---: |
| Bargaining | 5,724 | 1,964.77 | 352 games, −33.48 rating |
| Negotiation | 4,052 | 1,830.96 | 324 games, −97.65 rating |
| Persuasion | 3,255 | 1,939.14 | 333 games, +156.96 rating |
| Combined | 13,031 | 1,911.62 mean | 1,009 games, +8.61 mean rating |

The v108 epoch began from displayed ratings `1,998.25`, `1,928.61`, and `1,782.18` after 5,372, 3,728, and 2,922 games respectively. These before-and-after changes are descriptive only: opponent composition, exact game configurations, inactivity pull, concurrent completions, and the evolving rating landscape prevent a causal interpretation.

The authenticated history store had synchronized 13,148 game records, of which 13,033 carried a rating delta, by `2026-08-28T03:50:23Z`. This is an evidence-store inventory rather than a second statement of the live scoreboard count; timeout, synchronization, and record-retention semantics differ from the displayed per-family counters.

The final live epoch is `glee-v108-deeprmm-terra-terminal-race-compiled-bargaining-meta15-balanced-self-mirror-shared-w20-g48-1800-20260827T140153Z`. Its immutable source snapshot records Git commit `81bc0c53964d7ed3bc89f180c0acb0aca71d0a4c`, module specification SHA-256 `a00a3a8848f50c1ff86e4e1f991f7dc61d6dd14f40a4f1749bf81a57999b7fd6`, and shared-contract SHA-256 `eaff61d5ea1019232a89e2732c5954653f946009705950c2e37bd0460c2a6f40`. The source tree had 5 recorded dirty files at launch, so the immutable snapshot, not an assumed Git checkout, is the implementation authority for this epoch.

## Shutdown tail

`stop all` changed every module's desired state nearly simultaneously. The 17 admitted games continued to drain, but the optional conditional-twin and public-self-mirror services exited before those games ended. Their later calls therefore failed closed and used the existing guarded deterministic fallback. 15 games completed during this interval, while one round-43 Bargaining game and one round-17 Persuasion game remained active long enough that the canonical hard-abort path was used to satisfy the decision to stop every process.

The shutdown tail contained 86 worker fallbacks caused by absent optional services: 39 conditional-twin connection failures and 47 buyer-continuation connection failures. It produced zero invalid submissions. This confirms safe degradation, but the decisions did not receive the ordinary v108 evidence surface and must not be mixed into a fully supported policy analysis.

The exact sensor frontier above is consequently the clean live-operational boundary. After the game process exited, the v108 journal ended at event sequence `180911` and timestamp `2026-08-28T04:07:27.713524Z`; it contains 180,911 newline-terminated events in 424,359,544 bytes with SHA-256 `c31565f3acd9a9eeecafff7570521f90f85a6b33485baa1ed954aeb7dd4dabb8`. The companion `llm_calls.jsonl` contains 8,528 records in 53,404,086 bytes with SHA-256 `f7c30fdb5e81cc0e2fef1c3893c369a92930a8cbc7351096281685099734f5bc`, and `manifest.json` has SHA-256 `a404cbac6939fb0eecfd2fb41416ea605615e1dcea0a6605cbea8ced323f6fea`.

The sealed v108 journal covers 1,040 admitted games: 356 completed Bargaining games, 327 completed Negotiation games, 355 completed Persuasion games, and the 2 intentionally incomplete games. A read-only API snapshot at `2026-08-28T04:09:36Z` confirmed zero active games and showed 5,726 Bargaining games at `1,958.21`, 4,052 Negotiation games at `1,830.96`, and 3,268 Persuasion games at `1,954.12`; the 2 aborted server-side games had resolved without changing these counters. This later snapshot describes shutdown aftermath rather than replacing the clean frontier.

## What the competition established

- A persistent agent can operate across thousands of games with one credential-owning supervisor, immutable turn receipts, exactly-once submission intent, bounded transport recovery, transactional behavioral memory, and independent read-only statistics services.
- The original explicit tetrad carrier did not earn continued inclusion: all 1,502 preserved successful selector calls returned null updates, while the carried state largely duplicated observable own actions and a neutral seed. Removing it was an evidence-based negative result, not abandonment of NoMMD's broader theory.
- Compact statistical opponent packages were operationally more useful than growing prose dossiers. Population priors, sparse identity-conditioned counts, current-game updates, and family-specific causal models supplied clearer, cheaper, and more testable signals.
- The 1.5-round controller made a concrete order-2 computation possible: generate candidate actions, predict opponent responses conditional on those actions, and select with those counterfactuals visible. The public self-mirror added a population-level estimate of which action an observer might expect from DeepRMM-01's public behavior.
- Known-identity and hidden-identity play exposed identity itself as a strategic variable. Timing, language, activity, and action style can serve both as behavioral evidence and as a concealment surface, although current deanonymization remained too uncertain for authoritative routing.
- The strongest practical gains came from specialized economics, causal data boundaries, deterministic safeguards, transport discipline, and iterative error analysis. This is a substantive result about the demands of repeated-agent deployment even where it is not evidence for deep RMM.

## What the competition did not establish

- It did not provide a controlled transcript-only versus flat-memory versus recursive-model ablation under fixed opponents and configurations, so it cannot identify the causal effect of recursive depth.
- It did not establish that the conditional twin, self-mirror, statistical packages, identity concealment, or cloud selector improved rating. Their receipts establish use and causal availability before action, not counterfactual benefit.
- It did not make leaderboard rating a stable proxy for RMM capability. Ratings depended on exact configuration, role, opponent strength, activity, population drift, transport failures, and widespread strategic adaptation.
- It did not validate reliable hidden-opponent identification or opponent-specific order-2 modeling. True-identity routing improved offline next-action prediction, but inferred soft routing did not clear the live authority gate.
- It did not show that more natural-language reasoning was better. The game families were simple enough that sufficiently large empirical corpora supported fast heuristics and personalized policies, while long cloud reasoning often added cost and timeout exposure.

## Architecture at retirement

The retired v108 system used one 20-worker pool with independent `1,800/1,800/1,800` admissions-per-48-hours targets, Terra High as the primary planner and selector, Luna High only as a capacity fallback, and guarded deterministic fallback when model paths were unavailable. Its meta-controller contract was `glee-pre-terra-one-and-half-round-controller-v2.2` with seed-stable candidate permutation.

The pinned family releases were Bargaining `v2.22-bounded-v3-low-share`, Negotiation `v2.14-buyer-surplus-and-single-counter`, and Persuasion `v2.8-bounded-no-response-routing`; rating v3 remained advisory. Hidden-identity games used the joint lexical and timing persona release `v2-joint-hi-persona`. The conditional twin and public self-mirror supplied selector evidence but held no final-action authority.

This stack is preserved as an experimental artifact, not maintained as a live service. Its modular controller, immutable snapshots, tests, models, policies, reports, and raw event journals remain available for audit and selective reuse.

## Publication and preservation boundary

The compact [GLEE competition-paper draft](../../paper/README.md) remains worth completing as a candid methods and negative-results report. Its current quantitative claims are bound to the earlier newline-terminated prefixes in `evidence.json`; the retirement frontier may be added as a clearly labeled later operational endpoint but must not silently change those frozen manuscript statistics.

The paper should state directly that the system progressed from explicit mental-state prose toward specialized control, causal opponent prediction, public self-modeling, and ecosystem-level strategy, while the competition did not isolate a leaderboard benefit from deeper RMM. That conclusion is more scientifically useful than presenting rank as validation.

No raw run material is deleted by this decision. Before any local cleanup, unique GLEE journals and model artifacts must have a verified off-device Git LFS or equivalent backup with a manifest and restore check; reproducible caches and environments remain excluded. The present stop does not itself assert that this backup is complete.

## Reuse gate

A future GLEE-derived experiment should begin from a bounded preregistered question, fixed evidence frontier, explicit control arm, predefined sample size and cloud budget, and a stopping rule independent of leaderboard emotion. Preferred designs compare transcript-only, flat persistent statistics, and recursive counterfactual modeling against matched mechanics and opponent populations, or move to richer games in which nested beliefs can alter objectively scored outcomes.

Without that protocol, the retirement decision stands: preserve the evidence, finish the paper, and redirect active research effort to NoMMD, exTSR, and the broader research program.
