# GLEE causal identity fusion and predictive routing v1

**Status:** Frozen offline design and completed baseline, 2026-08-12; no causal-fusion or routing result under this protocol had been inspected when it was written. The analysis starts no matchmaking, sends no model call, changes no live prompt or action, publishes no dossier, and supplies no online identity or routing authority.

## Objective

The retrospective Stage 6 fusion showed that behavior improves identity ranking inside postgame activity candidates, while Stage 3 produced a strong causal presence forecast but only a narrow 60-second identity-ranking gain. This experiment combines the frozen 60-second pregame presence prior with the first opponent move, gives `unknown` representative natural support, and tests whether the resulting identity posterior usefully routes an identity-specific action model for the next opponent move.

The routing target is predictive: an identity-specific package must predict the next opponent action better than a generic population package. This is a necessary test of routing value, not an estimate of the payoff of an unobserved counterfactual action. No result can authorize live strategy changes without a later economic decision-value and prospective-shadow experiment.

## Frozen sources

The presence source is `glee-causal-presence-forecast-v1` at frontier `18108`, manifest SHA-256 `283efba4956e5af5c86cfa58963ecef3c78c504260ebbb92dcb761aa7decb7f7`. The behavior source is `glee-collision-safe-behavior-corpus-v1`, manifest SHA-256 `206e0e0f8ea2b2a2898e7775f2af36c9ebcf5a9697bc63e0e435ecc7eceba1b3`. Channel bins and hyperparameters come from `glee-behavior-channel-evaluation-v1`, manifest SHA-256 `aebf601db25e0383781293e7a50a72ee2320d2d4b65b5fe2f58d822c4043cbaf`. The corrected event cache, temporal identity registry, reporter frontier timestamps, and source hashes must reproduce the Stage 3 receipt before fitting.

The analysis reuses the Stage 5 family-specific 60/20/20 chronological whole-game split and its overlap purges. Presence models and feature definitions are loaded from the completed Stage 3 receipt rather than refitted or retuned. The stored Stage 3 test posteriors must reproduce exactly; calibration-game priors are reconstructed from the same frozen model and causal public evidence because Stage 3 did not serialize them.

## Causal observation point

The identity query occurs immediately after the first observed opponent move. The public prior is frozen at game `started_at`, so no public pulse observed after assignment enters the prior. Behavioral evidence contains exactly the first opponent-attributed move from the collision-safe corpus. Later opponent moves, terminal outcome, disclosed label, later public activity, rating delta, dossier text, and postgame statistics are unavailable.

Action, timing, lexical, and discourse tokens use the frozen Stage 5 feature grammar. First-move snapshots omit the whole-game `terminal-style` token. Sequence tokens cannot arise from one move. Profiles used on calibration games fit only training snapshots; profiles used on test games fit training plus calibration snapshots, whose games complete before the test boundary.

## Representative `unknown`

The family gallery contains every public ID with at least 3 complete training games, regardless of whether a particular channel has evidence. The frozen galleries contain 56 Bargaining, 50 Negotiation, and 14 Persuasion identities. A calibration or test target is known only when its public ID is in the training gallery and in the causal public candidate set; otherwise its target is `unknown`.

For each pregame prior, gallery candidates retain their individual mass. The fixed Stage 3 5% reserve and all mass assigned to non-gallery public candidates are summed into one `unknown` class. This produces natural open-set cases rather than synthetic identity withholding: the untouched suffix contains 66 Bargaining, 44 Negotiation, and 21 Persuasion unknown targets, 131 of 379 games in total. These cases represent sparse or unseen identities and causal candidate misses; they do not fully represent an agent absent from every public board.

The uniform baseline first assigns equal mass to every causally present public candidate under the same 5% base reserve, then performs the same gallery collapse. Static-rate and full 60-second priors use their frozen Stage 3 probabilities. No post-result unknown threshold or reserve is selected.

## Fusion models

The fixed arms are `uniform-open-set`, `static-presence`, `full-presence`, `uniform-behavior`, `static-behavior`, and `full-behavior`. Raw presence arms require no fitting after gallery collapse. Each behavior arm is a nonnegative conditional log-linear stacker over log prior, timing, action, lexical, and discourse scores, with ridge `0.1`, 300 full-batch Adam iterations, and learning rate `0.03`. Channel alpha and score temperature are inherited family-by-family from Stage 5; the fusion stacker fits one unknown intercept on calibration games.

The untouched metrics are top-one and top-5 accuracy, mean reciprocal rank, NLL, multiclass Brier score, calibration error, known-conditional ranking, unknown recall and precision, predicted-unknown count, family and role breakdown, and confidence/abstention curves. `Full-behavior` passes the causal-fusion gate only if it improves both NLL and Brier over `full-presence` and `uniform-behavior`, improves known-conditional top-5 or MRR over `full-presence`, and does not reduce unknown recall by more than 5 percentage points relative to `full-presence`.

## Predictive identity routing

Routing is evaluated only on calibration and test games with at least 2 opponent moves. The identity posterior is computed from the pregame prior and first opponent move; the second opponent move is the untouched routing target. This leaves 59 Bargaining, 81 Negotiation, and 40 Persuasion test games before any target-token exclusion.

An action package is a family-specific hierarchical token model fitted to individual opponent moves from causally earlier complete games. It predicts the frozen action tokens for one next move, excluding `terminal-style`; out-of-vocabulary evidence maps to one explicit token. The population model is the generic route. Each gallery ID receives a population-smoothed package using the Stage 5 action alpha. `Oracle` selects the true identity package for a known target and generic for an unknown target. `Soft route` mixes all identity packages and generic by the causal identity posterior. `Hard route` selects the maximum-posterior named package only when it outranks `unknown`; otherwise it uses generic. No confidence threshold is selected from test data.

The primary routing score is per-game mean next-action-token NLL. Secondary scores are known and unknown NLL, family NLL, hard-route coverage, wrong-route excess loss, and the fraction of the oracle-versus-generic gap recovered by soft routing. Uncertainty uses 5,000 deterministic opponent-cluster bootstrap replicates with seed `20260812`; the point estimate remains game-weighted and the interval resamples identity-level mean paired differences.

Predictive routing passes only if the oracle package improves over generic and soft `full-behavior` routing improves over generic with the 95% paired interval entirely below zero. The unknown subset may worsen by at most `0.02` NLL. Failure of the oracle comparison means the package itself lacks useful identity specificity; failure of soft routing with a useful oracle means identity inference or posterior mixing is the limiting layer.

## Interpretation and promotion boundary

The experiment separates 3 questions: whether causal evidence identifies a gallery member or `unknown`, whether an identity-specific behavioral package contains future-action information, and whether an uncertain posterior can route that package without losing its value. It does not infer ownership, prove one stable internal policy, or recover an opponent's actual code.

Even a passed predictive-routing gate cannot establish economic decision value because only one historical action and response were observed. The next gate would freeze a family-specific planner or prospective shadow policy, register generic and routed recommendations before outcomes, and evaluate payoff or rating utility with deterministic safety guards. Live identity routing, named-dossier injection into hidden games, and Strategic Identity Concealment remain prohibited under v1.

## Artifacts and reproducibility

The implementation must emit compact source/model summaries, gallery inventories, calibration and test identity predictions, next-action routing predictions, bootstrap intervals, and a manifest hashing every artifact and implementation file. Large candidate-level posterior rows remain ignored and reference immutable source artifacts rather than copying them. A second clean run must reproduce every artifact byte-for-byte before the report is accepted.

## Completed result

The completed private-archive causal identity-fusion report `glee-causal-identity-routing-v1-20260812` verifies exact Stage 3 prior parity, evaluates 131 natural unknown targets among 379 test games, and passes the frozen causal-fusion gate. Full presence plus first-move behavior improves NLL from `2.991` to `2.815`, Brier score from `0.871` to `0.826`, known-conditional top-5 accuracy from 27.02% to 34.68%, and known-conditional MRR from `0.180` to `0.259`, while preserving 100% unknown recall.

The routing result isolates the remaining failure. A true-identity oracle lowers next-choice NLL from `1.0467` to `1.0188`, with opponent-clustered 95% difference interval `[-0.0487, -0.0070]`, so identity-specific action packages contain future information. Soft causal routing instead reaches `1.0482`, with interval `[-0.0025, +0.0059]` relative to generic, and fails the predictive-routing gate. The generic route therefore remains the safe baseline; no identity posterior or package receives live authority.
