# GLEE public-information self-mirror v1

**Status:** Final DeepRMM-only CPU release active in the unified v97 keepalive stack with bounded near-equivalent selector authority and prospective receipts

## Purpose

The public-information self-mirror estimates which DeepRMM-01 move a bounded observer could expect from the authenticated game prefix. It is the mirror image of the activated conditional opponent twin: the opponent twin estimates the opponent response after each candidate DeepRMM-01 move, while the self-mirror estimates the probability or numeric density that an observer would assign to each DeepRMM-01 candidate before it is submitted.

In TSR terms, this is population-level order-2 recursive mental modeling and a functional form of self-awareness: DeepRMM-01 models how a generic outside observer can model DeepRMM-01 from its public history. It is not yet peer-awareness, which would require a specific representation of how one particular opponent models DeepRMM-01. Because the self-mirror is trained against DeepRMM-01's realized historical policy rather than inaccessible opponent beliefs, its accuracy establishes the public predictability of DeepRMM-01, not that any actual opponent formed the same model; prospective residual effects remain necessary for that stronger empirical claim.

## Information boundary

Every sample ends immediately before one authenticated DeepRMM-01 move. Inputs include public game mechanics, public roles, public horizon and communication settings, mutually visible prior actions, prior messages and their semantic/style encodings, observed response timing, and quality revealed after a purchase. The target move, its message, its latency, unrevealed product quality, incomplete-information player values, internal deterministic-advisor state, statistical-package features, rating forecasts, Terra prompts or outputs, policy and engine versions, and future events are absent.

Complete-information values remain available because both players receive them. In incomplete-information games both player-value fields are masked, even though an actual opponent knows its own private value; v1 deliberately estimates the strictly public marginal expectation rather than reconstructing inaccessible private types. Hidden-identity games also mask offline account linkage, while known-identity games may use an account head because that opponent knows its own identity and interaction history.

The historical Negotiation event representation was normalized by DeepRMM-01's private value. Corpus construction reverses that normalization before masking the value and stores the public fixed coordinate `log1p(price) / 20`. Persuasion quality is retained only on explicit environment reveal events. Target messages are never reconstructed; their strategic content is represented by categorical action or numeric proposal targets where applicable.

## Model and split

V1 uses the existing compact hierarchical Mamba-2 sequence architecture. One population path models DeepRMM-01's public policy across opponents, and sparse known-account embeddings model recurrent deviations with account dropout; forced-population evaluation measures whether those heads add held-out value. Whole games retain the frozen family-stratified chronological train, validation, and test assignment from an analytics-backed all-game corpus. A post-planner response corpus is not a valid source because its inclusion rule selects DeepRMM-01 proposal or signal bridges followed by opponent responses and therefore omits roles such as Persuasion buyer; the reproducible v1 launcher requires the generic all-games corpus contract and refuses a filtered source manifest.

Categorical strategic actions, public proposal coordinates, and response delays are training targets. Checkpoint selection gives each family equal weight and combines categorical action negative log likelihood with proposal-coordinate mean absolute error when both exist, preventing Bargaining and Negotiation proposal quality from disappearing behind a categorical-only checkpoint criterion. Response delay remains diagnostic rather than a promotion objective.

No synthetic games enter v1. Synthetic policy traces could improve coverage later, but they would teach a designed policy distribution rather than what real opponents can learn from DeepRMM-01 and therefore require a separate ablation.

## Concurrent-training boundary

Training is behaviorally disconnected from the live GLEE stack and reads one immutable corpus frontier. The current first experiment uses one low-priority process, at most 2 CPU math threads, no data-loader workers, mixed precision, batch size 64, and a PyTorch allocator ceiling of 35% of the 6 GiB RTX 2060. The live conditional twin and family runners retain their existing processes, sockets, manifests, and policy authority. An out-of-memory error fails only the offline experiment and cannot replace, restart, pause, or mutate a live module.

## Evaluation and promotion

The first retrospective questions are whether the mirror beats public train-prior baselines on the chronological test suffix, whether its proposal-coordinate errors are usable, whether account heads improve over the forced-population path, and whether uncertainty is calibrated enough to compare candidate expectedness. A favorable retrospective result authorizes only a prospective shadow epoch.

Prospective shadow registration must occur before the corresponding DeepRMM-01 move and must store candidate probabilities, realized action, downstream opponent move, message class, and response delay. Raw surprise `-log P(candidate)` is not itself strategic value. V1 originally required prospective evidence before any action-authoritative promotion. The 2026-08-20 execution plan narrows initial authority to a bounded tiebreak among economically near-equivalent candidates and treats that deployment as an adaptive online experiment; evidence that expectedness adds out-of-sample information beyond the existing conditional response twin remains mandatory before stronger directional deviations or causal claims.

The eventual selector should receive expectedness as one fallible feature, not an instruction to choose the least likely move. Always selecting the least expected action creates a predictable anti-predictor and can sacrifice immediate payoff. The intended mechanism is selective opacity or strategic model interference: choose among near-equivalent moves so important latent policy variables remain difficult to infer, and use directional deviations only when their expected future benefit exceeds bounded immediate regret. The 2026-08-20 execution decision permits bounded near-equivalent opacity after offline validation while collecting prospective evidence at keepalive rate; stronger directional deviations remain disabled without evidence.

## First retrospective result

The corrected all-games cut contains 7,876 complete games, 128,737 public event rows, and 57,188 self-move targets. Persuasion is role-balanced at 15,539 seller signals and 15,840 buyer responses; Bargaining and Negotiation contain both roles and both proposal/response phases. The earlier 28,124-target post-planner derivation was rejected because all of its Persuasion targets were seller signals.

A 1-epoch concurrency canary reached 79.4% action accuracy and 0.066 public proposal-coordinate mean absolute error on the chronological test suffix. The full run early-stopped after 9 epochs and selected epoch 3, reaching 82.7% action accuracy, 0.057 proposal error, and 0.403 log-delay mean absolute error over 948 test games and 9,561 targets. Family action accuracies were 73.0% for Bargaining, 90.2% for Negotiation, and 82.8% for Persuasion; proposal errors were 0.090 for Bargaining and 0.032 for Negotiation.

A structured train-only prior that chooses the modal action and median proposal within each family, role, and target kind reached 70.7% action accuracy and 0.143 proposal error. The mirror therefore adds substantial sequential signal overall, although Negotiation categorical accuracy remains below its unusually strong 95.1% structured prior while its proposal error improves sharply from 0.169 to 0.032.

Known-account heads have small mixed aggregate effects. Their paired game bootstrap clearly improves Persuasion buyer-response loss and game-macro proposal error, weakly favors Bargaining game-macro action loss, is neutral-to-adverse for Negotiation actions, and worsens delay likelihood. They remain a shadow feature rather than a promotion premise.

Concurrent training used 365 MB peak allocated and 409 MB peak reserved GPU memory in the offline process. Total observed GPU use remained about 3.5–3.7 GiB with at least 2.3 GiB free; live conditional-twin observations remained within 28–99 milliseconds with 0 failures. The canary and full run therefore support concurrent offline training under the current envelope.

## Final DeepRMM-only rebuild and CPU runtime

The final 2026-08-20 rebuild froze 9,587 authenticated DeepRMM games at public frontier sequence 87,007 and excluded Fieldglass. The usable self-mirror corpus contains 9,572 games, 168,592 public events, and 74,687 self-move targets: 10,341 Bargaining, 22,507 Negotiation, and 41,839 Persuasion targets. The 2 independently seeded components reached 84.0–84.3% action accuracy, 0.354–0.364 action negative log likelihood, and 0.0523–0.0542 public proposal-coordinate mean absolute error on the untouched 12,414-target test suffix. The frozen release is `public-self-mirror-v1-deeprmm-final-9587-20260820`, with manifest SHA-256 `f6fcd1e715784fd230526629ac9bb237193fe66bcec3b54943264a7ae0f055f8`.

The promoted execution backend is the 1-thread pure-PyTorch Mamba-2 CPU reference, not CUDA. A full 74,687-row CPU/CUDA semantic audit found 0 component action-argmax mismatches, 0 ensemble candidate-choice mismatches, 0 displayed categorical pairwise mismatches, and 0 robust public-proposal reversals at the declared `1e-4` semantic margin. Maximum observed action-log-probability drift was `3.73e-6`, public-coordinate drift was `4.77e-6`, and diagnostic delay drift was `2.42e-5`. Direct per-family CPU inference medians were 5.82–6.30 milliseconds; the detached socket smoke over 24 requests from 12 concurrent clients had 101.7–105.8 millisecond medians, a 160.3 millisecond maximum, exact serialized registry accounting, and 0 failures.

The CPU runtime is `public-self-mirror-cpu-reference-v1.0-final-deeprmm-20260820`, with manifest SHA-256 `50e33a6ae94ad70b8076ff1383b7c7f3ae9b23840d2d5124b2e17087a6d877fb`. Its full-corpus equivalence receipt is SHA-256 `039da7da0d33d77db460532ab8020a316816a8aa429f2f6b9a21c82ce43ade11`, and its concurrent transport receipt is SHA-256 `8235728bd8a33144f719cb2d2c5511a1e6ccf004808ce72e1b08588319631ba1`. The live launcher rejects backend or runtime overrides, snapshots these receipts, forces CPU execution with 1 intra-op and 1 inter-op thread, and exposes only prospective expectedness and selection accounting through its Unix socket.

The service entered the canonical controller on 2026-08-21 at 00:30:59 UTC and was bound into unified DeepRMM run `glee-v97-deeprmm-final-9587-meta15-balanced-self-mirror-shared-w12-g48-120-20260821T004138Z`. The run manifest confirms that the planner cannot see the self-mirror, candidate generation and hard-control authority remain false, and only the final selector may consume bounded expectedness evidence. No CUDA process remained resident after activation.

The 2026-08-25 live v104 audit found a mixed-frontier transport defect rather than a model or service failure: Negotiation decision sets containing terminal `AcceptOffer` or `WalkAway` candidates alongside compound `RejectOffer+counteroffer` candidates were rejected locally because the reused opponent-response bridge required every candidate to be a counteroffer. The corrected transport preserves one immutable candidate set and one prospective forecast. Terminal candidates receive their categorical response expectedness, while each compound candidate receives the joint log expectedness of rejecting the current offer and then making that specific counteroffer. Candidate indices remain aligned across the source set, projected counteroffer subset, selector input, submitted action, and prospective selection receipt; the self-mirror remains advisory and economically bounded.

## Reproducible command

Run the immutable v1 suite from `nommd-arena` while the live stack remains under its canonical controller:

```bash
UV_CACHE_DIR=/tmp/glee-sequence-lab-uv-cache bash opponent-sequence-lab/tools/run_public_self_mirror_v1.sh
```
