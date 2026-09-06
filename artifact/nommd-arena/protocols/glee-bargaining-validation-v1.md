# GLEE bargaining validation v1

**Status:** Offline rolling-origin validation design; no output may affect live action selection

**Relationship to the model protocol:** This protocol evaluates `bargaining-twin-v0` under stronger causal comparisons defined by [executable opponent models v1](glee-executable-opponent-models-v1.md). It does not promote the twin or revise the live GLEE policy.

## Objective

The v0 suffix pilot showed aggregate predictive gains against an unconditional empirical baseline, but that baseline ignored the current offer and the single suffix supplied few decisions for several opponents. Validation v1 tests every eligible opponent repeatedly under whole-game rolling origins, introduces population-only, opponent-only, recency-weighted, and regularized tabular comparisons, and distinguishes negative population transfer from limitations of the complete program grammar.

## Causal frontier

Games are ordered by authenticated completion time, completion order, and game ID. An opponent becomes evaluable after 4 earlier games, provided that its final corpus contains at least 10 games. Immediately before each later target game, every model is fitted using only games earlier in that total order; all actions from the target game are predicted from the same frozen origin, and the game enters training only after every prediction has been recorded.

The population set contains all causally prior games from other named opponents, including opponents that do not meet the target eligibility threshold. The target set contains only causally prior games from the modeled opponent. This construction prevents later rows from the same game, future games, and future population behavior from entering a prediction.

Hidden-opponent games remain excluded because the current named job tree does not supply authenticated persistent identity for them. Cancelled, abandoned, and timeout terminal states remain censored under the v0 extraction rule. Every exclusion and duplicate is recorded in the corpus artifact.

## Models

### Hierarchical program posterior

The primary twin uses the population-tempered prior and opponent-specific posterior defined in the v0 protocol. It tests whether partial pooling improves sparse adaptation while preserving a posterior over compact acceptance and proposal programs.

### Population-only program posterior

The population-only model uses the same candidate grammar and population-tempered weights but receives no target-specific update. Its difference from the hierarchical model estimates the value of recurring opponent identity under a fixed grammar.

### Opponent-only program posterior

The opponent-only model starts from the grammar's complexity prior and updates only on the target's earlier games. Its difference from the hierarchical model estimates population transfer. Hierarchical performance worse than opponent-only performance is operationally labeled negative transfer; this label concerns prediction, not the cause of the failure.

### Recency-kernel empirical model

The recency model applies exponential decay to prior observations, gives target observations full unit weight, normalizes the decayed population to the same 12-row effective mass used by the program prior, and adds similarity kernels for offer share and visible context. Response probabilities use a share-distance kernel plus information-, horizon-, and role-matching factors. Proposal means and variances additionally condition on the share most recently offered to the opponent. This model tests whether local interpolation and recency explain the same behavior without a program grammar.

### Regularized tabular model

The response baseline is an L2-regularized weighted logistic regression over offered share, offered-share curvature, round progress, information regime, horizon visibility, player role, approximate parity, prior-offer availability and value, and our visible discount factor. The proposal baseline is an L2-regularized weighted linear regression over progress, information regime, horizon visibility, role, prior offers and responses, and our visible discount factor; its weighted residual variance supplies a predictive density. Population rows carry 12 units of total weight and each target row carries one unit.

### Unconditional empirical reference

The unconditional model estimates target-adapted acceptance frequency and proposal mean without current-state features. It preserves continuity with the v0 comparison but is a reference rather than the principal strong baseline.

## Prediction records

Every decision record contains the target identity, game and job provenance, causal origin index, number of prior target and population games, action type, authenticated outcome, visible context, out-of-distribution flags relative to the target's prior support, and every model's probability or predictive distribution. Program records additionally contain normalized posterior entropy. Proposal intervals in the rolling analysis use moment-matched 80% intervals; exact Gaussian-mixture likelihood remains the proper-score calculation.

## Metrics

Response metrics are negative log likelihood, Brier score, and threshold accuracy. Proposal metrics are mixture or Gaussian negative log likelihood, mean absolute error, root mean square error, and empirical coverage of the nominal 80% interval. The primary comparisons use proper scores: hierarchical versus population-only and opponent-only program posteriors, and hierarchical versus regularized tabular prediction.

Micro averages weight every decision equally and reveal total predictive performance under the observed encounter distribution. Macro averages first compute each metric within opponent and then average opponents equally. Per-opponent results remain visible because both pooling and policy stability may differ sharply by target.

Response calibration uses fixed 0.1-wide probability bins and reports mean prediction, observed acceptance frequency, count, and count-weighted expected calibration error. Proposal calibration uses interval coverage and observed-versus-predicted plots. Sparse bins are shown with their counts rather than smoothed into apparent certainty.

## Negative-transfer diagnostics

The diagnostic layer reports evidence rather than assigning a hidden cause. It calculates hierarchical minus opponent-only response NLL and proposal MAE, hierarchical minus regularized-tabular errors, out-of-distribution rate, first-half versus second-half predictive error, message-conditioned proposal residuals, action counts, and posterior entropy.

Candidate explanations use frozen exploratory thresholds: fewer than 10 evaluated actions marks sparse effective evidence; hierarchical response NLL more than 0.05 or proposal MAE more than 0.02 above opponent-only marks population negative transfer; recent response NLL more than 0.25 or proposal MAE more than 0.04 above the earlier half marks a policy-change candidate; an out-of-distribution rate above 0.20 marks context shift; and a message-conditioned proposal error gap above 0.03 with at least 3 messaged proposals marks omitted message semantics. A program error above the regularized tabular comparator under low context shift marks candidate grammar mismatch.

These labels may overlap because sparse evidence, policy change, context shift, message use, and grammar mismatch can produce the same residual pattern. They are hypotheses for inspection and grammar revision, not adjudicated properties of an opponent.

## Figures

The validator emits scalable color SVG and source CSV for 6 analyses: per-opponent response NLL differences, per-opponent proposal MAE differences, response calibration, observed versus predicted proposal shares, posterior entropy versus prior target games, and rolling predictive surprise. Negative per-opponent differences mean that the hierarchical twin is better than its named comparator. The palette separates improvement, degradation, and model families without relying on red–green contrast alone.

## Interpretation gates

Aggregate improvement is insufficient when several opponents show negative transfer. A future promotion requires positive proper-score differences against strong baselines in both micro and opponent-macro analyses, calibrated uncertainty, no recurrent catastrophic subgroup, and a prospective suffix whose predictions were committed before outcomes.

The diagnostic analysis is exploratory because its thresholds and candidate explanations were designed after the v0 pilot. A paper may report it as mechanism-generating analysis, while confirmatory claims require frozen hypotheses evaluated on later data.

## Artifacts

The offline command writes a source corpus manifest, JSON Lines prediction and origin records, aggregate and per-opponent evaluation, diagnostics, SVG figures, CSV figure data, and a run manifest that pins code and protocol hashes. The output directory must be absent or empty; an existing validation receipt is never overwritten.

The command is:

```bash
UV_CACHE_DIR=/tmp/artifact-uv-cache uv run nommd-arena glee-bargaining-validation --dossier-root opponent-dossiers/incremental-v3 --output-dir /tmp/glee-bargaining-validation-v1 --min-games 10 --warmup-games 4
```

## Live boundary

The validator reads completed immutable jobs and never creates a GLEE client, queues an agent, reads credentials, modifies dossiers, imports into a worker, or submits an action. Its results can motivate a separately reviewed prospective shadow recorder, but retrospective rolling evaluation cannot itself authorize live use.
