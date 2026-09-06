# GLEE account-level hidden-identity and performance analysis v1

**Status:** Frozen offline design, 2026-08-14; no account-level inference or account-specific KI/HI result had been calculated when this protocol was written. The analysis starts no matchmaking, sends no model calls, changes no live policy, and gives no online routing authority.

## Questions

This analysis asks 2 questions. First, how accurately can the already-frozen causal identity model identify a candidate owner account rather than one specific public agent after seeing the public presence state available at game start and exactly one opponent move? Second, what does that calibrated but uncertain account signal suggest about DeepRMM-01's rating performance against candidate accounts in known-identity (KI) and hidden-identity (HI) games?

## Frozen sources

The identity model is `glee-causal-identity-routing-v1` at public frontier `18108`. Its causal observation point, train/calibration/test split, profiles, galleries, feature grammar, and fitted full-presence-plus-behavior stacker remain unchanged. The account partition is `reports/glee-account-linkage-v1-20260814/groups.json`, which contains 23 non-overlapping candidate accounts and 95 public IDs. Outcomes come from the immutable Polars lake `glee-analytics-lake-v1/frontier-36748`, but eligible inference and performance games must complete no later than frontier `18108` at `2026-08-12T20:44:14.708316+00:00`.

The account partition was reconstructed after the identity-model frontier from public owner-best, naming, volume, and activity evidence. It is therefore a retrospective account hypothesis, not information shown to the original model at game time and not proof of common ownership. Results must be stratified by linkage confidence where support permits.

## Masked-KI account evaluation

The untouched 379-game causal identity test suffix contains exact temporal public IDs from KI games. The analysis reconstructs every full agent posterior from the frozen sources and verifies its true probability, ranking, confidence, and stored top candidates against the sealed receipt before account aggregation.

For every game, probabilities of public IDs assigned to the same candidate account are summed. The original `unknown` mass and every unlinked public ID are summed into one `unlinked-or-unknown` class. This is a deterministic coarsening of the frozen posterior; no account-specific feature, weight, threshold, or test-selected calibration is introduced.

Primary account metrics are top-one accuracy, top-3 accuracy, mean reciprocal rank, negative log likelihood, and multiclass Brier score. They are reported for all account/open-set targets, for linked-account targets, by family, and by account. A paired comparison with exact-agent inference uses only games whose exact agent was a named frozen target and whose ID belongs to a candidate account. Named-account coverage and precision are reported across fixed confidence thresholds without promoting a threshold.

Because all validation labels come from KI games, this measures masked-KI account identification. It estimates HI identifiability only under the unverified assumption that an opponent's first-move behavior and public activity relationship are not changed materially by whether names are disclosed.

## HI account inference

The unchanged frozen model is applied to every HI game in the analytics lake that completed by frontier `18108`, has a hash-addressed terminal archive, and exposes at least one opponent move. The public prior is frozen at `started_at`; behavior contains exactly the first opponent-attributed move. Later moves, terminal results, rating deltas, disclosed names, postgame public pulses, and account outcomes are excluded from inference.

Each HI receipt stores the account posterior, top account, confidence, source archive hash, and frozen model/account-map identities. An account posterior is an inference distribution, not an authenticated assignment.

## KI versus HI performance estimation

KI account outcomes use exact contemporaneous public-ID resolution from the frozen analytics lake and only non-ambiguous candidate-account mappings. HI outcomes retain the complete account posterior. For each account and family, the analysis reports exact KI game count and mean rating delta, HI posterior mass, posterior effective sample size, posterior-weighted mean rating delta, and their difference.

The soft HI estimate is descriptive shrinkage, not a causal or unbiased latent-class estimator: diffuse probabilities pull every account toward the overall HI mean. Fixed named-account confidence cuts provide a smaller hard-assignment sensitivity view, with masked-KI precision reported alongside it. No outcome is allowed to update the identity posterior.

An adjusted sensitivity analysis residualizes rating delta against variables observed independently of the hidden account: family, role, engine version, time trend, DeepRMM rating, imputed opponent rating, and the rating-imputation indicator. Identity mode and account label are not controls. Uncertainty uses deterministic time-block bootstrap resampling and is reported only where exact KI support, HI posterior mass, and posterior effective support are adequate.

## Interpretation boundary

The analysis can show that sibling-agent pooling improves identity resolution and can identify candidate accounts for which KI and probabilistically attributed HI outcomes differ. It cannot prove the true owner of an HI opponent, distinguish multiple policies on one account, establish why KI and HI outcomes differ, or authorize account-specific live play. Family, role, engine, matchmaking, identity disclosure, and strategic adaptation can remain confounded even after adjustment.

All artifacts remain offline. Any later online use requires a fresh prospective gate with account mapping available before the game, causal posterior registration, calibrated decision value, and a separately reviewed policy contract.
