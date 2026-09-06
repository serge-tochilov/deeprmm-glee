# GLEE Bargaining prospective rating canary v1

**Status:** Implementation protocol for a 20-game Bargaining shadow canary; it changes no submitted action and grants no rating model live authority.

## Question

The canary asks whether the frozen exact-configuration rating model remains calibrated on causally fresh Bargaining games and whether its estimate of an opponent's displayed-rating consequence supplies useful information for later action selection. It separates 3 claims: predicting DeepRMM-01's terminal displayed-rating delta, predicting a collision-free known opponent's terminal displayed-rating delta, and ranking candidate acceptance branches before the game outcome. Success on one claim does not imply success on the others.

## Frozen model and causal frontier

The predictor is the prospective Bargaining model stored in the hash-verified `glee-joint-rating-reconstruction-v2` artifact fitted through terminal cutoff `2026-08-12T18:31:08.569568+00:00`. A sealing command extracts only its executable Bargaining state into a portable self-hashed Git-LFS seed that also pins this protocol and the canary implementation. Its structural adjusted-percentile regression, exact-configuration empirical payoff ranks, shrunken configuration residuals, learning-rate schedule, and retrospective interval offsets remain frozen for the entire canary. Statistical opponent packages continue their independent per-game live updates, but no canary observation refits or recalibrates the rating predictor before the canary closes.

Every game captures one immutable pregame context at its first observed turn. DeepRMM-01's displayed family rating and game count come from the causal authenticated sensor frontier. A known opponent's rating and game count come from the public Bargaining reporter frontier only when the statistical package resolves the displayed label to exactly one stable public ID, that ID is present in the latest successful public poll, and the frontier is fresh. Hidden identities, colliding labels, absent rows, truncated stale rows, and stale reporter frontiers receive an explicit unavailable opponent-rating state rather than a guessed top candidate.

## Mandatory statistical packages

The canary may start only with the collision-safe statistical-package base and `glee-opponent-statistical-live-overlay-v1` active. Startup aborts unless the package reader exposes the verified base manifest and a durable live-overlay revision and state hash. Every pregame and turn receipt pins the package release, base manifest, identity-resolution status, overlay revision, and overlay state hash. Missing package context or a missing live-overlay receipt is a configuration failure, while an individual rating-shadow calculation failure is recorded and leaves the already selected legal move unchanged.

## Hidden-identity treatment and SIC

Strategic Identity Concealment is disabled for hidden-identity games in this canary. Hidden games use the ordinary canonical DeepRMM policy, prompt, wording, and natural response latency; the supervisor applies no artificial move-delay concealment to them. Known-identity games may retain the existing bounded delay policy, but no canary rating field enters the model prompt, deterministic guards, action normalization, or submission choice.

This restriction keeps the hidden-game behavior comparable with the frozen corpus and prevents a simultaneous identity intervention from contaminating the rating intervention. It does not identify a hidden opponent, route a named package into an anonymous game, or turn family-population evidence into an identity claim.

## Turn-time shadow surface

Immediately after the ordinary worker and existing Bargaining advisor choose an action, but before any network submission, the canary registers the selected action and a bounded acceptance-branch surface. An offer turn evaluates the submitted split, the frozen advisor's canonical share grid, and its modeled offer when distinct. Each candidate records the existing behavior model's acceptance probability, the accepted terminal payoffs after player-specific round discounting, predicted self and opponent displayed-rating deltas where their pregame states are available, exact-configuration support, residual support, and 80% and 95% diagnostic intervals.

The canary records a myopic rating shadow recommendation and the ordinary behavior-and-payoff recommendation side by side. The myopic rating score treats non-acceptance as zero immediate rating change and is therefore not a full continuation value. It is a diagnostic for whether rating information would change candidate ordering, not a licensed policy. A decision turn records the rating consequence of accepting the visible offer; rejection receives no invented terminal value unless the authenticated state makes it terminal.

## Terminal forecast and maturation

As soon as a terminal state is observed, the canary registers terminal self and opponent forecasts before reading any newly synchronized rating target. The terminal receipt hashes the final state, pregame context, frozen model, package frontier, and point and interval forecasts. It remains valid when opponent prediction is unavailable, provided the self forecast is registered.

Authenticated self-rating deltas mature automatically from the read-only history synchronizer only after their game ID appears in a synchronization frontier observed later than registration. Opponent deltas mature only through a later collision-safe public-event assignment; a change in the same public row is not automatically attributed to this game because the opponent may have played overlapping games. Registration and maturation tables are append-only, conflicting retries fail, and exact retries are idempotent.

## Authority and safety

The canary is shadow-only. It does not alter worker input, model choice, effort, timing budget, candidate generation, deterministic arithmetic, legality checks, timeout safeguards, submission, opponent package updates, or family-advisor updates. Existing legal and catastrophic-loss guards retain their authority. A canary error may invalidate one shadow record but must not sacrifice an active game or replace its selected move.

No conclusion may claim that an opponent consciously models GLEE's rating system. A useful opponent-delta feature can arise because competent payoff-seeking policies correlate with the same score surface.

## Initial run and promotion gate

The initial canary admits exactly 20 Bargaining games with 5 workers. The report must state registration completeness, package and overlay continuity, KI/HI and collision counts, public-rating availability, configuration support, self-delta MAE, RMSE, sign accuracy, interval coverage, opponent-target maturation count, frequency with which rating and ordinary recommendations differ, realized acceptance by predicted opponent-delta band, latency overhead, and every excluded or failed record.

Promotion beyond shadow requires all registered predictions to precede target observation, zero package-contract violations, zero action changes caused by the canary, complete self-target maturation except explicitly censored platform rows, no material timeout or latency regression, and prospective self-delta accuracy that remains useful relative to zero change. Opponent rating may influence Bargaining action ranking only after enough clean opponent targets mature to evaluate calibration and after accept-versus-continuation value is implemented; Negotiation, Persuasion, hidden-identity exact routing, and rank-aware opponent harm remain outside this canary.

## Storage

The run-local SQLite registry is the canonical canary receipt. It stores immutable pregame contexts, pre-submission turn shadows, terminal forecasts, and later maturations without copying terminal game archives or public reporter history. The ordinary run manifest pins the model and protocol hashes, while terminal archives, authenticated history, reporter changes, and statistical-package overlay remain independent recovery and audit sources.
