# GLEE rating v3 dual-channel advisory v1

**Status:** Activated in all 3 canonical family modules under the reviewed v34 epoch on 2026-08-15; manifests, startup reconciliation, and the first Bargaining terminal registration were verified separately from code and configuration publication.

## Purpose

The adapter promotes rating model v3 from operational nonexistence to model-visible advice without granting it deterministic action authority. Before Terra sees any v3 estimate, the supervisor stores the exact pregame context and turn advisory in an append-only registry. Terra then receives a compact projection of the same record under `rating_v3_advisory`. This dual channel preserves prospective evaluation while allowing the model to use the estimate immediately.

Feeding a forecast to Terra means it is no longer a pure shadow with respect to play. It remains a shadow outcome record because its prediction, model hash, causal frontier, and submitted-policy context are frozen before the rating target becomes observable. The correct label is therefore **prospectively sealed advisory**, not shadow-only.

## Causal pregame state

Each game captures DeepRMM-01's displayed family rating and completed-game count from the authenticated sensor frontier, falling back only to the supervisor's previously authenticated stats cache. A collision-free known opponent may receive a public pregame rating from the fresh public reporter when the statistical package resolves the current label to exactly one stable public ID. Hidden identities, label collisions, stale reporter data, and absent rows remain unavailable rather than being replaced by a likely identity.

The current public family median enters the residual context when available. The 300-second and 1,800-second traffic and fleet features are explicitly unavailable in v1 and retain their model defaults. Their absence is recorded in every game context rather than silently described as measured zero traffic.

## Family branch projections

Bargaining offer turns score a compact split surface as immediate agreement branches and combine the existing opponent model's conservative acceptance probability with the predicted self-rating delta as a separately labeled myopic diagnostic. Decision turns score acceptance of the authenticated current offer. Rejection receives a terminal no-deal score only at a known final round; otherwise its continuation is explicitly omitted.

At discounted rounds under incomplete information, the opponent's discount factor is semantically hidden even if an internal transport object happens to contain a value. The advisory integrates every accepted branch over the empirically observed platform support `{0.8, 0.9, 0.95, 1.0}` with equal weights, reports the resulting mean and range, and never promotes one guessed discount. The target player's own visible discount remains exact. This repair replaces the v104 failure mode in which 344 of 959 Bargaining turn advisories raised `Bargaining candidate cannot invent a hidden discount factor`; the strict terminal constructor itself remains unchanged and continues to reject an unqualified hidden value.

Negotiation always exposes a no-deal branch. Agreement branches are scored only when both reservation values are visible, because v3 uses both players' terminal payoffs and must not invent a hidden value. Candidate prices come from the authenticated current offer, the existing executable advisor's private shadow surface, and applicable deterministic concession bounds. Rejection and future-round continuation remain omitted.

Persuasion exposes stop-now pass, buy-low, and buy-high branches. Buyer advice combines the low/high rating branches with the existing seller-reliability posterior. Seller advice combines the branch deltas with the existing buyer-response forecasts for positive and negative signals. Before the final round these are stop-now diagnostics, not full continuation values.

## Authority

Terra may use v3 evidence together with payoff arithmetic, opponent-response forecasts, and the official game state. V3 has no direct access to action normalization, deterministic guards, timeout fallbacks, submission, matchmaking, identity routing, lexical realization, or timing concealment. Legality, deadline, and catastrophic-loss controls remain authoritative.

The prompt explicitly warns Terra not to optimize an immediate branch as though an omitted continuation had value zero. A model-selected action can nevertheless change because Terra saw the advisory; that is the intended intervention and must be identified as such in analysis.

## Prospective terminal record

At the first local terminal observation, the supervisor predicts the realized self-rating delta and appends it before reading the authenticated history target. It then inserts the observed configuration and payoff into the process-local causal rank index. A restarted process replays every registered terminal forecast from the shared registry exactly once, allowing later queries to use only earlier terminal observations without copying game archives.

The history reconciler matures a forecast only when the immutable game ID appears with a rating delta observed later than forecast registration. The raw dashboard value remains in the immutable outcome row. A separate append-only correction row may replace it for performance evaluation only when the versioned correction manifest pins the exact history receipt and a one-game authenticated sensor transition; exact retries are idempotent, conflicting records fail, and SQLite WAL plus append-only triggers protect concurrent family supervisors. The registry does not rewrite the frozen model release.

Each rebuilt rating-model release writes to a distinct versioned registry because registry metadata binds every prospective row to one immutable model hash and causal cutoff. A model cutover must preserve the prior registry and start a fresh registry for the replacement release; reusing the old path correctly fails closed with a metadata conflict rather than mixing forecasts from different models.

An outcome first observed at or before forecast registration is causally ineligible rather than a training or evaluation target. The reconciler appends that status to a dedicated immutable quarantine and continues to later forecasts; one ineligible row must never abort the full reconciliation pass or repeatedly prevent newer outcomes from maturing.

Inactivity-decay contamination is distinct from causal ineligibility. A forecast may have been validly sealed before its outcome while the dashboard later attaches accumulated pre-game decay to that completion. Such an outcome remains causally eligible, preserves its gross value, and uses the separately evidenced game effect in prospective metrics; it must never be silently clipped or mislabeled as an ordinary loss.

## Operational audit on 2026-08-16

The live v81, v82, and v84 family processes were found repeatedly aborting reconciliation on one causally ineligible forecast, although their move paths and rating-v3 prompt advisories continued to run. This did not explain the contemporaneous rating drop because the frozen advisory model does not consume newly matured deltas online, but it blocked prospective calibration evidence. The repaired registry behavior passes a regression in which an early outcome is quarantined and a later valid forecast matures in the same pass. Deployment requires the next controlled family restart; the running processes do not hot-load Python implementation changes.

## Activation boundary

The implementation adds optional `glee-parallel` paths for the frozen model root, shared registry, public reporter, authenticated history database, and versioned rating-effect correction manifest. All 5 paths are required together so immutable run-snapshot code never infers the live manifest from its own filesystem location. Merely merging this code does not change a running family: activation requires a new control-module specification and controlled restart, producing supervisor contract `glee-parallel-v34`.
