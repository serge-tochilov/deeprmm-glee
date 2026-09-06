# GLEE persistent RMM regime v1

**Status:** Implemented across all 3 families. Bargaining, Negotiation, and Persuasion start at 300 Poisson admissions per 48 hours with independent schedules and family-local slot pools, and each running family accepts an acknowledged live `g48` target change. Every family uses all-identity move-delay concealment, per-game hidden-identity message-style pinning, the shared transactional statistical-package overlay, bounded prompt projections, prospective rating-estimate receipts, rolling recent-game review, and atomic per-game declarative policy pinning.

## Research claim

The development loop itself is a recursive mental modeling (RMM) process. Each engine revision models an opponent policy, later revisions model how opponents react to the behavior exposed by earlier revisions, and a capable opponent can perform the same recursion against DeepRMM-01. The relevant question is therefore which side reaches a useful higher RMM order while retaining calibrated evidence, legal actions, and deadline safety, rather than whether one isolated move contains a recognizable nested-belief statement.

The motivating maHCE attributes an advantage to the combined Serge--Codex research system. Codex performs the repository work in one serialized focus, preserving the causal chain from evidence through model revision, while Serge supplies patient longitudinal supervision and social RMM through Strategic Identity Concealment (SIC), opponent activity interpretation, and adversarial hypotheses. Here *intentionality* is used in the operational sense of sustained directed focus; philosophical intentionality more commonly means the aboutness or object-directedness of a mental state and does not by itself imply a strict one-focus processing bottleneck.

This architecture does not prove a particular internal RMM order from competitive success. A policy change can improve through arithmetic correction, better calibration, or ordinary adaptation. Claims of order-2 or higher RMM require a causal account of the nested model, a prediction that differs from a lower-order account, and prospective behavior consistent with that prediction.

## Continuous development cycle

Persistent RMM replaces isolated canary epochs as the ordinary development unit. Recent completed games form a rolling evidence window; an identified defect is reproduced in chronological offline replay; a bounded repair is tested against historical controls and named regressions; and an accepted declarative release is published atomically for new games. An ordinary development cycle must not pause, stop, drain, or restart its family process: preserving the uninterrupted Poisson stream is part of both the experimental protocol and the Strategic Identity Concealment activity model.

Every game pins the release visible when its first turn is observed. Later turns of that game use the same release even if `current.json` changes, preventing one trajectory from mixing policies. Each assignment embeds the complete release and its SHA-256 digest in an fsynced append-only journal. A malformed or partially written pointer cannot affect play: the reader rejects it and assigns the last validated release to new games while recording the pointer error.

Hot promotion is intentionally narrower than arbitrary code replacement. The Bargaining, Negotiation, and Persuasion contracts admit validated scalar thresholds and preimplemented choices while preserving per-game release pinning. A candidate requiring new executable logic, schemas, prompt contracts, seed engines, shared interfaces, or safety invariants remains offline and staged while its family stream is live; it does not justify an intentional lifecycle interruption merely to deploy it. Such code may enter only at an independently necessary natural recovery boundary or through an explicitly authorized migration whose benefit exceeds the experimental and identity cost of interrupting the point process.

## Poisson admission process

Each family uses one aggregate Poisson arrival process rather than a clock per worker. Bargaining, Negotiation, and Persuasion each target 300 arrivals per 48 hours, or `lambda = 1/576 s^-1`; the exponential interarrival mean is 576 seconds and the distribution has neither a lower nor an upper bound. Each process has independent private entropy, and monotonically indexed HMAC variates make its schedule restart-stable and auditable without exposing the seed in the manifest.

Worker slots are interchangeable within a family. No Poisson clock belongs to a worker lane, so one occupied slot has no effect while any slot in that family is free. If all family-local slots are occupied, the arrival remains in a durable FIFO and is dispatched when capacity becomes available; it is never skipped, coalesced, or resampled. The scheduler records scheduled time, dispatch time, matchmaking time, queue latency, pending depth, and actual start lateness, allowing capacity-censored departures from the intended Poisson starts to be measured.

The latest 20-game v2.17 sample had a 44.7-second median observed duration and a 205.2-second mean because one game lasted about 52 minutes, implying offered concurrency near `0.36` at the current 300-game rate. Eight Bargaining slots give substantial slack, but they do not make actual starts mathematically Poisson under arbitrary saturation. Persistent monitoring must treat pending arrivals or material start lateness as a failed capacity assumption and add slots or lower the rate rather than describing the distorted process as fully independent.

Pause, deliberate stop, outage, and restart are interventions on the point process. Pause closes matchmaking, clears unstarted arrivals, and resumes with a fresh exponential wait by the memoryless property; it never emits a catch-up burst. A restart recovers a dispatched arrival whose match was not observed, but rebases an overdue unmaterialized future arrival from restart time. These common-cause interruptions are explicit receipts, are excluded when testing ordinary activity fingerprints, and must not be introduced as a routine deployment mechanism.

The canonical controller can change one running family's target with `g48 <family> <games-per-48h>`. This is also an explicit intervention: it preserves active games, pending FIFO arrivals, and an outstanding queue admission, but draws the next unmaterialized arrival afresh from the new rate at the change boundary. The scheduler stores the startup target separately from the effective target so a process recovery preserves a live override without mutating the immutable manifest; a no-op target request does not resample time, and a paused-family change is realized through the ordinary fresh wait on resume.

Rates specify offered admissions, not guaranteed completed games; realized completions depend on game duration, matchmaking latency, pauses, capacity, and terminal reconciliation. Separate family supervisors prevent one family's long game or queue depth from changing another family's sampled schedule.

## Cross-family evidence path

All 3 family supervisors read the same manifest-pinned statistical-package base and update one transactional live overlay after every terminal game. The immutable raw package remains upstream; the model prompt receives only a bounded current-turn projection of the opponent's likely next move, with identity resolution, direct and population support, and no independent legal or deterministic authority. This avoids duplicating a large cross-game package in each prompt while retaining an exact receipt for later audit.

Negotiation retains its frozen v2.4 displayed-rating surrogate. Persuasion v2.3 adds a scale-normalized ridge surrogate fitted with fixed regularization and evaluated on a final chronological 20% suffix; it reports pass, buy-low, buy-high, and role-appropriate expected branch estimates. Non-final Persuasion estimates are explicitly stop-now diagnostics rather than continuation values. Both families register the exact rating estimate in the event ledger before model inference, so authenticated postgame deltas can evaluate calibration without reconstructing a prediction after the outcome.

The first persistent cross-family epoch pins 1,868 authenticated Negotiation games and 937 authenticated Persuasion games. Negotiation's visible-information rating model records a 148-game chronological holdout with MAE 2.1738, correlation 0.7771, and sign accuracy 0.6824; Persuasion records a 187-game holdout with MAE 2.6986, correlation 0.6543, and sign accuracy 0.7005. These are retrospective promotion receipts, not randomized estimates of live policy value.

Statistical packages and rating estimates are different signals. The package predicts opponent behavior conditional on visible context; the rating surrogate estimates DeepRMM-01's displayed score under candidate terminal branches. Neither may override authenticated state, payoff dominance, family-specific causal models, legal-action normalization, deterministic guards, or deadline safety merely because it is present in the prompt.

## Concealment scope

The first persistent epoch activates every concealment channel already implemented and regression-tested: stochastic activity timing applies before identity is revealed, and randomized 14--46-second total move latency applies to both known-identity and hidden-identity games subject to the 12-second deadline reserve. This removes the earlier `known-only` timing asymmetry.

The next persistent epoch adds coherent per-game message-style concealment. KI messages retain the native renderer; the first message-bearing HI action selects and pins a family-specific renderer whose lexical surface is anti-aligned with the economic style of the actual action, and later turns preserve that persona while allowing the current semantic act or Persuasion polarity to change. The transformer never changes numeric action fields, never creates a message where the action was silent, and never overrides legality, deterministic guards, or economic policy. Numeric-action mimicry and action-level identity routing remain offline.

Opponent language is also an inward signal. A corrected chronological test conditioned on visible state, opening action, and represented counterparty stimulus found small untouched-suffix gains from lexical features for Bargaining and Negotiation next-action prediction, while Persuasion remained unvalidated. The live prompt therefore receives only a compact, identity-free Bargaining or Negotiation forecast shift with explicit soft-prior authority; it does not name an opponent and cannot override analytic or calibrated controls. The private-archive evidence receipts are `glee-message-style-correlation-v1-20260813` and `glee-message-policy-signal-v1-20260813`.

## Monitoring and improvement

Operational monitoring reads recent terminal games, scheduler receipts, policy assignments, rating synchronization, statistical-package revisions, model-call outcomes, timeouts, deterministic fallbacks, and memory use. It does not wait for a fixed canary boundary. Reviews report results by pinned policy revision and preserve chronological order, because pooling an old and new release can conceal regressions.

The minimum live checks are pending-arrival depth, scheduled-to-start lateness, active-slot occupancy, completion rate over 48 hours, endpoint and synchronized game-level rating changes, role and identity-mode splits, policy intervention counts, malformed or fallback calls, and long-game tails. A release is rolled back by atomically restoring a prior validated pointer; already active games remain on their pinned revision.

The 300-per-family startup targets are experimental operating rates rather than promises to maintain that density. A target can be raised or lowered live, or a family can be paused or stopped, when quota, ratings, platform rules, machine health, or new evidence changes the risk calculation. The GLEE controller remains the sole lifecycle and live-rate authority.

## Reproducibility boundaries

The process is adaptive and therefore is not one frozen confirmatory experiment. Scientific use must preserve immutable game records, policy-release files, assignment journals, scheduler events, source snapshots, model receipts, and the exact evidence frontier supporting each promotion. Confirmatory analyses require separately frozen hypotheses and untouched suffixes; continuous competitive improvement supplies engineering and longitudinal RMM evidence.
