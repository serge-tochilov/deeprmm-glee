# GLEE Bargaining live advisor v2.17

**Status:** Frozen intervention design pending deterministic replay and one prospective 20-game canary

**Machine model identifier:** `bargaining-live-advisor-v2.17`

## Scope

V2.17 inherits the complete v2.16 opponent model, analytic hierarchy, settlement policy, cloud transport, capacity fallback, and v1 prospective rating recorder. It changes only 5 linked elements: monotone projection of response curves, support-gated exact-identity package blending in offer candidate scoring, protection against post-rejection capitulation, protection against repeating a rejected extreme offer, and a narrow response guard against rating-catastrophic or unscored sub-20% settlements when rejection continuation is non-dominated.

## Evidence boundary

The 20-game v2.16 canary is development evidence for this release. Its 25 prospectively registered offer predictions showed that hierarchical identity conditioning beat the population forecast on negative log loss and Brier score and retained its negative-log-loss advantage under an opponent-cluster bootstrap, but 20 of 25 projected candidate sets contained at least one non-monotone comparison. The gate therefore supports identity conditioning at observed points but does not support unconstrained optimization over the raw sparse-bin curve.

V2.17 resolves that mismatch before granting authority. For each fixed visible decision context, population, exact-identity, and hierarchical acceptance probabilities are projected separately onto the nondecreasing cone over offered opponent share with weighted pooled adjacent violators. Raw probabilities and per-point adjustments remain in receipts; only the monotone probabilities may enter counterfactual policy scoring.

## Package authority

Package evidence may alter only an offer, never an accept-or-reject decision. The identity must resolve as one collision-free `exact-current-label`; the candidate must have exact plus weighted-nearby effective support of at least `2.0`; population-only points are ineligible; loss-minimization states are ineligible; and the candidate must remain inside the inherited policy bounds.

At an eligible point, the monotone package probability is blended with the specialized advisor probability using weight `min(0.35, 0.35 × effective_support / 5)`. Candidate value is recomputed from the inherited accepted and rejected branch values. The package candidate replaces the selected offer only when its blended expected value exceeds the selected candidate by more than `0.02` of the initial pool. This is bounded statistical authority, not replacement of the specialized response model.

## Offer catastrophe guards

After any opponent rejection, a nonterminal offer that leaves this agent below 20% of the nominal pool is replaced by the first safe candidate among the eligible package recommendation, specialized modeled offer, specialized myopic offer, and a 50/50 fallback. A nonterminal offer that repeats a previously rejected extreme allocation, defined as at most 5% or at least 80% to the opponent, receives the same correction. The replacement remains inside inherited policy bounds, does not repeat a rejected extreme when an alternative exists, and discards a paired cognitive update because the outward action changed.

## Response guards

The frozen rating reconstruction may reject an otherwise selected nonterminal acceptance only when the specialized continuation is non-dominated within `0.01` of normalized value and the accepted branch is strongly rating-negative: either its predicted self delta is at most `−5.0`, or its predicted delta and upper 80% interval endpoint are both below zero. Known-final-round and observed round-99 positive-payoff acceptance remains exempt.

When the rating forecast is unavailable because a hidden parameter prevents a valid terminal construction, a nonterminal offer below 20% nominal own share is rejected only under the same non-dominated-continuation requirement. This fallback targets the observed 8% Rubinstein settlement failure without restoring the broad fixed share floor rejected by v2.9 replay.

## Causal receipts

The package curve, blend weights, support gate, rating candidate surface, and all authority inputs are frozen before model inference. The final submission writes a separate `bargaining_v217_intervention_selected` receipt with the model proposal, outward action, activated v2.17 safeguards, selected package row, package recommendation, and response rating forecast. The unchanged v1 rating canary continues to register its own shadow prediction before network submission; v2.17 intervention receipts do not rewrite that historical contract.

## Restart state

The immutable seed corpus remains the provenance source, but normal startup does not reconstruct the advisor by re-scoring every historical Bargaining game. A first verified build compiles the seed into content-verified sufficient statistics keyed by the sealed seed identity and source-storage receipts. The ignored cache lives beside the canonical seed in `.bargaining-advisor-compiled`, making its location invariant when the executable code is imported from an immutable run snapshot. Later starts hash the canonical seed and immutable game pack, verify all implementation receipts, and restore the compiled score vectors, causal observations, game counts, adaptive expert states, and completed-game identities directly.

Each new terminal game is still written first to the append-only run journal with its authenticated final state. The record additionally carries a content-verified compiled update containing the expensive score deltas and post-game adaptive state. Restart extracts the game rows for evidence verification but applies the compiled update without rerunning the response and proposal particle populations. Missing legacy compiled updates remain readable through the historical reconstruction path; every newly written record uses the fast path.

The cache is disposable derived state, never authority by itself. A missing, corrupt, mismatched, or stale cache causes one full reconstruction from the sealed corpus and an atomic cache replacement. A seed, implementation, game-pack, dimension, causal-index, row-hash, or terminal-game mismatch fails closed. This removes the approximately 9-minute Bargaining replay from ordinary supervisor recovery while preserving the exact model frontier.

## Validation sequence

First run deterministic tests and chronological replay, treating only the first changed turn in each historical game as on-policy-prefix evidence. Then run one bounded 20-game, 5-worker canary through the canonical controller with hard `max_games=20`, unchanged 100-second model timeout plus 12-second emergency reserve, Terra High capacity chain, active statistical packages, active rating recording, and known-identity timing concealment. Review action counts, guard triggers, agreements, long games, authenticated rating deltas, model/fallback health, monotonicity, and immutable receipt reconciliation before considering any 50-game stage. No 50-game run is authorized automatically by a clean canary.
