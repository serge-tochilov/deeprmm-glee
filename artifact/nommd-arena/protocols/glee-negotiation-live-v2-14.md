# GLEE Negotiation live policy v2.14

**Status:** Implemented and replay-gated for the next controlled frontier

**Advisor engine:** `negotiation-live-advisor-v2.10`

**Policy revision:** `v2.14-buyer-surplus-and-single-counter`

## Scope

V2.14 repairs 2 independent Negotiation failures observed in the frozen v104 corpus. A complete-information buyer no longer abandons visible positive joint surplus while a legal counteroffer remains, and no longer accepts less than 1% of visible joint surplus. Separately, a one-counter candidate set no longer violates the conditional service's 2-to-5-candidate contract.

The buyer guard is deliberately narrow. It applies only when the buyer sees both reservation values, positive joint surplus exists, and the per-game pinned feature is `buyer_scale_aware_decision=one-percent-counter-first`. A proposed `WalkAway` becomes a protected counteroffer at the inherited 50% own-surplus proposal boundary when countering is legal. An `AcceptOffer` below the 1% floor becomes the same protected counteroffer, or `WalkAway` when no counteroffer is legal. Material acceptances, no-overlap states, seller decisions, incomplete-information decisions, and legacy pinned releases retain their prior semantics.

## Single-counter forecast repair

The conditional service compares 2 to 5 action-conditioned candidates and therefore cannot identify a comparative surface for exactly 1 counteroffer. Engine v2.10 now reuses that counteroffer's already frozen Negotiation family-advisor acceptance probability, assigns the residual probability to rejection, assigns no invented walkaway mass, and marks the receipt `locally-projected-single-counter`. Candidate sets containing 2 or more counters continue through the conditional service unchanged. Terminal `AcceptOffer`, `WalkAway`, and bare terminal rejection candidates remain exact terminal values without fictitious opponent replies.

## Evidence and causal boundary

The v104 review found 23 buyer walkaways, including 10 in round 1 and visible feasible-overlap cases, 2 pathological tiny positive acceptances at `$1/$499,999` and `$0.01/$2,999.99`, and 128 conditional failures caused by one-counter batches. These observed failures justify legality- and arithmetic-preserving repairs; they do not reveal the counterfactual rating outcomes of the corrected actions.

The 1% acceptance floor is a fixed scale-aware safety boundary below the existing 50% proposal boundary. It prevents effectively zero buyer surplus without forcing acceptance or claiming an equilibrium. The family-advisor single-counter projection remains prospective shadow evidence and does not gain deterministic action authority.

## Compatibility

The feature is optional under the stable live-policy contract. Legacy releases omit it and preserve old action semantics for already pinned games. Engine v2.10 also retains v2.13's one-round incomplete-information seller candidate selection, v2.9's shadow-grid correction, and every inherited reservation, concession, hardening, and stalled-exit control.
