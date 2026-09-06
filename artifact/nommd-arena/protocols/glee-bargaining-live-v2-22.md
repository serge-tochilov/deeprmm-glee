# GLEE Bargaining live policy v2.22

**Status:** Sealed on 2026-08-25 as a stopped v103 frontier; this document does not authorize matchmaking.

## Purpose

V2.22 adds one narrow action guard for a nonterminal acceptance that gives DeepRMM less than half of the current allocation while behavioral rejection remains non-dominated and rating v3 predicts a bounded loss. It addresses 3 round-1 acceptances at 35% own share in the drained v102 window without restoring the older response-rating v1 authority.

## Authority contract

The guard may replace `accept` with `reject` only after all of the following hold: the current action is an acceptance decision; own share is below `0.5`; the behavioral continuation comparison places rejection within the live continuation tolerance of acceptance; the current-offer rating-v3 acceptance branch is available; its predicted self-rating delta is below `0`; and the upper endpoint of its 80% interval is also below `0`.

Any missing, malformed, point-positive, or interval-crossing rating evidence fails open to the previously selected action. Rating v3 does not create an offer, choose a proposal, or authorize a rejection when behavioral continuation is materially worse. The legacy response-rating-loss guard remains `shadow-only`.

## Evidence boundary

The v102 window contained 3 35% acceptances that satisfy the complete contract, with predicted deltas and 80% intervals wholly below zero. A distinct 42% complete-information acceptance is preserved because rejection had lower modeled continuation value and its rating interval crossed zero. No threshold was fitted to retrospective terminal outcomes beyond the predeclared half-share boundary and bounded-loss condition.

## Replay

The complete 23-game v102 Bargaining replay changed exactly those 3 first decisions and left the other 62 submitted turns unchanged. This proves narrow mechanical scope, not counterfactual rating gain; all post-divergence outcomes remain unknown.
