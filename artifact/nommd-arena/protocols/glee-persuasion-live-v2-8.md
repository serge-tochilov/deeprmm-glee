# GLEE Persuasion live policy v2.8

**Status:** Implemented and regression-tested on 2026-08-26; live activation requires the v107 controlled process boundary.

## Scope

V2.8 preserves every v2.7 buyer decision and every responsive seller branch. It changes only seller-message and seller-recommendation turns after the current game has reached the existing no-response diagnostic: at least 4 visible passes and a Beta-smoothed buy rate no greater than `0.2`. Bargaining and Negotiation remain unchanged.

## Bounded deterministic route

Once the no-response boundary is crossed, the seller uses a quality-consistent signal: positive for a high-quality item and negative for a low-quality item. Binary and text actions are normalized through the same legal-action machinery, recorded as `persuasion-v2.8-no-response-authority`, and submitted without planner, conditional-twin, self-mirror, or selector calls.

The boundary is trajectory-local rather than an opponent-type claim. It says only that 4 or more visible passes have made another 2-call wording search unsupported for the current game; it does not claim that a buyer can never resume purchasing, that the buyer is irrational, or that the quality-consistent action is globally optimal.

## Hot-policy compatibility

The optional `seller_no_response_routing` feature accepts `advisory-only` or `deterministic-quality-consistent`. Older v2.7 releases omit the feature and retain advisory-only behavior, so the code path is reversible by atomically restoring the prior policy pointer. Every newly admitted game pins exactly one release; an active game never changes policy midway.

## Objective and evidence boundary

The official organizer clarification states that an agent's rating is based on its own payoff, the exact game configuration, and opponent rating rather than the ratio of player payoffs. V2.8 therefore treats opponent payoff or damage only as evidence about future behavior and never as independent utility. Discord participant reports about heuristic stonewalling and cold buyers motivate diagnostics but remain low-authority external observations, not mechanics or action authority.

## Timing boundary

The accompanying v107 supervisor uses `hidden-only` timing concealment. Hidden-identity games retain the established joint lexical and timing persona because concealment has an identity purpose there; known-identity games submit as soon as computation and API spacing allow because delaying a visible identity cannot conceal it and consumes shared worker capacity. This is an efficiency cutover, not evidence that legal wall-clock response time directly changes rating.

## Receipt and paper boundary

Paper-facing analysis must separate v2.8 bypass turns from Terra-selected turns, report avoided model calls and worker occupancy, and compare complete games only within immutable policy epochs. Historical replay can verify activation, legality, and first divergence but cannot assign the later observed outcome or rating of a counterfactual trajectory.
