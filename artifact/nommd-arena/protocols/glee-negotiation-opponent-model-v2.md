# GLEE Negotiation executable opponent model v2.0

**Status:** Retrospective shadow-only development protocol; no output is connected to live action selection

**Engine release:** `v2.0`

**Machine model identifier:** `negotiation-twin-v2.0-development`

## Selection decision

Negotiation precedes Persuasion because it provides the cleaner second test of the executable-opponent-model architecture developed in Bargaining: both families expose explicit proposals, responses, continuation, private boundaries, recurring opponents, and optional messages, whereas Persuasion adds asymmetric information, fixed 20-round trajectories, role-dependent objectives, and longitudinal reputation effects. This ordering is an engineering and evidence decision, not a claim that Negotiation requires deeper recursive mental modeling than Persuasion.

## Objective

V2.0 tests whether a structured recurring-opponent model predicts negotiation proposals and responses better than population behavior alone, whether prior games from the named opponent add signal, and whether causally earlier actions in the current game add further signal. The modeled object is the endpoint's exposed policy under recorded conditions, not its hidden prompt, operator, model family, private value, or subjective state.

## Lessons transferred from Bargaining

Prediction and action utility remain separate layers: a model can forecast an opponent accurately while a rollout or acceptance rule still loses rating, so no predictive result directly authorizes an offer, rejection, acceptance, walkaway, message, or matchmaking change. Every later live unit will require a separately sealed seed, a conservative decision interface, exact pre-network receipts, a bounded canary, and authenticated rating deltas.

The model reads only authenticated structured actions and state. Model-authored prose dossiers are excluded from fitting and evaluation because Bargaining found no aggregate predictive gain from dossier augmentation, because prose can recursively preserve semantic mistakes, and because exact actions provide the cleaner causal substrate. Dossiers remain a separate RMM channel that can be tested later under a frozen ablation.

Whole games define update and uncertainty clusters. A 99-round stalemate supplies a rich within-game policy trace but must not count like 99 independent encounters when estimating generalization; the evaluation therefore reports action-micro, complete-game-macro, opponent-macro, and paired opponent/game cluster-bootstrap results.

Repeated exact and alternating prices remain a multimodal distribution rather than being collapsed into one mean. Within-game adaptation uses only already visible transcript entries, and no target-game action enters intergame fitting before every prediction for that game has been recorded.

## Role-invariant price coordinate

Raw prices span several currencies of scale and reverse strategic direction between buyer and seller. V2.0 maps every price to `opponent_demand = direction × log(price / our_private_value)`, where `direction` is `+1` for an opponent seller and `-1` for an opponent buyer. Zero is our reservation boundary, positive values favor the opponent beyond that boundary, and negative values leave us positive nominal surplus; the transform requires only information visible to DeepRMM-01 in both complete- and incomplete-information games and is exactly invertible for positive prices.

When both reservation values are visible, the receipt additionally records the opponent's total-surplus share without clipping it to `[0, 1]`. This complete-information coordinate is diagnostic rather than a substitute target because it cannot be computed in incomplete-information games.

## Causal extraction

One negotiation history entry contains an offer by one player followed by a decision by the other. If the opponent made the offer, the predicted action is its numeric proposal and the later DeepRMM-01 decision is unavailable to that prediction; if DeepRMM-01 made the offer, the predicted action is the opponent's response and the offer plus DeepRMM-01 message are visible inputs. The current opponent-authored message is part of the proposal being predicted and cannot be used to predict its own number.

The context records role, our private value, the opponent value only when complete information exposes it, information regime, horizon visibility, visible maximum rounds, message availability, current round, the latest offer from each side, the latest response from each side, and the current DeepRMM-01 offer and message act for response prediction. Final outcome, later target actions, current opponent prose, rating delta, and future dossier synthesis are excluded.

## Initial model

The `population_kernel` transfers context-weighted behavior from causally earlier games against other named opponents and normalizes that evidence to a fixed mass. The `target_kernel` shrinks causally earlier games from the recurring opponent toward that population forecast. The primary `adaptive_v2` forecast then updates from matching opponent actions already visible in the current transcript, with explicit same-game recency and finite prior mass.

Response prediction pools `RejectOffer` and `WalkAway` as non-acceptance because opponent-attributed walkaways are too sparse for a stable competing-risk model in the current corpus. The output remains an acceptance probability with proper negative log likelihood and Brier score; exact decisions are retained so a later release can separate rejection from walkaway if new data support it.

Proposal prediction is a context- and recency-weighted Gaussian mixture in opponent-demand space. Each causally prior proposal preserves its observed mode, repeated modes accumulate mass, and a broad low-mass floor prevents a novel price from receiving effectively zero density. Mean, central 80% interval, density score, absolute error, and squared error are all retained because density improvement can coexist with poor point localization.

## Initial fixed settings

| Component | V2.0 development value |
| --- | ---: |
| Population game recency | `0.997` per intervening game |
| Target game recency | `0.90` per intervening target game |
| Same-game action recency | `0.92` per matching prior action |
| Population evidence mass | `4.0` rows |
| Population mass inside target response forecast | `2.0` rows |
| Same-game response/proposal strength | `2.0` |
| Base acceptance probability | `0.15` with `2.0` rows |
| Response demand bandwidth | `0.22` log-price units |
| Proposal mode sigma | `0.07` log-price units |
| Proposal floor sigma | `0.70` log-price units |
| Proposal floor mass | `0.35 × max(1, sqrt(effective evidence))` |

These are one recorded retrospective-development configuration, not preregistered constants. Any behavior-affecting change after inspecting its result consumes a new Negotiation minor version and records the observed failure that motivated it.

## Rolling evaluation

Games are ordered by authenticated completion time, completion order, and game ID. An opponent is eligible after at least 10 complete games, and each game after 4 earlier target games defines one origin; population evidence contains only earlier games from other named opponents, target evidence contains only earlier games from the same named opponent, and current-game adaptation receives only the visible prefix.

Micro metrics describe the encountered action distribution. Complete-game macro metrics first score each game and then average games, preventing long trajectories from controlling the result. Opponent macro metrics first score each recurring opponent and then average identities. Fixed slices separate buyer and seller opponents, complete and incomplete information, known and unknown horizons, one-round games, and decisions after round 10.

Paired uncertainty uses a fixed-seed 2-stage cluster bootstrap that resamples opponents and then complete games within sampled opponents. Primary-minus-comparator error below zero favors `adaptive_v2`; the target-kernel comparison estimates incremental within-game adaptation, while the population-kernel comparison combines recurring-identity and within-game signal.

## Interpretation gates

V2.0 is useful only if the primary model improves proper response and proposal scores beyond population transfer, adds measurable value beyond prior target games, remains directionally coherent under complete-game and opponent macro weighting, and avoids a catastrophic role or information-regime slice. A large within-game gain concentrated in 99-round loops is descriptive but insufficient for promotion.

Residual analysis must inspect response calibration, exact-price modes, role asymmetry, one-round complete-information boundaries, plateau trajectories, message-conditioned errors, sparse walkaways, and opponents for whom population transfer is harmful. A later decision layer must model the value and cost of delay rather than treating rejection as a free transition to a favorable predicted counteroffer.

## Prospective and live boundary

This release imports no GLEE client, reads no credential, creates no matchmaking request, modifies no dossier, and exposes no forecast to a live worker. Retrospective chronological replay prevents direct future-row fitting but does not prove that an earlier-completed overlapping game was available before every target action; a paper-facing prospective recorder must commit predictions from a self-contained seed before outcomes and preserve start-time concurrency where available.

The first possible live release follows this sequence: residual audit, one fixed model revision if justified, prospective shadow collection, utility-aware policy design, deterministic legality and reservation guards, latency benchmark, exact epoch seed, small canary, review, and only then a bounded rated batch. Persuasion development can begin in parallel at the corpus-contract level, but it should reuse rather than bypass these gates.
