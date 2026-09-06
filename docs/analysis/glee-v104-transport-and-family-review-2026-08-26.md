# GLEE v104 transport incident and family review

**Status:** The v104 unified runner drained cleanly on 2026-08-26 after 799 games: 269 Bargaining, 254 Negotiation, and 276 Persuasion. The quantitative review freezes the first 797 games, before the final 2 long Negotiation games completed; transport v2 is implemented, tested, and resealed as the stopped v105 frontier. No family relaunch is authorized by this review.

## Transport incident

At 08:08:26 EDT, a move POST raised a response-side transport error and escaped the shared supervisor; the 20-worker process ended at 08:08:46. Admitted games then timed out, and the platform quarantined new queue joins until 08:40:41. Restart attempts encountered the quarantine 403 as an unhandled fatal error, which extended a local move failure into a supervisor-wide interruption.

The immutable game history assigns approximately `-99.1` displayed-rating points to 14 timed-out games in the incident window: 3 Bargaining games contributed `-21.8`, 3 Negotiation games contributed `-20.9`, and 8 Persuasion games contributed `-56.4`. These losses are transport damage, not evidence about family policy quality.

The pinned GLEE SDK contained a deeper duplicate-submission risk. It retried every `requests.ConnectionError` for every HTTP method on the premise that such an error proves non-delivery. The incident log contains `RemoteDisconnected` after a POST, which can instead mean that the peer received the request and closed before returning a response. Retrying a move or queue POST in that state is unsafe.

[Transport fault containment v2](../../artifact/nommd-arena/protocols/glee-transport-fault-containment-v2.md) gives each non-idempotent POST exactly 1 network attempt. An ambiguous move outcome becomes a durable per-turn suspension and is never replayed, including after supervisor restart. An ambiguous queue outcome remains outstanding until a match is observed or idempotent queue cancellation establishes absence. The exact timeout-quarantine 403 now pauses only admission until the server retry time while active games and the supervisor continue.

## Rating decomposition

The 797-game analysis frontier contains 269 Bargaining, 252 Negotiation, and 276 Persuasion games. Authenticated history supplied ratings for 766 games. Before the outage, all 3 families were net positive; after recovery and excluding timeout losses, Bargaining and Negotiation remained positive while Persuasion was approximately flat.

| Family | Rated games | Gross delta | Delta excluding timeouts | Pre-outage mean | Post-recovery mean excluding timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bargaining | 263 | `+146.5` | `+168.3` | `+0.738` | `+0.406` |
| Negotiation | 243 | `+131.9` | `+152.8` | `+0.761` | `+0.256` |
| Persuasion | 260 | `+36.5` | `+92.9` | `+0.476` | `-0.036` |

Known-versus-hidden identity cuts continue to support Strategic Identity Concealment rather than showing a general SIC cost. Excluding timeouts, Bargaining averaged `+0.735` against known opponents and `+0.545` against hidden opponents; Negotiation averaged `+0.364` known and `+0.856` hidden; Persuasion averaged `+0.181` known and `+0.539` hidden.

## Bargaining systematics

Bargaining remains strongly role-asymmetric. Excluding timeouts, player 1 produced 111 rated games, `+121.4` total, and `+1.094` mean; player 2 produced 149 games, `+46.9` total, and `+0.315` mean. Agreements below half of the available share remain expensive: 43 games below 40% own share averaged `-2.867`, 61 games from 40% through below 50% averaged `-1.679`, and 156 games at or above 50% averaged `+2.526`.

The rating-v3 advisory failed on 344 of 959 Bargaining turns with `ValueError: Bargaining candidate cannot invent a hidden discount factor`. This removes rating guidance from a large, structurally identifiable subset and should be repaired before interpreting those turns as a failure of the rating model itself.

41 games activated v2.17 intervention changes and together lost `-133.3`, but this is selected-state evidence: the guards fire precisely in already adverse positions. Low-share, post-rejection capitulation, nominal-share fallback, repeated-extreme-offer, patient-acceptance, and dominated-acceptance cases deserve chronological replay; their observed terminal deltas do not by themselves prove that the guards caused the losses.

## Negotiation systematics

Negotiation is also role-asymmetric. Excluding timeouts, 123 seller games contributed `+163.9` with `+1.333` mean, while 117 buyer games contributed `-11.1` with `-0.095` mean. Agreements averaged `+1.688`; no-deal and walk-away outcomes averaged `-0.532` and `-0.481` respectively.

The first review overcounted premature abandonment by treating game `29808bc4` as counterfactual evidence. A later event-level audit established that this was a single-round game whose prompt explicitly prohibited counteroffers: buyer value was `10,000`, seller value was `8,000`, the seller asked `12,000`, and both rejection and walkaway correctly ended at zero rather than accepting negative surplus. The game remains a negative rating outcome but is not a policy defect. Other buyer walkaways require the same legal-horizon check before they can support an abandonment claim.

A second defect is scale-insensitive acceptance near zero surplus. The game with ID prefix `85e876e6` accepted buyer payoff `$1` against seller payoff `$499,999` and lost about `-7.2`; the game with ID prefix `3f26055d` accepted `$0.01` against `$2,999.99` and lost about `-5.9`. The existing exact-zero guard needs a scale-aware minimum-surplus boundary.

The conditional selector also failed open on 128 Negotiation turns with `conditional request must contain 2 to 5 candidates`; every affected turn used the deterministic branch. This did not crash the runner, but it spent planner capacity before discarding conditional inference whenever guard projection left 1 candidate. The next Negotiation cycle should treat that state as an intentional guarded single action, analogous to Bargaining's duplicate-collapse repair.

## Persuasion systematics

Persuasion's seller side remains productive while the buyer side is flat. Excluding timeouts, 121 seller games contributed `+88.7` with `+0.733` mean; 131 buyer games contributed `+4.2` with `+0.032` mean. Sellers with 0–4 purchases averaged `-2.131`, while sellers with 15–20 purchases averaged `+2.531`.

Buyer behavior varies sharply by prior. Games with prior `1/3` averaged `+0.316`, prior `0.5` averaged `+0.347`, and prior `0.8` averaged `-0.794`. This is not a legality failure: high-prior buying can produce large payoffs, but the rating result indicates that competitors often extract more from the same favorable regime or exploit our predictable response.

Seller text does not support a simple conclusion that positive wording is itself defective. High-quality sellers were overwhelmingly positive; low-quality sellers used a mixture of positive, negative, and neutral messages. Games with no negative message averaged higher rating and purchases than games containing a negative message, but opponent and policy selection confound that comparison. Hard passers also produced zero purchases under varied wording.

## Runtime and infrastructure observations

Model execution was not the incident's general cause. Excluding the isolated long tail, Bargaining worker time averaged `9.637` seconds with `26.183` at p90, Negotiation averaged `21.570` with `30.454` at p90, and Persuasion averaged `7.678` with `19.908` at p90. The 120-second move window provided ample normal headroom.

The unified API limiter deferred substantial background work at the `1800/1800/1800` rate: the event journal includes 7,064 pending-game poll deferrals, 974 stats poll deferrals, and 3,518 activity-dispatch deferrals. Those controls preserved move throughput and did not create the outage, but they show that this rate has little background-observation headroom.

## Next policy work

Transport v2 is the immediate repair. The next gameplay work, after a separately authorized restart boundary, is to repair Bargaining rating-v3 hidden-discount projection, replay selected Bargaining intervention states, prevent Negotiation buyer walkaway only when a legal counter inside visible feasible overlap remains, introduce a scale-aware minimum buyer surplus, collapse the 1-candidate Negotiation conditional path without error, and investigate Persuasion high-prior buyer predictability. These policy changes should not be mixed into the transport-only v105 seal without a new replay frontier.
