# Persuasion buyer-continuation live protocol v1

## Decision

The frozen `persuasion-buyer-continuation-v1-frontier-85595` release is promoted as bounded advisory evidence for unresolved, nonterminal Persuasion buyer decisions. It is not promoted as an action policy and cannot replace current-payoff arithmetic, seller-reliability evidence, deterministic safeguards, or Terra's final selection.

## Frontier

The path applies only when DeepRMM-01 is the buyer, the legal action is `buyer_decision`, at least 1 later product remains, and the v2.7 advisor reports `advisory-only` authority with no selected action. Categorical and bounded-authoritative decisions bypass cloud inference, while terminal buyer decisions retain the existing path because no next seller signal exists to predict.

## Runtime sequence

Buy and pass form the exhaustive action set, so no candidate-planner call is useful. The worker normalizes and guards both actions, freezes their exact identities, requests one local candidate-conditioned batch from the existing conditional sidecar, and sends the aligned surface to one Terra selector. The selector can return only an index into the frozen 2-candidate set.

For each candidate, the reversed head receives the authenticated causal prefix plus the proposed buyer action while masking eventual response time and excluding any quality revelation caused by buying. It predicts the opponent seller's next signal as positive, negative, or unknown. The service writes one immutable combined candidate-set forecast to a SQLite WAL registry before the future signal is visible.

## Authority

Immediate expected buyer payoff remains primary. Terra may prefer a candidate with lower current expected payoff only when visible current-game evidence supports a bounded future information or payoff advantage large enough to cover the current gap after uncertainty. Generic trust, retaliation, exploration, consistency, or raw model sensitivity is insufficient.

The head is an observed-policy predictor rather than an identified causal model. It does not predict the current hidden quality, the truth of the next seller signal, the next product's quality, later seller behavior, or the payoff-optimal action. The preceding-negative-signal plus buy cell has only 135 frozen examples and is marked sparse at runtime.

## Evidence and evaluation

The validation-selected DeepRMM-only 2-seed ensemble achieved test game-macro NLL 0.6318 versus 0.7266 for the preceding-signal-and-action Markov baseline; the paired game-bootstrap interval for ensemble-minus-baseline NLL was [-0.1467, -0.0393]. Fieldglass augmentation was rejected by the validation gate. These results justify prospective deployment, not a claim that the selector improves payoff.

Every live forecast is registered before its target seller signal, and every selected move records the candidate set, conditional surface, selector receipt, safeguards, and fallback disposition. Inference failure, malformed evidence, timeout, or selector failure returns the existing guarded deterministic action.

## Negotiation decision

Negotiation does not receive an analogous reversed head. `AcceptOffer` and `WalkAway` are terminal, while `RejectOffer` normally carries a counteroffer that becomes DeepRMM-01's own next proposal; the already promoted forward offer-to-opponent-response twin models the useful direction after that proposal. The 2-event reject-plus-counteroffer bridge therefore reuses that twin rather than training an action-to-next-opponent-proposal head.

The immutable all-history corpus confirms this event structure. Among 3,937 self `reject` response events in 1,185 Negotiation games, 3,785 are immediately followed by a self proposal carrying the counteroffer, and 3,782 of those proposals are immediately followed by an opponent response: 3,550 rejects, 209 accepts, and 23 walk-aways. The other 152 self rejects occur at the end of the recorded horizon, while only 3 bridged proposals lack a recorded opponent response. Training an action-to-next-opponent-event head on this corpus would therefore duplicate the forward response target for almost every nonterminal decision while omitting the counteroffer value that causally distinguishes the candidates.

The live bridge is category-gated before planning. It activates only when the legal schema permits a counteroffer and the existing guarded deterministic or advisor path has already selected `RejectOffer` with a finite `product_price`; positive-offer acceptance, walk-away, final rejection, and bare rejection retain the established one-call or hard-authority path. Candidate planning may vary only the attached counteroffer and its legal message. Any candidate whose guarded form changes category is collapsed to the guarded baseline, and fewer than 2 distinct compound candidates fails closed.

For local inference, the credential-free client copies the visible state, records the current opponent offer with DeepRMM-01's fixed rejection, advances to the equivalent next offer frontier, strips only the fixed `decision` field from each immutable candidate, and sends the projected counteroffers through the unchanged frozen Negotiation offer-response twin. Projected hashes are validated on return and then mapped back to the original compound candidate hashes before selector construction. The source game and candidate set remain immutable, the sidecar receives no new authority, and the receipt binds both identities under `glee-negotiation-reject-counteroffer-bridge-v1`.
