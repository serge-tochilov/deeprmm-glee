# GLEE account prompt context v1

**Status:** Implemented, frozen, and activated in the Bargaining v69 epoch on 2026-08-14; retained unchanged across the v34 rating-advisory cutover.

## Purpose

This protocol turns the useful part of the account-level hidden-identity analysis into bounded evidence for Terra. It does not identify an opponent as fact and does not create account-specific action authority. It answers a narrower question: after exactly one visible opponent move in a hidden-identity Bargaining game, does pooled behavior resemble one account strongly enough to give Terra a candidate hypothesis that survived chronological masked-known-identity validation?

## Frozen inputs and observation point

The source behavior corpus is `glee-behavior-corpus-v1/frontier-18108-final`, the source channel calibration is `glee-behavior-channels-v1/frontier-18108-final`, and the candidate account partition is `reports/glee-account-linkage-v1-20260814/groups.json` at public frontier `36499`. The account partition is a research hypothesis inferred from public evidence rather than an owner identifier supplied by GLEE.

The classifier sees exactly the first opponent move represented by the collision-safe timing, action, lexical, and discourse feature grammars. It receives no current public-presence input. This deliberately avoids applying a stale historical activity frontier to new games and preserves the temporally stable part of the earlier detector.

Whole games retain the existing chronological 60% train, 20% calibration, and 20% untouched-test split. Profiles pool sibling public agents directly at the account level. The channel stacker is fitted on the calibration block, while runtime profiles use train plus calibration. The untouched final block is used only to measure performance and select a deliberately post-hoc account whitelist for prospective use.

## Validation gate

The direct account model considers 15 Bargaining, 14 Negotiation, and 13 Persuasion candidate accounts. Across 379 untouched games it emits 20 named account predictions and gets 19 correct, or 95% precision at 5.28% coverage. All named predictions occur in Bargaining.

Only accounts with at least 3 untouched named predictions and precision of at least 80% are admitted. The resulting whitelist contains 3 Bargaining account hypotheses with supports of 5/5, 6/6, and 7/7. One candidate is rejected at 0/1, another remains too sparse at 1/1, and Negotiation and Persuasion always abstain. Public labels are omitted because the account partition is an inferred research hypothesis rather than authenticated ownership data.

This is a low-coverage gate selected after inspecting one untouched block, not a fresh prospective validation. Its first online epoch must therefore preserve every admitted and abstained receipt for later scoring rather than treating the historical 18/18 whitelist result as guaranteed future precision.

## Terra transport

The supervisor evaluates the frozen model independently for every visible turn. Before the first opponent move it records `awaiting-first-opponent-move`. Known-identity games record `known-identity-not-routed`. Unsupported families, an open-set `unknown` winner, and non-whitelisted account winners produce no model-facing context.

An admitted hidden Bargaining turn receives `opponent_account_hypothesis` containing the candidate account, candidate member labels, account posterior, unknown posterior, runner-up, account-linkage confidence, untouched masked-KI support, and evidence counts by channel. The context says `advisory-only` and contains no raw hidden-game prose beyond what Terra already sees in the authenticated transcript.

Terra may use the candidate to interpret present behavior and choose among otherwise defensible hypotheses or messages. It must not state the inferred identity to the opponent, reconstruct a named package from it, update named evidence from the hidden game, or let the hypothesis override authenticated state, exact arithmetic, deterministic guards, analytic authority, or the executable Bargaining advisor. Current-game evidence outranks the candidate.

## Activation and receipts

The immutable release is rooted at `opponent-account-models/current.json`. A future Bargaining supervisor activates it with `--opponent-account-model-root opponent-account-models`; the release receipt is pinned in the run manifest, and each turn logs `opponent_account_prompt_assessed` or `opponent_account_prompt_failed` before model inference.

The currently running Bargaining process does not load edited Python modules, and adding this source to its existing pinned run label would make resume configuration inconsistent. Activation therefore waits for an explicit controlled epoch cutover rather than silently restarting a steadily improving live campaign.

## Private verification boundary

The private source archive can rerun the compiler against its frozen behavior corpus and compare `model.json.gz` and `test-predictions.jsonl.gz` byte for byte with the reviewed release. The public artifact preserves the implementation and aggregate design evidence but intentionally omits the account partition, predictions, corpus, and model.
