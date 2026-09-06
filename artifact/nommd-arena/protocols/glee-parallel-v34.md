# GLEE parallel supervisor v34

**Status:** Activated across all 3 game families on 2026-08-15; the shared sensor, authenticated-history services, and public reporter remained continuously running through the cutover.

## Purpose

V34 activates the frozen rating-v3 dual-channel advisory in Bargaining, Negotiation, and Persuasion. It changes the information shown to Terra and creates immutable prospective forecasts, but it does not change family-specific executable-advisor logic, declarative policy parameters, legality guards, matchmaking targets, worker counts, model selection, timing concealment, message-style policy, statistical-package behavior, or infrastructure services.

## Preserved opponent-model frontiers

The cutover folds every completed predecessor journal into a fresh content-addressed seed so the new execution bodies do not forget online evidence. Bargaining advances from 1,413 seed games plus 34 journal games to a 1,447-game seed with SHA-256 `155ccac8d1cb79f13d0f87a07ba621d5532310c62c691dd30eb1b978c5917524`; Negotiation advances from 2,045 plus 87 to 2,132 games with SHA-256 `20a9c8e74ade7db792e60b3a6970db7003b70765c4009e48a5dfc97d8e8e2881`; Persuasion advances from 1,168 plus 8 to 1,176 games, first promoted as v73 with SHA-256 `ad96d2788e12976ba1817e4e1e4d7956dd2d6028179da8a1acf21d73ea3c2246` and then receipt-only resealed as v74 with the identical corpus and seed SHA-256 `2891a95fce00815d492e96a26f546d3855e850db3fe695830e0def5e39ea40ea`.

The promoted seeds refresh implementation receipts against the v34 supervisor and worker while preserving the exact preceding corpus plus its contiguous run-local journal. The shared statistical-package overlay remains at the same WAL-backed live frontier and is not copied into the seeds.

## Rating-v3 channels

All families load `rating-models/current.json`, currently selecting `v3.0-frontier-38209`, and share `runs/glee-rating-v3-advisory-v1/registry.sqlite3`. Before Terra receives a turn advisory, the supervisor records the exact causal game context and advisory in that append-only registry. At terminal observation it records a pre-history-sync rating forecast; the existing authenticated-history database later supplies the immutable target.

The bounded prompt projection may affect Terra's choice and is therefore an intervention, not an action-independent shadow. The simultaneously sealed forecast remains prospective evaluation evidence because it precedes the outcome. Rating v3 has no direct normalization, guard, submission, matchmaking, identity-routing, lexical, timing, or fallback authority, and Terra is warned that omitted continuation is not zero-valued continuation.

## Lifecycle boundary

The canonical control modules use fresh labels `glee-v71-bargaining-v2-17-policy-v2-20-account-context-v1-rating-v3-g48-w8`, `glee-v72-negotiation-v2-9-policy-v2-11-rating-v3-g48-w2`, and `glee-v74-persuasion-v2-7-policy-v2-7-rating-v3-g48-w3`. Starting these modules creates new immutable manifests and journals; existing infrastructure processes continue under their unchanged module specifications.

Successful activation requires each new manifest to pin the rating release, model hash, registry path, history path, `prompt_authority=advisory-only`, and `guard_authority=none`. Early monitoring must verify pre-model turn registration, prompt transport, terminal registration, target maturation, SQLite concurrency across 3 supervisors, model latency, malformed-advisory failures, and zero rating-v3 deterministic safeguards.

## Activation record

Bargaining v71 entered the supervisor at `2026-08-15T05:46:11Z`, Negotiation v72 at `2026-08-15T05:49:04Z`, and Persuasion v74 at `2026-08-15T06:00:31Z`. Authoritative host-side control status then reported all 4 infrastructure modules and all 3 family modules running with valid PID identities, matching module specifications, and the unchanged shared-contract hash `e039920bcd4413851a2ede227bb0c8403a58cf2d809e97dfae15e0a0887353c2`.

The first Bargaining terminal receipt registered a pre-outcome rating-v3 forecast, matured it from authenticated history, and updated both the statistical overlay and the v2.17 advisor journal. Negotiation and Persuasion both wrote valid startup manifests and successful shared-registry reconciliation receipts. The initial Persuasion v73 wrapper failed locally before manifest creation or matchmaking because its sealed `live_protocol` receipt preceded a final documentation status edit; the controller stopped its retry loop, v74 refreshed implementation receipts without changing the 1,176-game corpus, the full suite passed, and only then did the repaired module start.
