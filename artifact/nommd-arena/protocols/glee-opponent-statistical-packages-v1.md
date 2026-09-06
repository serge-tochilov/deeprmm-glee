# GLEE opponent statistical packages v1

**Status:** Implemented with a compiled frontier-18108 base and a transactional per-game live overlay; replaces prose opponent dossiers in the current live path, with advisory rather than action authority.

## Decision

The current GLEE stack retires model-written opponent dossiers and switches to deterministic statistical packages. Historical dossier corpora, revisions, cloud calls, and audits remain immutable research evidence, but no canonical service updates them, no completed game enqueues a dossier job, and no current worker prompt receives dossier prose.

The decision follows the frozen 292-call Bargaining comparison over 146 decisions from 9 recurring opponents. None of the 5 opponent-cluster bootstrap intervals comparing direct structured history with dossier-assisted Luna Max predictions excluded zero; point estimates were mixed, while the dossier arm cost approximately 27 additional seconds and 1,449 additional reasoning tokens per successful call. The result does not prove that prose summaries can never help, but it does not justify their latency, quota, collision, and prose-drift burden in the production path.

## Compiled matrix

Release `frontier-18108-v1` covers every public opponent ID known to the frozen temporal registry crossed with all 3 families: 200 opponent IDs, 3 families, and 600 package cells. Bargaining has direct evidence for 91 IDs from 1,040 games and 1,688 action observations; Negotiation has direct evidence for 79 IDs from 651 games and 4,578 observations; Persuasion has direct evidence for 58 IDs from 199 games and 3,875 observations. The remaining cells explicitly inherit the corresponding family population rather than pretending that missing evidence is an opponent profile.

The package stores each family-population count model once and stores only sparse per-ID count deltas. This avoids copying a full smoothed distribution into all 600 cells and keeps the complete tracked release below 1 MB. Each family uses the action-channel smoothing strength selected under the frozen chronological analysis: `5.0` for Bargaining, `20.0` for Negotiation, and `5.0` for Persuasion.

An observed opponent move becomes a scale-normalized outcome conditional on move kind, opponent role, information regime, horizon regime, round phase, and message availability. Proposal and signal values use width-`0.1` bins on the clipped interval `[-2.0, 2.0]`; response conditions additionally include the normalized offered-value bin, and response outcomes encode the observed decision. The runtime projection ranks stored contexts against the visible turn, emits at most 8 contexts and 5 explicit outcomes per context for an exact ID, emits at most 4 contexts for a population fallback, and preserves omitted probability explicitly.

## Per-game live overlay

Every terminal game played by DeepRMM-01 is applied immediately through one `glee-opponent-statistical-live-overlay-v1` SQLite transaction. The transaction records a content hash and normalized observations under the server game ID, increments the family-population counts, and also increments the direct opponent-family counts only when the disclosed label resolves to exactly one current public ID. Because every direct distribution is smoothed toward its shared family population, one population update immediately affects every package in that family without rewriting 200 duplicated files. A terminal game with no observable opponent action still advances the overlay revision and is recorded as zero-observation evidence.

The 3 family supervisors share one WAL database tied to the exact base-release manifest. `BEGIN IMMEDIATE`, a bounded busy timeout, a monotonic revision, and a hash chain serialize concurrent completions; replaying the same game with the same normalized observations and identity resolution is idempotent even if unrelated terminal metadata was enriched, while replaying the same family/game ID with changed applied evidence fails closed. Before queue admission after a restart, each supervisor reconciles its durable terminal game files against the overlay, closing the crash window between server completion and local statistical publication. A persistent write failure gracefully drains that family instead of continuing to collect games against a stale package.

Every turn reads the overlay inside one SQLite snapshot and merges it with the verified frozen base. The broker snapshot and event receipt record the overlay revision and state hash used for that decision. The live database is reproducible high-churn state under `opponent-statistical-packages/live/`, so Git ignores its SQLite, WAL, and shared-memory files; terminal game archives and the applied-game hash ledger remain the recovery sources.

## Identity and collision boundary

A server-disclosed label selects an ID-specific package only when the frozen public registry maps its normalized current label to exactly one present stable public ID in that family. Hidden identities, missing labels, and current collision sets use the family population. `RESERVE`, `agent`, `concord`, and `test` are collision sets in all 3 families at frontier 18108; their candidate IDs are exposed for audit, but the runtime never silently selects one. The overlay retains each unresolved game's normalized observations and candidate set, allowing a later validated clustering release to add direct counts without adding its already-counted observations to the family population again.

The separately frozen conditional Bargaining identity-route shadow remains an experiment. Its package gate may register generic and candidate predictions prospectively, but v1 statistical-package integration does not promote hidden-ID routing into live action selection and does not turn a candidate posterior into identity fact.

## Authority and update lifecycle

The package is empirical advisory evidence. Authenticated game state, legal-action schemas, deterministic guards, explicit analytic authority, and family-specific executable advisors retain their existing precedence. Direct support, population support, smoothing strength, visible-context match, and evidence tier must control how strongly a model uses the package; normalized historical bins must never be interpreted as nominal current-game amounts.

Every live run pins the frozen package manifest SHA-256 and live-overlay contract in its immutable run manifest, and every broker snapshot records the base manifest, overlay revision, overlay state hash, and identity-resolution status. Ordinary game completions update the shared overlay without restarting any service. Recompiling from a later public frontier creates a new immutable base and therefore a new manifest-addressed overlay database; that base migration requires new run labels and an explicit control-plane contract-change restart, with frontier selection preventing games already absorbed into the base from being counted again. A base release remains deterministic from hash-verified identity, behavior, channel-calibration, and activity sources.

The reproducible compilation command is:

```bash
UV_CACHE_DIR=/tmp/nommd-arena-uv-cache uv run --offline --no-sync nommd-arena glee-statistical-package
```

## Historical dossier boundary

`opponent-dossiers/` remains a preserved archive for method comparison, failure analysis, and paper receipts. The dossier compiler, updater, and production-replay commands remain available only to reproduce historical experiments; they are not part of the canonical control registry or current worker transport.
