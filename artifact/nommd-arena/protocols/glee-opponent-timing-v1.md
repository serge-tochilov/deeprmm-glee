# GLEE opponent move timing v1

**Status:** Implemented as a shared compact shadow ledger and deadline-safe outward timing concealment; timing candidates do not yet route named dossiers or alter economic actions.

## Objective

Move latency is both evidence and an outward signal. For an opponent, a concentrated sub-2-second response distribution suggests a deterministic or heavily cached branch, a broader model-scale distribution suggests inference, and a latency increase relative to the same opponent's comparable states may indicate that the current state was harder for its machinery. For DeepRMM-01, the same distinction can disclose when a deterministic guard, short model call, long model call, retry, or fallback generated the move.

Timing never proves substrate, identity, or internal difficulty. Network transit, platform scheduling, polling cadence, queue contention, intentional delay, model capacity, retries, and process load are alternative causes, so all labels are hypotheses and every receipt preserves source quality.

## Evidence channels

The authoritative channel is the server-supplied `response_time_ms` embedded in authenticated completed history entries. It is attributed only when the corresponding response actor is the opponent: the player deciding the other player's Bargaining offer, the `decided_by` player in Negotiation, or the buyer in Persuasion. Re-reading a longer history does not duplicate an observation because the row identity is derived from game, round, actor, move kind, and timing source.

The supplemental channel is a local causal wall interval from the timestamp of DeepRMM-01's accepted nonterminal submission to the first later observed opponent-caused turn or terminal state. This interval captures opponent proposals and Persuasion signals that lack an exact server response field, but includes network and polling delay. It therefore carries the configured polling resolution, receives less weight during attribution, and is censored from live inference above 180 seconds while remaining stored for audit.

No latency is invented for a game's first visible move because its causal start is unknown. An accepted terminal move creates no future anchor. A rejected action is not an anchor; an accepted emergency retry replaces it. Restart recovery never fabricates a missing interval.

## Compact store

All 3 family supervisors write `runs/glee-opponent-timing-v1/timing.sqlite3` through SQLite WAL with a 30-second busy timeout. The store contains one compact row per delay observation plus at most one pending submission anchor per active game; it references source run, event sequence, game, turn, and context fingerprint without copying game JSON, prompts, transcripts, or model outputs.

The historical bootstrap command is `UV_CACHE_DIR=/tmp/nommd-arena-uv-cache uv run --locked --no-sync nommd-arena glee-timing-bootstrap --source-root runs --output-root runs/glee-opponent-timing-v1 --poll-resolution 4`. It scans immutable event and terminal-game references, deduplicates repeated archives by logical move identity, and can be rerun incrementally without increasing duplicate rows.

## Fingerprint and state-hardness hints

Every named opponent-family profile separates exact server timing from local wall timing and then separates move kinds. It records count, median, p10, p90, log-scale interquartile range, mass at or below 2 seconds, mass at or above 8 seconds, and mass at or above 30 seconds. A stable SHA-256 identifies the complete derived profile revision.

The conservative engine hint requires at least 3 exact observations. At least 80% at or below 2 seconds with p90 no greater than 4 seconds is `deterministic-like`; at least 60% at or above 8 seconds or a median at or above 8 seconds is `model-call-like`; the remainder is `mixed-delayed-or-hybrid`. The names describe observational regimes, not implementation facts.

For a new exact move, the state-hardness hint compares log latency against the same named opponent, family, and move kind, preferring prior states within 2 units of a compact structural-complexity score. A residual of at least `log(2)` is `hard-state-like`, a residual no greater than `-log(2)` is `easy-state-like`, and the middle is `typical-latency`; deterministic-like profiles are retained but explicitly not interpreted as cognitive difficulty.

Timing deanonymization compares a hidden game's source- and move-kind-conditioned log delays against named profiles in the same family, using near-complexity reference rows when available and weighting exact server timing above local wall timing. The live receipt exposes up to 5 timing-similarity candidates with support counts. This is a distinctive behavioral channel but not a posterior: it remains separate from presence, numeric policy, language, action, and public-delta evidence, retains an `unknown` possibility at the future fusion layer, and cannot update a named dossier.

## Outward concealment

Concealment is implemented once in the credential-owning supervisor after inference, deterministic safeguards, and advisor submission projection but before the API move. It applies uniformly to deterministic decisions, short model calls, ordinary fallbacks, and other valid prepared decisions by sampling a target *total latency since first observation*, then waiting only for the difference between that target and the natural elapsed time. Long calls that already exceed the target receive no extra delay.

The production target is a fresh cryptographically seeded beta-2-2 draw bounded between 14 and 46 seconds. The realized target and delay are receipted after use, but no reusable random seed is published. This does not make timing unidentifiable; it deliberately overlaps fast and ordinary model branches and raises the cost of a cheap branch classifier while preserving natural long-tail variation.

Waiting is non-blocking: the decision enters a supervisor-owned ready queue, model threads can serve other games, API polling continues, and no sleep occupies a worker. Release occurs at the target or earlier when waiting would approach the emergency reserve. A 12-second reserve remains inviolable, with an additional scheduler guard for the polling interval; server-rejection retries and recovered prepared actions submit immediately because legality and deadline recovery outrank concealment.

Every move records `move_delay_scheduled` and `move_delay_released`, including natural elapsed time, randomized target, requested, scheduled, and applied delay, release reason, selection branch, remaining deadline, and reserve. Exact monotonic clock values are used internally but omitted from durable public-facing receipts.

## Validation and promotion boundary

Deterministic tests cover actor attribution, deduplication, exact versus wall provenance, fast versus model-scale profile separation, state-hardness residuals, hidden timing similarity, historical reference-only bootstrap, nonblocking release, long-call no-op behavior, and reserve-preserving immediate release. Full-suite validation is required before any family process starts from the changed snapshot.

Timing candidates remain shadow evidence until chronological masked-name replay measures top-1 and top-5 recall, mean reciprocal rank, log loss after calibrated fusion, unknown rejection, family and role slices, and ablation against presence and non-timing behavioral features. Promotion may route a candidate dossier only through the larger presence-and-behavior protocol and must never convert hidden evidence into named direct evidence.
