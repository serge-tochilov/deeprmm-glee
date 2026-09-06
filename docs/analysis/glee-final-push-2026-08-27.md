# GLEE final-push plan for 2026-08-27

**Status:** Plan recorded on 2026-08-21; the Terra repair bundle was implemented as v100, exercised through v102, and tightened in the stopped private-archive v103 family-repair record `glee-v102-drain-family-repair-2026-08-25` on 2026-08-25. The unified game module remains stopped, and this page does not authorize an early restart.

## Objective and present boundary

Cloud quota is reserved until the reset expected on 2026-08-27. DeepRMM game admission remains stopped and its rating may decay toward the `1,800` defense threshold. Read-only and statistical infrastructure may remain active, but it must not admit games or invoke decision models. The final push begins only after the reset is confirmed from current provider state, not merely from the calendar.

The implementation bundle consists of the evidence-backed private-archive records `glee-bargaining-post-retrain-repair-plan`, `glee-negotiation-post-retrain-review`, and `glee-persuasion-post-retrain-review`. Work should be done as one coherent offline release so shared candidate, guard, rating, receipt, and controller contracts are tested together before any live game sees them.

## Offline sequence after reset

1. Reconfirm that the canonical unified `deeprmm` game module is stopped and identify one immutable source frontier for code, models, statistical packages, rating evidence, and all 3 post-retrain corpora.
2. Implement the narrow deterministic and composition repairs in the 3 family plans without changing scheduling, SIC, timing concealment, authentication, or storage contracts unless a repair strictly requires it.
3. Run focused unit and integration regressions for every named failure, including Bargaining intervention construction and one-rejection surrender, Negotiation categorical candidate comparison and hardening progress, and Persuasion buyer admissibility, terminal choice, and selector gating.
4. Run no-cloud chronological first-divergence replay over each complete post-retrain family corpus, preserving observed actions outside targeted classes and censoring outcomes after the first changed action.
5. Decide explicitly whether a longer replay is necessary before restart; do not treat it as an automatic ritual.
6. Seal the release, test the canonical launcher and hot-policy loading, verify all infrastructure and prospective registries, then start the unified module once at the planned increased density, initially `g48=600/600/600` with the shared 12-worker pool unless current quota or platform conditions require a lower declared value.

## Long-replay decision gate

The required baseline is a complete no-cloud replay of the post-retrain B/N/P windows plus focused synthetic regressions. Before restart, review whether “long replay” means scanning older all-history trajectories, re-running broad model prompts, or both, and record a yes/no decision with its evidence frontier, expected runtime, provider-call cost, and the promotion decision it can change.

A long replay is warranted when a repair changes a shared contract across many otherwise untargeted states, targeted replay reveals unexplained action drift, older history contains materially different mechanics absent from the post-retrain window, or the result can realistically block or alter promotion. A long provider replay is not warranted merely to produce another large number: stochastic re-calls do not recover counterfactual outcomes, historical rating cannot be assigned after the first divergent action, and consuming the newly reset quota can reduce the value of the live final push.

The current prior is therefore asymmetric: a full local mechanical prefix scan is likely useful because the Bargaining and Negotiation composition changes are broad, while a long cloud replay should be skipped unless the focused results expose a concrete ambiguity that only fresh model judgments can resolve. Persuasion's buyer gates can be tested primarily from frozen posterior, continuation, selector, and revealed-purchase receipts without re-calling a provider.

## 2026-08-25 offline execution

Sol was retired from production. The canonical planner is Terra High, the only capacity fallback is Luna High, and exhaustion of both paths falls back deterministically. Historical Sol runs and dormant offline synthesis utilities remain immutable research evidence, but no active config, current pointer, required path, or production fallback depends on Sol.

The no-cloud replay covered 52 Bargaining games and 223 turns, 54 Negotiation games and 262 turns, and 52 Persuasion games and 1,031 turns from v97 through v99. It made 0 provider calls and found 17 first-divergence games: 3 Bargaining, 9 Negotiation, and 5 Persuasion. Every first divergence belonged to an intended repair class; later changed actions in the same trajectory were censored and received no historical rating attribution.

Bargaining changed first through 2 rejected-extreme-offer guards and 1 restored exact-authority guard. Negotiation's 9 first changes were all one-probe hardening-budget interventions. Persuasion changed 3 terminal expected-value decisions, blocked 1 materially negative purchase, and restored 1 positive local expected-value action outside the continuation-sensitivity gate. No untargeted action drift, invalid action, malformed prefix, or transport failure appeared.

A longer all-history or provider replay is not required before a future restart. The complete post-retrain corpus contains the current mechanics, all observed changes are narrow and explained, focused counterfactual branches are covered synthetically, older epochs mix obsolete contracts, and fresh cloud calls cannot recover terminal outcomes after the first divergent action. This decision preserves quota and does not claim a historical rating benefit.

The v100 release preserves the v97 historical corpora exactly while refreshing implementation receipts: 4,520 Bargaining games, 2,940 Negotiation games, and 2,096 Persuasion games. Its canonical label is `glee-v100-deeprmm-terra-repair-meta15-balanced-self-mirror-shared-w12-g48-120`; sealing the release did not start it.

The disabled source-local collector was resealed against the same repaired shared worker and supervisor rather than left on stale Terra receipts or restored to its later Opus/Sol receipts. Its worker-frontier, pilot-runtime, and 30-game review-runtime audits all pass with 0 cloud calls, 0 network requests, and no live authority. Final verification passed all 723 repository tests.

## v101 burst scheduler

The planned final burst is now represented by canonical release `glee-v101-deeprmm-terra-repair-meta15-balanced-self-mirror-shared-w12-g48-1200`. It preserves the v100 family policies and historical frontiers, reseals their implementation receipts against the changed supervisor, and raises all 3 independent demand clocks to `1200` admissions per 48 hours. Configuration alone does not authorize or start matchmaking.

The first authorized v101 startup reached its live preflight but a timed-out `POST /queue` exposed an ambiguous transport case before any game was admitted. The retrying wrapper was stopped cleanly, queue cleanup confirmed that no admission remained, and the failed immutable run was preserved. V102 retains the same strategic policies and rates while adding bounded, idempotent reconciliation: an uncertain queue request is never blindly repeated, an observed game consumes the retained arrival, and 2 empty `leave_queue` checks return it to the durable schedule without crashing the supervisor.

The capacity estimate used the nearest complete Terra epoch rather than arithmetic means alone. At `1200/1200/1200`, a 500-replication service-time bootstrap projects `58.5%` mean use of the 12 shared slots, a `6.44%` full-pool fraction, `4.71` seconds mean dispatch lag, and `140.8` seconds 99th-percentile lag. One unavailable worker raises the estimated full-pool fraction to `12.36%` and the 99th-percentile lag to `232.25` seconds; `1200` is therefore a high burst setting with retained-arrival backpressure, not an indefinitely safe steady-state claim.

Independent model calls remain unqueued across games. The existing ready-move set now supplies a narrow API-only queue: unified admissions and ordinary ready moves are each spaced by at least `2.5` seconds, admissions retain global FIFO order, and moves retain earliest-original-deadline-first order. The spacing rule yields before the original deadline or transport waiting budget can be consumed. Polling changes from `3` to `4` seconds, and the supervisor RSS ceiling changes from `1,024` to `2,048` MiB because the nearest measured peak was `981.195` MiB.

At the selected rate, the historical workload projects approximately `11.67` move submissions, `1.25` admissions, and `17.18` independent Terra calls per minute. The expected cloud-call concurrency is only about `2.1`; adding a model-call queue would therefore reduce useful parallelism without addressing the actual shared bottleneck, which is authenticated GLEE API traffic. The detailed contract is [GLEE API dispatch smoothing v1](../../artifact/nommd-arena/protocols/glee-api-dispatch-smoothing-v1.md).

## Reply-time and rating assessment

The [official GLEE scoring rule](https://glee-competition.com/llms.txt), rechecked on 2026-08-25, does not include legal reply latency. A terminal payoff is ranked within the same configuration and role, adjusted for opponent strength, mapped to a game rating, and propagated through the experience-dependent update and display shrinkage. Bargaining discount is applied by round after rejection, not by wall-clock seconds spent inside a turn. Wall-clock latency can affect rating only indirectly through timeout, reduced throughput, a shifted hourly scoring frontier, or an opponent reacting to timing as behavioral evidence.

The existing timing policy supplies a useful quasi-experiment. For each rated game with the canonical `ki-beta-2-2-v1` profile, the analysis selected the first fresh target in the randomized `14–46` second envelope and estimated its association with the final authenticated rating delta after removing run, family, player role, first phase, and identity-mode fixed effects. The source joined 34,476 unique timed turns from 52 immutable event journals to the 9,744-game authenticated history, found 0 duplicate conflicts, and retained 1,743 games in supported fixed-effect cells: 624 Bargaining, 540 Negotiation, and 579 Persuasion.

Across all families, one additional target second corresponded to `+0.01894` displayed-rating points, with a heteroskedasticity-robust 95% interval of `[-0.00883, +0.04671]` and a within-cell permutation `p=0.1696`. Moving across the complete 32-second target range therefore had an estimated effect of `+0.6062` points and a lower 95% bound of approximately `-0.283`. Bargaining was almost exactly null at `+0.00124` points per second; Negotiation and Persuasion point estimates were positive but individually imprecise. Realized first-reply latency showed the same non-negative pattern, but that observational result is secondary because hard states can themselves take longer.

This result finds no evidence that longer replies inside the active concealment envelope reduce rating, including through an opponent's downstream reaction. It does not identify the counterfactual effect of removing concealment entirely because near-instant replies lie outside the randomized support, and it cannot exclude an opponent reacting nonlinearly to a conspicuously slow versus fast identity. KI timing remains a stable public signature, while HI timing personas address that linkability channel; neither should be removed on the theory of a direct scoring penalty.

Operationally, the 34,476 releases applied a mean deliberate pause of `18.54` seconds and submitted with a mean `88.71` seconds remaining. No release used a deadline-pressure reason. The only rated timeout among the 3,771 timed games was the already diagnosed transport incident in which a prepared action had been released with about `93` seconds left before DNS failure and unsafe restart recovery; it was not caused by exhausting the turn through deliberate waiting.

Historical mean game durations project only `0.714` occupied worker slots at `g48=120/120/120`, of which deliberate pauses account for `0.360`, in the shared 12-slot pool. Scaling the same means to the planned `600/600/600` burst projects about `3.57` occupied slots, including `1.80` from pauses. This is substantial added latency but not mean-capacity saturation; pending depth, dispatch lag, long-game tails, and actual completion rates remain the authoritative live checks.

At a pure Poisson mean of 120 arrivals per family per 48 hours, the chance of fewer than 100 arrivals is approximately `2.79%` for one family and `8.13%` that at least one of 3 independent families falls short. That defense-margin risk comes from admission stochasticity rather than reply delay. A low-rate keepalive should therefore use a larger declared margin if avoiding hourly defense decay matters, while the planned `600` burst is far above the activity floor.

## v107 supervisor incident and recovery

The 2026-08-27 v107 burst exposed a compound reliability failure rather than a gameplay-policy collapse. A server-side terminal race returned `400 Game is not active` for a prepared move, and 2 delayed prepared actions remained in the release set after their broker receipts had already been reconciled. Each per-turn condition escaped the unified Python supervisor; the shell correctly retried, but every retry reconstructed the Bargaining advisor from 4,743 seed games, 11,240 decision rows, 2,430 response particles, and 3,645 proposal particles before it could serve live turns again. The approximately 8-minute-40-second recovery gap synchronized 78 timeout outcomes and about `-402.6` displayed-rating points.

V108 contains the fault at both boundaries. `Game is not active` now reconciles only the affected turn, delayed actions are revalidated against the durable broker immediately before release, and any other unexpected release fault is isolated before the supervisor enters a graceful drain. Bargaining startup now restores content-verified sufficient state from a seed-adjacent cache whose location is invariant across canonical and immutable-snapshot execution; every new terminal journal record carries a verified compiled update, so an ordinary retry does not rerun the historical particle populations. The immutable seed corpus and append-only journal remain the authority, and a missing or invalid cache still triggers one explicit rebuild rather than silent state adoption.

## Restart gate

Restart requires zero invalid actions in tests and replay, explicit receipts for every new guard or bypass, no unexplained behavior outside targeted failure classes, current model and policy hashes, healthy sensor/history/reporter/analytics services, writable prospective registries, and a verified canonical controller status. Any stale contract, missing release, broken fallback, or unexplained broad replay drift blocks the restart.

The restart must use only `nommd-arena/tools/glee_control.sh`; no direct worker launch, parallel legacy runner, or second launch point is allowed. One controlled start activates the unified shared pool. The first live review should be triggered by completed-game counts rather than wall time and should separate family, role, information regime, identity mode, model path, deterministic guard, selector path, fallback, and displayed rating.

## Stop conditions

The final-push epoch should be stopped gracefully if invalid submissions appear, the intended release is not assigned to new games, provider failures create a sustained fallback burst, authenticated history falls materially behind, a family exposes a catastrophic deterministic pattern, or remaining quota crosses the predeclared reserve. Existing games should drain unless continued play itself creates the material harm that justifies a hard family-specific stop.
