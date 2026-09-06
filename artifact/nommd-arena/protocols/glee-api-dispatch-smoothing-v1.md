# GLEE API dispatch smoothing v1

## Purpose

This protocol raises the canonical DeepRMM burst target to `g48=1200/1200/1200` without turning independent model work into a serialized pipeline or allowing coincident queue and move calls to consume the agent-wide API budget in bursts.

## Empirical capacity boundary

The nearest complete Terra epoch contained 47 Bargaining, 47 Negotiation, and 46 Persuasion games. Mean game durations were `115.803`, `174.084`, and `721.611` seconds respectively. Scaling those observed service times to `1200` admissions per family per 48 hours projects `58.5%` mean occupancy of the shared 12-slot pool.

A 500-replication bootstrap over the observed family service-time samples estimated a `6.44%` full-pool fraction, `4.71` seconds mean dispatch lag, `17.1` seconds 95th-percentile lag, and `140.8` seconds 99th-percentile lag at `1200/1200/1200`. With one unavailable slot, the corresponding estimates rose to `12.36%`, `11.25`, `83.74`, and `232.25` seconds. The same simulation showed materially less headroom at `1400/1400/1400`, so `1200/1200/1200` is the selected burst ceiling rather than an assertion that all higher rates are unstable.

The authorized v104 expansion uses `1800/1800/1800` with one shared 20-worker pool. A 500-replication bootstrap over the newer v102 service corpus estimated `56.80%` mean utilization, `1.47%` full-pool time, `0.69` seconds mean dispatch lag, `0` seconds 95th-percentile lag, and `17.01` seconds 99th-percentile lag. Simulating one unavailable worker produced `59.79%` utilization, `2.76%` full-pool time, `1.46` seconds mean lag, and `52.10` seconds 99th-percentile lag. The earlier 47/47/46 Terra corpus gave slightly more favorable results, so the v102 estimate is the operational planning boundary.

## Scheduling contract

The 3 family demand clocks remain independent exponential arrival processes. Every observed arrival is durably retained, and shared-pool admission remains globally ordered by the oldest eligible arrival. Actual `queue` API calls are separated by at least `2.5` seconds when capacity and deadlines permit; smoothing delays dispatch but never drops or resamples demand.

Model calls remain independent across games and execute in the configured shared worker pool. There is no model-call queue, global model semaphore, or cross-game planner serialization. Planner and selector calls remain sequential only within the same 1.5-round turn because the selector consumes the planner candidates and local counterfactual forecasts.

Prepared moves use the existing in-memory ready set as an earliest-original-deadline-first API submission queue. Ordinary move releases are separated by at least `2.5` seconds. The spacing rule yields before either the original turn deadline reserve or the latest point at which the agent-wide limiter can still wait while preserving the bounded transport-retry budget; those authorities are stronger than smoothing.

The randomized SIC target remains a preferred release time, not a new deadline. Smoothing may add a small queue delay after that target, while deadline pressure may release earlier. Every release records the requested delay, applied delay, delay beyond the preferred release, spacing value, remaining original deadline, and release reason.

## Ambiguous queue transport

The GLEE SDK correctly does not retry a timed-out `POST /queue`: a read timeout does not establish whether the server accepted the request, and an immediate retry could create a second admission. The v101 launch preflight exposed this ambiguity before any game was admitted.

V102 keeps the dispatched Poisson arrival outstanding, marks its family locally queued, and continues observing pending games. If a game appears, that game consumes the original arrival. Otherwise, after a 12-second-or-3-poll grace period, the supervisor issues the idempotent family-specific `leave_queue` operation. A positive removal safely returns the original arrival to the durable scheduler. An empty result receives one further grace-and-leave check before the arrival is returned, covering a slow server-side queue completion without allowing an unbounded blocked family.

Transport errors or local API-budget deferral during reconciliation extend the observation period instead of crashing the supervisor. A process restart restores any still-outstanding arrival through the scheduler's existing durable recovery before new matchmaking begins. Events distinguish uncertain dispatch, extended observation, deferred reconciliation, queue cancellation, confirmed absence, and resolution by an observed match.

## API and process budget

At `1200/1200/1200`, the historical turn counts project approximately `11.67` move submissions per minute and `1.25` queue admissions per minute. The canonical runner poll changes from `3` to `4` seconds, reducing its pending and statistics polling load while the existing SQLite rolling limiter continues to reserve critical and control capacity under the 60-request-per-minute server ceiling.

The v102 live epoch observed `8.28` move submissions per minute and a rolling 60-second maximum of `14` while every move retained a `target-reached` release reason and at least `20` seconds of deadline reserve. Linear scaling to `1800/1800/1800` gives approximately `12.4` moves per minute and a rough peak near `21`, below the `24`-per-minute ordinary move-spacing capacity. The same epoch averaged `1.59` concurrent Terra calls with a peak of `6`; proportional planning values for v104 are approximately `2.39` mean and `9` peak, below the 20-worker pool. The 60-request agent-wide limiter remains the closest shared boundary and retains authority over lower-priority polling.

The prior 1,024 MiB RSS limit sat only about 43 MiB above the nearest observed 981.195 MiB peak. The canonical ceiling is therefore `2,048` MiB on the 15 GiB host, while the same periodic RSS sampling and graceful-drain behavior remain active.

## Operational boundary

This protocol changes configuration and dispatch behavior only. It does not authorize matchmaking, alter family decision policies, change model prompts, or weaken the existing graceful stop conditions. Live activation still requires an explicit user request through `tools/glee_control.sh`.
