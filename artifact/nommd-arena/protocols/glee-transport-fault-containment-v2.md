# GLEE transport fault containment v2

**Status:** Implemented for the shared Bargaining, Negotiation, and Persuasion supervisor after the 2026-08-26 transport incident.

GLEE's move and queue endpoints are non-idempotent POST operations. A response timeout, connection closure, or server error does not prove that the server failed to receive or apply the POST. The production client therefore makes exactly 1 network attempt for every POST. It retains the pinned SDK's retry behavior for GET and DELETE requests, whose repetition is safe for the operations used here.

This boundary intentionally overrides the pinned SDK's broader `ConnectionError` retry. The incident log contained a `RemoteDisconnected` after a POST, which can occur after the peer receives the request; classifying every such failure as pre-delivery and replaying it is unsound.

An ambiguous move outcome suspends only the affected turn. The broker durably records `transport-suspended`, the supervisor does not replay that turn during the current process or after restart, and a later visible turn or authenticated terminal state reconciles the suspended receipt. Other games and family services remain live.

A `400 Game is not active` response is a terminal-state race, not an ambiguous transport failure and not a supervisor failure. The broker reconciles only that prepared turn, the action is never replayed, and the supervisor continues serving every other admitted game.

A delayed prepared action is revalidated against its durable broker receipt immediately before release. If a newer visible turn or terminal refresh has already reconciled it, the supervisor discards the stale ready action without a network call. Any other unexpected release exception is contained to that turn, recorded durably where possible, and changes the live process into a graceful drain instead of unwinding the unified supervisor.

An ambiguous queue outcome retains the dispatched arrival as outstanding, observes pending games for a match, and then uses the idempotent family-specific leave-queue operation to establish absence before returning the arrival to the scheduler. It never blindly submits a second queue POST.

The platform's exact timeout-safety 403 response is an admission-control state rather than a fatal process error. The supervisor restores the outstanding arrival, pauses new admissions until the server-provided retry time or a bounded fallback time, and continues serving admitted games. Rate-limit and local API-budget refusals remain explicit non-ambiguous deferrals.

The 2026-08-26 outage is the original motivating failure case: an uncaught ambiguous move transport error ended the unified supervisor, subsequent queue attempts encountered the platform timeout quarantine, and repeated process exits amplified the interruption. The 2026-08-27 incident added 2 broker/server races: a delayed action survived after its receipt had been reconciled, and the server closed a game before accepting a prepared move. Both ordinary per-turn outcomes escaped the supervisor, and every process retry then incurred the old Bargaining reconstruction delay. V2 now contains all of these paths without treating approval latency, model latency, or ordinary queue pressure as their cause.

This protocol changes transport semantics only. It does not alter family gameplay policy, authorize automatic relaunch, or convert a graceful drain into a hard stop.
