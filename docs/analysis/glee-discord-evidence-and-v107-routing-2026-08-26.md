# GLEE Discord evidence and v107 routing

**Status date:** 2026-08-26

## Source boundary

This note records a user-supplied excerpt from the public GLEE Discord discussion through 2026-08-24. Organizer statements are treated as authoritative clarifications of competition mechanics; participant reports are timestamped field observations; guesses about other agents, tactics, and identities remain hypotheses. None of the discussion is private game-state evidence, and no participant claim receives deterministic action authority.

## Organizer clarifications

Organizer Eilam stated that Bargaining rating is based on the player's own payoff rather than the ratio between player payoffs, and that the update also depends on the exact game configuration and opponent rating. This resolves the live objective: opponent payoff and opponent damage matter only insofar as they predict behavior, continuation, or final-rank dynamics; they are not direct game reward, and the controller must not spend expected own payoff merely to hurt an opponent.

The organizers declined to add per-game rating deltas to the API during the final week. Authenticated history synchronization and causally registered rating shadows therefore remain necessary, while simultaneous completions prevent exact API-only attribution. The paper must distinguish organizer-defined scoring from our reconstructed displayed-delta model.

The organizers also announced optional archival GLEE Competition proceedings, separate from official NeurIPS proceedings. Every accepted competition paper will appear on the workshop and competition sites, while authors may opt into the archival proceedings through OpenReview after publisher details are confirmed.

The participation census reported 181 agent operators with at least one game, 141 with at least 100 games, 115 with at least 1,000 games, 77 with at least 10,000 games, 38 with at least 100,000 games across their agents, and 14 with at least 250,000 games. At the individual-agent level, 310 agents had at least 1,000 games, 213 had at least 10,000, 92 had at least 50,000, and 40 had at least 100,000. These figures establish a high-volume ecology but do not imply that volume alone determines rating.

## Participant observations

Several leading participants described primarily heuristic systems rather than general LLM deliberation. One reported that unusually high Bargaining standards and stonewalling were effective against minimally customized LLM agents, while also saying that the effective ratio changed over time. Another reported a Negotiation opponent that repeatedly deadlocked, and multiple participants described Persuasion as long and tedious. These observations support nonstationarity, family specialization, deterministic baselines, and explicit deadlock handling; they do not establish universal optimal thresholds.

The participant discussion also identifies operationally meaningful opponent classes: accommodating agents, fellow stonewallers, cold or all-pass Persuasion buyers, Negotiation deadlockers, family specialists, and high-volume point-farm agents. These labels should be stored as low-authority behavioral tags derived from our own game traces, not copied as facts from Discord or bound permanently to a public name.

## V106 process audit

The reviewed v106 window completed 307 games in approximately 7 hours and issued 5,157 cloud calls, about 17 calls per game and about 37 calls per Persuasion game. Deliberate response delays consumed approximately 19.2 active-game slot-hours, about 14% of the 20-slot pool's available capacity during the window. The cost was concentrated in Persuasion and included known-identity games where timing could not conceal the already visible identity.

The audit does not show a direct reply-time rating penalty. Earlier within-envelope analysis found no significant negative association between randomized `14–46` second targets and displayed rating delta. The v107 timing change is therefore justified by throughput and identity purpose: retain timing concealment for hidden identity, remove it for known identity, and continue to reserve time for legal submission and transport recovery.

## V107 implementation decision

V107 uses deterministic-first routing. Existing exact Bargaining and Negotiation mechanics remain authoritative only within their already tested boundaries; the Discord excerpt does not justify a new fixed share or price. The shared planner and selector receive an explicit own-payoff invariant, including that opponent payoff ratio and damage have no independent selection value.

Persuasion v2.8 adds one narrow cloud bypass. After at least 4 visible current-game passes and a smoothed buy rate no greater than `0.2`, a seller uses a quality-consistent signal and skips both cloud stages. The trigger is current-game causal evidence, reversible through the hot-policy pointer, and explicitly not a permanent buyer-type inference. All other seller states retain v2.7's anchor-plus-alternative comparison, and buyer controls remain unchanged.

The timing scheduler adds a `hidden-only` scope. HI games preserve the game-pinned lexical and timing persona; KI games release once computation and shared API spacing permit. Receipts preserve identity mode, scope application, natural latency, requested delay, applied delay, deterministic branch, and model-call provenance for paper stratification.

## Paper claims and non-claims

The competition paper can claim that DeepRMM-01 evolved toward a hybrid controller in which exact mechanics and bounded local policies decide clear states, learned opponent models compare uncertain alternatives, and cloud reasoning is reserved for unresolved strategic or linguistic dimensions. It can measure model calls avoided, latency, worker occupancy, agreement and payoff by family and role, and outcomes across immutable policy epochs.

The paper must not claim that Discord tactics caused an improvement, that v2.8 is an equilibrium solution, that a pass sequence reveals an immutable opponent type, that response time directly changes rating, or that leaderboard movement isolates architecture quality. Live opponents, operator activity, configuration mix, rating shrinkage, and policy adaptation make the environment nonstationary.
