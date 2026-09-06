# GLEE Bargaining live policy v2.20

**Status:** Replay-approved persistent-RMM policy release for newly assigned games

**Advisor model:** `bargaining-live-advisor-v2.17`

**Policy revision:** `bargaining-v2.20-strong-continuation-margin`

## Scope

V2.20 leaves the sealed v2.17 advisor, exact-share interpolation, statistical package, rating model, analytic hierarchy, cloud transport, nonterminal own-share floor, and bidirectional continuation-coherence feature unchanged. It raises `continuation_tolerance` from `0.02` to `0.05` under the existing declarative contract.

## Strong continuation margin

The acceptance-to-rejection override now requires modeled rejection continuation to exceed current acceptance by more than `0.05`. The threshold preserves the historical doodod repair, whose continuation advantage was `0.079699`, while withholding authority in 3 new v59 cases with advantages `0.038742`, `0.020041`, and `0.048229`; the model had proposed acceptance in all 3, and their observed v2.19 outcomes lost `-11.8` displayed-rating points in total.

The same tolerance remains part of the existing no-better-next-settlement and rating-loss composition. This coupling is inherited from the stable declarative contract and is not interpreted as an ideal final design. A future split threshold requires executable contract work and cannot be introduced through this hot release.

## Replay boundary

The release was replayed over 471 preserved selected turns from v48, v49, v53, v55, and a frozen 53-game v59 prefix. Relative to v2.19, it preserves every previously approved repair, restores the 3 v59 model-proposed acceptances described above, and adds one historical rating-loss rejection on an observed `-6.8` outcome. Relative to each preserved source policy, the complete replay changes 10 first observed-prefix actions and no downstream-censored turn.

## Deployment boundary

Policy assignment remains immutable per game. Atomic pointer publication affects only games first assigned after v2.20 becomes current; active v2.19 games retain v2.19. The live Bargaining family continues its independent Poisson stream without pause, drain, or restart.

## Evidence

The development record is `glee-bargaining-v59-v2-19-development-20260814`, and its generated replay summary is `v2.20-release-replay/summary.json`; both remain in the private archive.
