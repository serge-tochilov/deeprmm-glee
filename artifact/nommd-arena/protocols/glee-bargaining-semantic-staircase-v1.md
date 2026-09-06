# GLEE Bargaining semantic-staircase guard v1

**Status:** Executable and replay-tested; not active in the currently running Bargaining process

## Purpose

The guard prevents an opponent from inducing repeated cloud inference through a low-information numeric staircase whose message intent stays stable. It treats such a trajectory as a compute-routing condition, not as evidence that the opponent intends resource exhaustion, will continue conceding, or will eventually accept.

## Detection contract

The current turn must be a Bargaining offer with an available message channel and an available frozen v2.17 advisor recommendation. The latest window must contain at least 5 opponent offers, at least 3 positive concessions toward DeepRMM, no reversal, no share step above 0.02, maximum positive-step dispersion of 0.35 relative to the mean, and a normalized message template occupying at least 80% of the window with no more than 2 distinct templates. Numeric fragments are removed before template comparison so a changing displayed ratio does not manufacture semantic novelty.

The detector classifies only the observed recent pattern. Its receipt explicitly withholds claims about motive, future concessions, hidden thresholds, model substrate, or strategic commitment.

## Action boundary

The guard never extrapolates the staircase and never creates a concession. It reads the current frozen advisor recommendation, compares that recommendation with DeepRMM's latest historical offer, and remains eligible only when the absolute opponent-share shift is at most 0.02. A larger shift is a policy-boundary event and receives full RMM inference.

When eligible, the guard submits the current advisor recommendation with deterministic surface prose, then passes the action through every ordinary safeguard and intervention. If any safeguard changes the numeric action away from that recommendation, the guard withdraws and full inference remains available. The branch discards cognitive updates because no cloud mentalizing occurred.

## Composition

Exact analytic and settlement authority remains first. Message-free authority remains second. The literal bilateral-plateau bypass remains third. The semantic-staircase guard runs fourth, before cloud inference. This ordering preserves every stronger existing decision authority and lets the cheaper exact-plateau proof retain its original receipt.

## Frozen replay

The target replay is one 86-round known-identity game. The guard first became eligible at round 10 and would have replaced 32 of the 37 historical Terra offer calls through round 74 while reproducing the frozen advisor's numeric action at every replaced prefix. It left cloud calls at rounds 2, 4, 6, 8, and 24, then yielded to the existing literal plateau branch for the final 6 offer turns. The game identifier and opponent label are omitted from the public artifact.

Those 32 replaced calls consumed 607,938 input tokens, 11,270 output tokens, and 348.246492 seconds of recorded worker time. These are counterfactual compute savings on the preserved prefix sequence, not a claim that downstream play would have remained identical after the first changed message or timing surface.

## Activation boundary

The current live Bargaining process loaded the earlier Python implementation and cannot acquire this branch through a policy-pointer update. Activation requires a controlled Bargaining process replacement with a resealed implementation receipt. Existing live work must be drained or migrated through the canonical controller; this protocol does not authorize a restart by itself.
