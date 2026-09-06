# Executable opponent models v1

**Status:** Implemented as a shadow-only bargaining prototype on 2026-08-09; no model output can affect a live GLEE action

**Research use:** Detailed methods draft intended to support a later conference paper; hypotheses, promotion gates, and analysis choices should be frozen before confirmatory evaluation

## Research question

The experiment asks whether a persistent NoMMD agent benefits from maintaining an executable probabilistic model of each recurring opponent in addition to its natural-language action–belief–desire–emotion memory. The executable model must predict concrete behavior under a specified game state, expose uncertainty over competing explanations, support controlled simulation, and fail visibly when the current state lies outside its evidence base.

The existing prose dossiers answer questions such as what an opponent may believe, want, fear, or expect from DeepRMM-01. They are useful for hypothesis generation and for representing recursive mental states, but they are difficult to calibrate and cannot be replayed mechanically. The executable layer asks a narrower question: given the observable history available before an opponent acts, what distribution does the opponent place over legal next actions? Its predictions can be scored without treating a model-generated explanation as ground truth.

## Hypotheses

**H1 — opponent adaptation:** A population-informed, opponent-specific executable model will predict held-out actions better than a population-only model and an unconditional opponent-history baseline.

**H2 — program structure:** A posterior over compact behavioral programs will generalize across sparse opponent histories better than a single fitted point policy with comparable inputs.

**H3 — qualitative–executable complementarity:** ABDE dossiers will improve program proposal, regime interpretation, and order-2 feature design, while authenticated action traces will remain the evidential authority for behavioral prediction.

**H4 — decision value:** Counterfactual action selection against a calibrated opponent model will improve expected payoff or final ranking utility over the existing generic policy after deterministic safety guards, uncertainty penalties, and distribution-shift checks are applied.

**H5 — nonstationarity:** Explicit policy-change detection will outperform indefinite pooling when a competing team changes its runner, prompts, fallback logic, or strategic objective.

H1 is testable with the current bargaining corpus. H2–H5 require stronger baselines, prospective shadow predictions, ablations, or controlled policy interventions and are not established by the v0 implementation.

## Unit of analysis and epistemic boundary

The modeled entity is the opponent's exposed decision policy in one GLEE family, not the opponent's cloud model, hidden prompt, operator, software stack, subjective experience, or immutable identity. The same named endpoint may change policy over time, and several endpoints may share code. A fitted program is therefore a predictive twin of observed behavior under the recorded support, not a recovered copy of the opponent's internal mechanism.

Let (x_t) be the engine-authenticated information visible immediately before an opponent action, (h_t) the visible interaction history, (a_t) the action, (z_t) a latent policy regime, (m_t^{(1)}) the opponent policy model, and (m_t^{(2)}) a bounded representation of the opponent's model of DeepRMM-01. The predictive target is

\[
p(a_t\mid x_t,h_t,D_o)=\sum_{p,\theta,z_t,m_t^{(2)}}p(a_t\mid x_t,h_t,p,\theta,z_t,m_t^{(2)})p(p,\theta,z_t,m_t^{(2)}\mid D_o),
\]

where (D_o) contains only causally prior evidence for opponent (o). Order-1 RMM is represented by the posterior over the opponent's policy and state. An order-2 component is admitted only when a variable about the opponent's model of us changes an observable prediction, such as a reaction to our previous concession, a learned estimate of our reservation boundary, or a response to what we have revealed. Higher recursive levels remain qualitative hypotheses unless repeated evidence supports a bounded predictive variable.

The ABDE ledger and executable model have separate epistemic roles. ABDE can propose latent motives and recursive hypotheses; the executable layer turns selected hypotheses into falsifiable action distributions. A natural-language claim gains predictive authority only after it is compiled into an observable feature or program component and improves chronological held-out performance.

## Evidence and provenance

The v0 source is the immutable v3 named-opponent job tree at `opponent-dossiers/incremental-v3/jobs/<opponent_id>/<job_id>.json`. Each job contains the terminal authenticated game state, server-supplied opponent identity, completion time and order, source-run reference, final-game SHA-256, and complete visible bargaining history. The extractor reads these jobs directly and does not use the prose dossier as a behavioral label.

Every offer–response history entry yields exactly one modeled opponent action. If the opponent proposed the offer, the row is an opponent proposal; if DeepRMM-01 proposed it, the recorded decision is an opponent response. This avoids counting our own decisions as opponent behavior. Each row retains the source job ID, path, file hash, game hash, game ID, opponent ID, completion coordinates, role, round, information regime, visible discount factors, prior offers and responses, message availability, message content or class, and response latency when the platform records it.

Incomplete-information boundaries are preserved. An opponent discount factor absent from the state visible to DeepRMM-01 remains missing; the extractor does not reconstruct it from terminal outcomes or other private parameters. Hidden-opponent games remain outside named-model training until a separate identity-mixture protocol is validated. Timeout, cancelled, and abandoned games are censored rather than silently treated as strategic rejection, while their exclusion is recorded in the corpus artifact.

The corpus manifest hashes the ordered source-receipt list. Every fitted artifact records all contributing game IDs and job hashes, the population evidence count, the fitting grammar, posterior truncation mass, support summary, and its own content hash. A model can therefore be traced back to exact authenticated games without committing the live generated dossier tree to Git.

## Population prior and opponent update

Sparse opponent histories require partial pooling. The candidate set contains compact executable programs (p_j). Population evidence from other opponents provides a tempered prior, and the target's own causally prior rows update that prior:

\[
\log \tilde{\pi}_j=\frac{\log p(D_{-o}\mid p_j)}{\max(1,n_{-o}/\kappa)}-\lambda C(p_j),
\]

\[
\pi_j=(1-\epsilon)\operatorname{softmax}(\log \tilde{\pi})_j+\epsilon/|P|,
\qquad
w_{o,j}\propto \pi_jp(D_o\mid p_j).
\]

Here (D_{-o}) is population evidence, (D_o) is opponent-specific evidence, (C(p_j)) counts active contextual terms, (kappa) is the effective population sample size, (lambda) penalizes complexity, and (epsilon) preserves nonzero support for programs disfavored by the population. Tempering prevents hundreds of population rows from making a small opponent history unable to move the posterior.

The artifact stores the highest-weight programs and renormalizes their weights, together with the posterior mass retained by truncation. This is a deployment approximation; confirmatory evaluation uses the full candidate set and must report sensitivity to the truncation level, prior strength, grammar resolution, and complexity penalty.

## Bargaining twin v0

The first implementation models 2 linked opponent acts: acceptance of our offer and the opponent share requested in its own proposal. Money values are normalized by the current pool so behavior can transfer across pool scales, while an unseen pool scale still raises an out-of-distribution flag because absolute denominations may influence model or operator behavior.

### Response program

For an offer giving the opponent share (s_t), candidate response program (j) predicts

\[
p_j(\text{accept})=\sigma\left(\frac{s_t-[\tau_j+d_jq_t+c_jI_t+r_jR_t]}{T_j}+e_jE_t\right),
\]

where (	au_j) is the acceptance threshold, (q_t\in[0,1]) is normalized round progress, (d_j) is deadline adaptation, (I_t) marks complete information, (R_t) marks the opponent's player role, (E_t) marks an approximately equal split, (T_j) controls stochasticity, and (sigma) is the logistic function. The program predicts agreement rather than motive. In v0, `reject` and `walkaway` are both non-acceptance outcomes; a later competing-risks model must separate them when the corpus supports reliable estimation.

### Proposal program

Candidate proposal program (j) predicts a Gaussian distribution over the opponent's requested share with clipped mean

\[
\mu_{j,t}=\operatorname{clip}\left(a_j+c_jI_t+r_jR_t+\rho_j(o_{t-1}-a_j)+\gamma_jq_t(0.5-a_j),0,1\right),
\]

where (a_j) is the anchor, (o_{t-1}) is the share we most recently offered that opponent, (ho_j) is reactive matching, (gamma_j) is concession toward parity, and the residual standard deviation (sigma_j) represents stochastic or unmodeled variation. Posterior prediction is the corresponding weighted Gaussian mixture, from which the implementation reports the mean and 10th, 50th, and 90th percentiles.

### Candidate grammar

| Component | v0 values |
| --- | --- |
| Response threshold | 0.20–0.80 on a nonuniform grid including 1/3 and 2/3 |
| Response temperature | 0.03, 0.08, 0.16 |
| Deadline slope | −0.12, 0, 0.12 |
| Information and role offsets | −0.05, 0, 0.05 |
| Equal-split logit bonus | 0, 1.5 |
| Proposal anchor | 0.20–0.80 on the same nonuniform grid |
| Reaction to our prior offer | 0, 0.5, 1.0 |
| Concession toward parity | 0, 0.5, 1.0 |
| Proposal residual standard deviation | 0.03, 0.08, 0.16 |

The grid is intentionally finite and hand-auditable. It is expressive enough to represent stable anchors, reciprocal adjustment, deadline concession or hardening, role asymmetry, information-regime shifts, equal-split preference, and stochastic choice, but it cannot recover arbitrary prompts or long-horizon planning algorithms. Grammar expansion should follow systematic residuals rather than retrospective storytelling.

### Message acts

Messages are mapped deterministically into a small strategic vocabulary: no message, unsupported authority or equilibrium claim, explicit allocation, fairness plus urgency, fairness, urgency, commitment, and other. The v0 message model estimates only the opponent's distribution over these acts using a smoothed population prior. It does not yet condition numeric offers or acceptance on message semantics, so the Rubinstein-label incident remains evidence motivating a later joint action–message model rather than a solved case.

## Chronological evaluation

Random row splitting would leak an opponent's later policy into its earlier predictions and place actions from the same game on both sides of the split. Evaluation therefore sorts complete games by authenticated completion time, completion order, and game ID; holds out the latest fraction of each eligible opponent's games; and keeps every row from one game in the same partition.

For each target, population-prior evidence is also restricted to games completed before the target's first held-out game. This prevents another opponent's future game from entering the target's historical prediction through the hierarchical prior. Final deployment artifacts may use all evidence available at generation time, but they are distinct from chronological evaluation artifacts.

The v0 reports response negative log likelihood, Brier score, threshold accuracy, proposal mixture negative log likelihood, proposal mean absolute error, root mean square error, and empirical coverage of the nominal 80% proposal interval. Its implemented comparison is an explicitly weak unconditional empirical baseline fitted from causally available population plus target training data. A paper-grade experiment must add population-only particles, opponent-only particles, recency-weighted empirical models, regularized tabular or tree models, direct-LLM prediction, prose-dossier prompting, and a target-adaptive text–tabular baseline.

Primary statistical analysis should average first within opponent and configuration strata, then across opponents, rather than allowing frequent opponents or long games to dominate. Confidence intervals should be obtained by block bootstrap over opponents and complete games. Hyperparameters and grammar revisions must be selected on development opponents or nested chronological validation, never on the reported test suffix.

## Distribution shift and policy change

An executable opponent model is unsafe if a team silently changes code. The v0 emits 2 distinct warnings. Support flags mark a current context with an unseen information regime, horizon regime, role, pool scale, or later round than observed. A recent-surprise heuristic compares recent-game predictive negative log likelihood against the earlier fitted regime and reports `stable`, `watch`, or `candidate-change`; this label is explicitly not a posterior probability of a change point.

A paper-grade version should replace the heuristic with online Bayesian change-point detection or a hidden Markov regime model, report detection delay and false-alarm rate on synthetic and naturally observed policy changes, and maintain both a long-run identity model and a recency-weighted active-regime posterior. When change evidence is strong, the live system should broaden uncertainty or fall back to the population policy rather than confidently extrapolating stale behavior.

## Counterfactual simulation and active information gathering

The posterior defines an opponent simulator rather than one deterministic clone. Candidate DeepRMM-01 actions can be evaluated against sampled programs and stochastic responses, producing expected payoff, lower-tail payoff, agreement probability, and sensitivity to model uncertainty. Robust selection should optimize a risk-aware objective over posterior samples, not a best response to the maximum-a-posteriori program.

Counterfactual outcomes outside observed support are model implications, not evidence about what the real opponent would have done. The analysis must identify overlap, distance from support, and posterior disagreement for every claimed counterfactual gain. A simulator can improve decision search while still being wrong about causal mechanism.

Controlled probes become defensible only after shadow calibration. A probe must be a legal, ordinary, payoff-seeking move; have bounded worst-case cost; discriminate among high-posterior hypotheses; and avoid revealing more about our policy than its expected information value warrants. Regret and information gain should both be logged. No live game should be sacrificed merely to make the opponent easier to study.

## Relation to recursive mental modeling

The executable layer does not replace NoMMD's ABDE tetrad. It constrains the behavioral consequences of recursive hypotheses. For example, “the opponent believes our reservation share is low” becomes a latent order-2 parameter only if it predicts how the opponent changes anchors, concessions, acceptance, or messages after our behavior. “The opponent expects our authority language to work” becomes testable through a conditioned message-response program. Unsupported mental-state detail remains a low-authority hypothesis in the dossier.

The intended online loop is: authenticated evidence updates the action history; ABDE retrieval proposes relevant motives and nested beliefs; executable candidates predict actions; predictive failures revise program weights or trigger a regime hypothesis; simulated responses inform a bounded action search; deterministic legality and loss guards retain final authority; and the resulting observation closes the loop. This architecture separates expressive recursive modeling from calibrated behavioral prediction without assuming a central homunculus that knows the opponent's true mind.

## Negotiation extension

Negotiation requires a joint posterior over reservation value, offer policy, acceptance policy, and walkaway policy. Candidate programs should condition on our revealed offers, inferred feasible surplus, role, horizon, previous messages, and an explicit estimate of what the opponent believes about our reservation boundary. Predictions should include next offer, accept, reject, or walkaway probabilities and a calibrated distribution over terminal agreement values.

The main evaluation complication is selective revelation: agreements censor how the same opponent would have behaved at later offers, while walkaways reveal only a boundary interval. A likelihood should model this event process rather than treating missing later rounds as random. The same chronological, population-prior, support, and change-point rules apply.

## Persuasion extension

Persuasion requires a partially observed state-space model. Candidate programs should represent seller reliability, reputation-building and harvesting regimes, buyer trust, evidence accumulation, quality-conditioned signaling, purchase quantity, and the opponent's estimate of our trust or exploitation threshold. A hidden Markov model or small probabilistic program can update these states after each sale and outcome.

Because seller and buyer roles expose different information and incentives, role-specific models should be fitted before attempting a shared parameterization. The model must distinguish strategic deception from stochastic product quality and avoid using quality information that was unavailable at decision time. Evaluation should report calibration of action probabilities, state-filtering likelihood, payoff regret, and the timing of detected reputation shifts.

## Shadow-to-live promotion gates

1. **Provenance gate:** all training and evaluation rows must resolve to immutable authenticated receipts, preserve information masks, and pass actor-attribution tests.

2. **Predictive gate:** the model must improve chronological held-out proper scores over preregistered strong baselines across multiple opponents and configurations, not only in pooled averages.

3. **Calibration gate:** probability reliability, interval coverage, posterior truncation error, and out-of-distribution behavior must be acceptable under a frozen criterion.

4. **Decision-value gate:** teacher-forced replay and controlled simulation must show positive risk-adjusted payoff or ranking utility after deterministic guards, with no material increase in catastrophic low-share actions.

5. **Latency gate:** model loading, projection, and action search must fit inside the existing deadline reserve without reducing the reasoning worker's usable window.

6. **Adversarial gate:** authority language, ratio-order ambiguity, unseen roles, extreme offers, sparse histories, policy switches, artifact corruption, and poisoned or malformed receipts must fail closed.

7. **Prospective gate:** a version-pinned shadow deployment must register predictions before outcomes for a fresh game suffix. Retrospective fit alone cannot authorize live control.

Promotion requires a new protocol revision and explicit review. The v0 code deliberately has no import path from the live supervisor, worker, broker, or deterministic guard.

## Implementation and artifacts

`src/nommd_arena/glee_bargaining_twin.py` implements exact extraction, candidate programs, population-tempered Bayesian updating, chronological evaluation, artifact hashing, deterministic sampling, support flags, message-act frequencies, and the preliminary change signal. `glee-bargaining-twin` is an offline command that writes `corpus.json`, `evaluation.json`, `manifest.json`, and one `opponents/<id>/bargaining/model.json` artifact per eligible target.

The reproducible shadow command is:

```bash
UV_CACHE_DIR=/tmp/artifact-uv-cache uv run nommd-arena glee-bargaining-twin --dossier-root opponent-dossiers/incremental-v3 --output-dir /tmp/glee-bargaining-twin-v0 --min-games 10
```

The generated tree belongs in an immutable experimental receipt or large-file archive only after review; it is not added to the live dossier tree. The Git-tracked implementation and protocol contain no opponent evidence and no credential.

## Threats to validity

Opponent matching is endogenous: active and successful agents may appear more often, and DeepRMM-01's current policy determines which states are observed. Predictive accuracy under this behavioral policy does not establish accuracy under substantially different interventions.

Named identity is a platform label, not proof of one stable policy or operator. Shared scaffolds create correlated opponents, and a named endpoint can update during collection. Hierarchical estimates and uncertainty intervals must account for clustering and time.

The current candidate grammar is researcher-designed after observing part of the competition. It may encode incident-specific hindsight, and later grammar expansion can overfit. Development and confirmatory opponents, dates, and configurations must be separated.

Natural-language messages are reduced to a coarse deterministic vocabulary in v0. Stylistic details, strategic framing, prompt-injection-like language, and semantic commitments may explain behavior that the numeric model attributes to noise.

The observed action is one stochastic realization from a scaffold that may itself race several model branches and deterministic fallbacks. A compact mixture can predict that endpoint without recovering its actual computation. Interpretability claims must remain behavioral.

Payoff improvement is not equivalent to evidence of deeper RMM. A fixed arithmetic exploit may outperform a rich recursive model. The paper should report predictive depth, behavioral adaptation, and competitive utility as separate outcomes and use the lowest-order explanation sufficient for each effect.

## Paper-grade experimental sequence

1. Freeze the extractor, candidate grammar, eligibility threshold, chronological split, baselines, metrics, and exclusion rules on a versioned development corpus.

2. Run actor-attribution and information-mask audits on sampled receipts, including multi-round games, both roles, complete and incomplete information, no-message games, walkaways, and extreme allocations.

3. Compare population-only, target-only, hierarchical executable, direct-LLM, prose-dossier, text–tabular, and ablated executable models on a sealed chronological suffix.

4. Report per-opponent and per-configuration proper scores, calibration, interval coverage, posterior concentration, support flags, and change-point behavior with block-bootstrap uncertainty.

5. Evaluate counterfactual decision value first in teacher-forced replay and then in a prospective shadow period whose predictions are committed before outcomes.

6. Promote a bounded advisory projection only if all gates pass; compare generic policy, prose dossier, executable twin, and combined ABDE-plus-twin under a versioned online or matched simulation design.

7. Preserve null results, negative-transfer opponents, policy switches, and semantic failures as primary evidence about the limits of literal opponent modeling.

## Research precedents

[ROTE: Modeling Others' Minds as Code](https://arxiv.org/abs/2510.01272) synthesizes compact behavioral programs with an LLM and performs probabilistic inference over them, reporting up to 50% improvement over behavior-cloning and LLM baselines in its tested environments. It motivates program-space uncertainty but does not establish that the inferred code is the actor's true mechanism.

[RevengeBench](https://arxiv.org/abs/2606.26094) treats policy recovery as an inverse problem in code space, adds controlled opponent-policy probes, and evaluates recovered programs both by action distance and downstream tournament value. It motivates iterative behavioral recovery and explicit intervention while reinforcing the distinction between raw traces and explanatory prose.

[Predicting Decision-Making Behavior of Unfamiliar Agents in Economic Games](https://arxiv.org/abs/2605.12411) studies target-adaptive prediction on bargaining and negotiation data closely related to GLEE. Its structured state, dialogue, and prior-game formulation supplies a strong task-specific baseline: at 16 prior games, the reported Observer representation improves response AUC by about 4 points and reduces bargaining offer error by 14% within its tabular architecture.

[Repeated Negotiation via Smooth Fictitious Play](https://arxiv.org/abs/2602.19309) uses an auxiliary opponent model to imitate time-averaged behavior and best-of-N simulation to improve repeated strategic decisions without parameter updates. It motivates the later decision-search layer but averages over behavior in a way that may obscure abrupt policy changes.

[Hierarchical Opponent Modeling and Planning](https://proceedings.mlr.press/v235/huang24p.html) separates goal inference, goal-conditioned opponent policy, and Monte Carlo tree-search response planning, with few-shot adaptation within and across episodes. It motivates keeping latent intent, observable policy, and planning as distinct levels.

[Autonomous Agent Modelling: A Comprehensive Survey and Open Problems](https://arxiv.org/abs/1709.08071) provides the broader taxonomy of policy reconstruction, type-based reasoning, recursive reasoning, and opponent modeling against which the NoMMD executable layer should be situated.
