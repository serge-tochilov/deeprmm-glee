# GLEE Competition 2026 paper

This directory contains the compact, candid DeepRMM-01 competition-paper draft for GLEE 2026. Its central claim is narrower than a leaderboard claim: explicit mental-state prose showed no measurable incremental value in the tested development path, while compact statistical and action-conditioned models became operationally usable under bounded authority. Leaderboard rank and ratings remain descriptive.

The quantitative final-deployment claims are bound to the exact newline-terminated journal prefixes recorded in [`evidence.json`](evidence.json). The public-safe [`all-history execution summary`](../receipts/analysis/all-history-execution-summary.json) reconstructs campaign-wide aggregates from a SHA-256-bound private receipt without exposing journal-level paths or identifiers, and its nested final-v108 values match `evidence.json`. [`evidence-sources.md`](evidence-sources.md) maps the remaining manuscript claims to content-hashed repository artifacts. Later journal growth cannot alter either frozen evidential boundary.

The draft uses the official NeurIPS 2026 style in `sglblindworkshop` mode, matching the competition's lightweight single-blind review and retaining submission line numbers, with APA 7 citations and references rendered by `biblatex-apa` and Biber. The 4 content pages are followed by references, Appendix Table A1, and the completed mandatory NeurIPS checklist. The current official competition instructions give the extended deadline as September 5, 2026 Anywhere on Earth.

The checklist answers rest on current evidence and make no replay claim. Released deidentified sufficient inputs and fixed estimators reproduce the KI/HI inference, held-out learned-model comparisons, dossier ablation, rating estimates, and execution telemetry at the precision reported in the paper. Exact retraining and replay of the live competition remain outside the claim. Every applicable checklist item is supported with a `Yes`, while genuinely inapplicable theory, safeguard, broader-impact, and human-subject items use `N/A`.

## Working section contract

The manuscript follows the organizer's requested structure directly:

- **Agent overview and motivation:** test whether persistent RMM can model an adapting opponent's responses and the agent's public predictability.
- **Technical approach:** describe the credential-owning supervisor, deterministic family engines, Codex-mediated offline update loop, Terra planner and selector, statistical packages, conditional twin, self-mirror, and transactional evidence path.
- **Strategic design choices:** explain payoff and reconstructed-rating reasoning, probabilistic strategic reconnaissance, KI/HI separation, SIC, uncertainty bounds, and the authority assigned to each component.
- **Development process:** report the progression from an action--belief--desire--emotion ledger and prose dossiers to statistical and executable models, including failed approaches, chronological replay, and transport failures.
- **Evaluation and Agent Behavior Analysis:** report predictive metrics, route and validity receipts, the mixed family-specific KI/HI result, deviations from intended behavior, and the controls that bound those deviations.
- **Limitations and reproducibility:** state that the low-dimensional repeated games favored heuristics and data volume, quantify the agent's limited compute and game-count position, avoid attributing rating causally to RMM, and identify the immutable artifacts that reproduce the supported claims.

The intended contribution is a candid architecture and negative-results report; leaderboard rank remains descriptive. The KI/HI result must remain family-specific: HI outcomes were better in Negotiation and Persuasion in the reviewed post-SIC epoch, while KI outcomes remained better in Bargaining.

## Supporting artifact

The supporting repository is [`serge-tochilov/deeprmm-glee`](https://github.com/serge-tochilov/deeprmm-glee). Its curated release history is derived from the immutable final repaired runtime snapshot and the promoted conditional-twin and public-self-mirror snapshots. It contains the final live runtime and model-code closures, pinned policies, focused tests, paper source and PDF, deidentified sufficient analysis tables, public-safe analytical receipts, scoped licenses, credits, and verifiable provenance. Participant-derived corpora, trained weights, identity-bearing model artifacts, and private operational history remain excluded.

Build with `./build.sh`. The script fixes `SOURCE_DATE_EPOCH` and `TZ` for byte-stable builds from identical sources and toolchains; the generated `main.pdf` is retained as a review artifact, and temporary LaTeX files are ignored.
