# GLEE v104 KI/HI rating-delta inference

**Status:** Complete retrospective inference over the frozen first 797 v104 terminal games. The join reproduces the manuscript's family totals, 766 authenticated ratings, timeout exclusions, KI/HI sample counts, and all 6 published means exactly.

## Method

The estimand is mean authenticated displayed-rating delta in hidden-identity (HI) games minus the corresponding mean in known-identity (KI) games, computed separately for Bargaining, Negotiation, and Persuasion after excluding timeout outcomes. Each family's eligible games remain in terminal chronology. The primary analysis uses a circular moving-block bootstrap with 20-game blocks, 50,000 resamples, and seed `1729`; 10- and 40-game blocks receive 20,000-resample sensitivity checks. Null-centered block-bootstrap tests produce raw 2-sided probabilities. Bonferroni 98.333% per-family intervals and Holm-adjusted probabilities control the 3-family family-wise error rate at `0.05`.

## Results

| Family | KI, n (mean) | HI, n (mean) | HI minus KI | Unadjusted 95% CI | Family-wise 95% CI | Raw p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bargaining | 140 (`+0.735`) | 120 (`+0.545`) | `-0.190` | `[-1.195, +0.827]` | `[-1.412, +1.061]` | `0.715` | `1.000` |
| Negotiation | 107 (`+0.364`) | 133 (`+0.856`) | `+0.491` | `[-0.745, +1.679]` | `[-1.031, +1.916]` | `0.431` | `1.000` |
| Persuasion | 120 (`+0.181`) | 132 (`+0.539`) | `+0.359` | `[-0.598, +1.379]` | `[-0.805, +1.580]` | `0.487` | `1.000` |

All primary intervals include zero. Every 10- and 40-game sensitivity interval also includes zero. The cut therefore establishes directional descriptive differences but no statistically non-null family contrast.

## Interpretation boundary

The moving blocks address local temporal dependence in ratings and population state, and multiplicity control prevents selecting a favorable family after inspection. They do not identify a causal SIC effect: identity mode was not randomized, hidden opponents cannot be clustered by actual identity, and opponent, role, configuration, and policy composition varied. Failure to reject zero is not evidence that the true contrasts are exactly zero; the intervals are broad enough to include materially positive and negative effects.

## Reproduction boundary

The method is implemented in `tools/glee_ki_hi_inference.py`. The deidentified sufficient input is `data/glee-v104-ki-hi-eligible.jsonl`, and the exact public-path output is `receipts/analysis/glee-v104-ki-hi-public-reproduction.json`. Run `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv run --no-project python tools/glee_ki_hi_inference.py --eligible-input data/glee-v104-ki-hi-eligible.jsonl --output /tmp/glee-v104-ki-hi-results.json`; the generated file should be byte-identical to the released receipt. The private source path remains independently bound by `receipts/analysis/glee-v104-ki-hi-inference-summary.json`, but raw events and rating-history databases are not needed to reproduce the published analysis.
