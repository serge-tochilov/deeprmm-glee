# Released reproduction data

`glee-v104-ki-hi-eligible.jsonl` is the deidentified input needed to reproduce the paper's KI/HI rating-delta analysis. It contains 752 chronologically indexed records and only 4 fields: `family`, `frontier_index`, `mode`, and `rating_delta`. It contains no opponent name, stable opponent identifier, game identifier, message, local path, or timestamp.

The input SHA-256 is `0e750482c09113bdc223ac8cf1e309c11ce7f1b9b2256598d47a1ce7d4c74913`. It was projected once from the frozen private v104 join by retaining only the sufficient fields consumed by the inferential analysis; the public and private-input paths produce exactly equal primary and sensitivity results.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-uv-cache uv run --no-project python tools/glee_ki_hi_inference.py --eligible-input data/glee-v104-ki-hi-eligible.jsonl --output /tmp/glee-v104-ki-hi-results.json` from the repository root. The result should be byte-identical to `receipts/analysis/glee-v104-ki-hi-public-reproduction.json`, whose SHA-256 is `ffe0fc8f3b98ec21c30e308fa96f078919880279da316b7dc19309b94dda89e1`.

The `reproduction/` directory contains 8 compact Parquet tables sufficient to recompute the paper's other principal quantitative results: held-out conditional-twin predictions, reversed-Persuasion predictions and baseline, public-self-mirror predictions, held-out rating estimates, the dossier comparison, and all-history plus final-v108 execution telemetry. Its [schema and privacy boundary](reproduction/README.md) are documented separately.

All released tables are licensed under `CC-BY-4.0` to the extent copyright or database rights subsist. Raw competition journals, messages, stable participant and game identifiers, exact timestamps, account links, sealed corpora, identity-bearing vocabularies, and trained weights are not required for metric recomputation and are not approved for public distribution.
