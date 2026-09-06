# Public-history disclosure audit

## Status

This receipt records the disclosure audit for the final sanitized release tree. The private source repositories remain the rollback and source authority. Private remote staging is authorized; public visibility remains withheld.

## Source and construction

The computational release source was clean and upstream-equal at `1a6625a39bf6cc95947b6b1815e4acb5565c6063`. The clean stage was assembled from that tracked tree rather than by rewriting its history. The submitted paper source and PDF were refreshed from canonical commit `a0566771b6025ae636a11fb2f025d1f943afc5ee`. The complete `artifact/nommd-arena/seeds/` and `artifact/nommd-arena/models/` trees were omitted; no credential, browser profile, mutable event journal, cloud-call journal, or Git metadata was copied.

The public tree preserves the final runtime, prompts, policy releases, model-training and inference source, paper, deidentified 752-row KI/HI input, 8 metric-level sufficient Parquet tables, aggregate receipts, and reproducibility tooling. It withholds 3 raw corpora containing 10,915 games, participant-authored messages, stable participant and game identifiers, inferred account links, exact timestamps, identity-bearing model vocabularies, learned weights, and mutable operational stores. Released row-level predictions are stripped of those fields and use unrelated table-local group labels.

## Recorded transformations

`paper/evidence.json` replaces 2 local private paths with logical private-archive references and binds the source private receipt by SHA-256. `receipts/v108/public-launch.json` replaces the private launch manifest with a minimal public-safe derivative bound to that manifest's SHA-256. The public tactic ledger removes 24 private game identifiers from 4 provenance arrays, replacing each array with an aggregate game count while leaving tactic conditions and guidance unchanged; both the source semantic hash and public derivative hashes are recorded. Two private-derived fixture labels in tests were replaced with neutral labels. Two protocols were generalized to remove an actual game identifier and named-account whitelist while retaining their technical contracts. Private console logs and path-bearing checksum manifests were omitted.

The sanitation pass extracts the selected Codex execution path into a self-contained local transport, removes unselected model backends and the unrelated historical package boundary, canonicalizes the final deployed prompt filenames, removes runtime prompt-variant machinery, and regenerates dependency locks. The metric-level data projection retains only evaluation targets, frozen outputs, table-local grouping structure, resource fields, and execution measurements. Private snapshot inventories containing unrelated path names are omitted; their hashes remain in `provenance/source-authorities.json`, and their exact bytes remain recoverable from the private archive.

The curated paper bibliography cites the published NoMMD Version 3 title, date, and version through its stable concept DOI. The reproducible paper PDF has SHA-256 `00d6586d515dea8a5510cd057ad0c728e8d3acf185698a9d3e45b5fa9cc47b62`; the concept DOI resolves to frozen Version 3 DOI `10.5281/zenodo.21926873`.

## Static disclosure scans

The precommit candidate file set was scanned recursively across text and structured files, excluding declared build, cache, and runtime products. No local home path, RFC 4122 UUID, common GitHub/OpenAI/Google/AWS/Slack credential prefix, PEM private-key header, bearer token, credential-bearing filename, or unexpected email address was found. The only email addresses are the author's declared contact in `paper/main.tex` and Roman Garnett's attribution comment in the upstream NeurIPS style file. Test-only `GLEE_API_KEY` values are visibly synthetic fixtures and do not match any live credential.

An exact private-name comparison extracted 76 corpus display labels; after separating generic words, synthetic fixtures, and one public collision-label example embedded in the frozen implementation, none of the 63 private corpus labels selected for disclosure checking occurred in the clean stage. No stable participant ID or game UUID remains. Public labels in frozen explanatory code are not accompanied by stable IDs, game records, messages, or account mappings.

The message audit normalized 14,539 unique messages from all 10,915 private games and compared them with every public text file. It found 28 literal overlaps: 26 were authored only by DeepRMM-01 and occur in its fixed message templates or deterministic policy code; the remaining 2 were the generic fixtures `I recommend buying this product.` and `I recommend this product.`, used by both sides and present in policy or test code. No opponent-specific prose survives.

## Executable verification

The current sanitation derivative synchronized from both regenerated locks and passed 260 Codex-only runtime tests plus 58 sequence-model tests with one intended optional GPU-path skip. The released KI/HI command completed its fixed 50,000-resample primary analysis and 20,000-resample sensitivity analyses; its output was byte-identical to `receipts/analysis/glee-v104-ki-hi-public-reproduction.json` with SHA-256 `ffe0fc8f3b98ec21c30e308fa96f078919880279da316b7dc19309b94dda89e1`. The metric-level command schema-validated all 8 Parquet tables, recomputed the declared estimators, and matched `receipts/analysis/reproduced-paper-metrics.json` exactly. Decoded categorical fields and anonymous labels passed their allowlists, the review paper rebuilt successfully, and the regenerated content manifest passed the artifact verifier and its explicit checklist-answer, excluded-backend, excluded-project, removed-prompt-experiment, credential, identifier, local-path, and email scans.

## Final-history verification

The repository is represented by one parentless sanitized commit containing the final state. Its complete reachable history and tree passed the disclosure scans, and a clean local clone reproduced the content manifest, both analytical receipts, focused test results, and review PDF. Private remote staging is authorized. Public visibility, submission tagging, and any OpenReview metadata update remain separate attended actions.
