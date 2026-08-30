# Tasks: sunscreen recommendation MVP

## Critical path

- [ ] **CP1 — Freeze sunscreen artifact contract.** Define fixed category constants, minimal path
  config, schemas, manifest fields, price wording, and evidence-count gate.
- [ ] **CP2 — Build scoped products and prices.** Select exactly `مراقبت پوست > کرم ضد آفتاب`;
  derive lowest valid positive historical price and brand fields.
- [ ] **CP3 — Single-pass scoped comment build.** Stream raw comments once, reuse cleaning
  semantics, retain only matching candidates, and write raw scoped comments.
- [ ] **CP4 — Canonicalize scoped comments.** Deterministically deduplicate only matching comments
  in bounded memory; write canonical and audit artifacts with provenance.
- [ ] **Checkpoint A — Data gate.** Manifest verifies 1,048 products, 53,522 raw comments, 892
  commented products, 175 brands, 1,046 priced products / 99.81%, 50,275 buyer comments, and
  53,517 non-empty bodies.
- [ ] **CP5 — Baseline retrieval and ranking.** Implement deterministic Persian lexical retrieval,
  brand/price filters, transparent score components, and stable tie-breaks.
- [ ] **CP6 — Evidence-backed comparison.** Compare two or more products with strengths,
  weaknesses, counts, historical-price labels, and bounded ID-tagged excerpts.
- [ ] **Checkpoint B — Recommendation gate.** Fixed queries yield deterministic ordered IDs and
  every rendered claim/excerpt is traceable.
- [ ] **CP7 — Local presenter demo.** Add a CPU-only local flow for query, filters, shortlist, and
  two-product comparison.
- [ ] **CP8 — Fixed evaluation and documentation.** Add Persian evaluation data, run it offline,
  document commands/limitations, and prepare presenter script.
- [ ] **Checkpoint C — Release gate.** Full test suite, lint, evaluation, clean diff check, and
  demo rehearsal pass.

## Exact test expectations

- [ ] Builder integration test asserts the complete evidence baseline in Checkpoint A and all
  artifact schemas/manifest fields.
- [ ] Instrumented builder test asserts exactly one invocation/pass of the raw-comment iterator;
  a repository test asserts no SQLite dependency, file, or connection is introduced.
- [ ] Canonicalization tests cover duplicate IDs, conflicts, missing IDs, stable tie-breaks, and
  equality of canonical output across at least two chunk sizes.
- [ ] Bounded-memory test fixture proves only matching candidates are accumulated; non-matching
  comments are not retained.
- [ ] Filter tests cover exact brand matching, inclusive historical-price bounds, and products
  with unknown historical prices both with and without a price filter.
- [ ] Retrieval tests assert deterministic ordering and score breakdowns for every fixed Persian
  evaluation query, including stable `product_id` ties.
- [ ] Comparison tests require at least two products, product/review counts, strengths,
  weaknesses, historical-price-or-unknown display, and excerpts no longer than the configured
  maximum with valid `{product_id, comment_id}` references.
- [ ] Safety snapshot tests prohibit current-price, availability, diagnosis, treatment, and
  suitability-guarantee language; unsupported SPF/skin-type/texture claims are absent or unknown.
- [ ] Evaluation runner test reads the fixed Persian JSONL without network/GPU access and reports
  case-level pass/fail deterministically.

## Optional tasks — drop first under deadline pressure

- [x] **O1a — Semantic artifact checkpoint.** From published sunscreen artifacts only, embed the
  53,365 canonical comments with `intfloat/multilingual-e5-small`; use `passage: ` prefixes,
  normalized float32 vectors, explicit vector/comment provenance, atomic staging, checksums, and
  `_SUCCESS`. This task changes no retrieval behavior.
- [x] **O1b — Hybrid retrieval.** Validate manifest/_SUCCESS/checksums and mmap metadata, cache
  CPU query embedding, retrieve exact normalized dot-product comment hits, use bounded max-hit
  product aggregation, and fuse lexical/dense ranks with RRF `k=60` plus stable provenance ties.
- [x] **O1c — Hybrid evaluation and safeguards.** Add low-overlap Persian paraphrases; compare
  lexical, semantic, and hybrid IDs/evidence/latency/determinism; assert lexical fallback and
  prohibit vectors or raw scores from Groq context and normal UI.
- [x] **O1d — Recommendation-status Macro-F1 baseline.** Evaluate preserved canonical labels with
  a deterministic product-group holdout, most-frequent dummy and character-TF-IDF logistic
  baselines, Macro F1/per-class metrics/confusion matrix, and leakage-safe reports.
- [ ] **O2 — Kaggle GPU preparation.** Supply an optional reproducible experiment setup only; no
  hosted runtime is required for final presentation.
- [ ] **O3 — Extra visual polish.** Improve presentation styling only after the scripted demo and
  evidence traceability are complete.

## Final presentation deliverables

- [ ] Generated scoped Parquet artifacts and manifest satisfying Checkpoint A.
- [ ] A local, CPU-only start command and short presenter script in Persian.
- [ ] Three scripted demos: filtered query, evidence-backed two-product comparison, and a safety/
  unknown-attribute example.
- [ ] Fixed Persian evaluation set plus the recorded offline result.
- [ ] One-page provenance/limitations note: scope boundary, historical-price disclaimer, review
  evidence IDs, deterministic baseline, and no medical claims.
- [ ] Final verification record: tests, lint, evaluation, `git diff --check`, and repository
  status; no commit is required.
