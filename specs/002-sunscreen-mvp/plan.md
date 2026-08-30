# Implementation plan: sunscreen MVP

## Technical approach

Build one narrow pipeline from canonical product/offer Parquet plus the raw comments CSV to four
sunscreen artifacts: scoped products, product historical prices, raw matched comments, and
canonical matched comments. Keep only scoped IDs and matching comment candidates in memory;
stream the raw CSV once using the established exact-token reader. No SQLite, vector service,
network dependency, or cross-category abstraction is planned.

The first usable recommender is a deterministic Persian lexical scorer. It token-matches a query
against title and canonical review text, combines that with explicit review-support counts, and
returns score components. Comparison uses only retrieved excerpts and deterministic aggregation.

## Artifact contract

| Artifact | Minimum contents | Purpose |
|---|---|---|
| `sunscreen_products.parquet` | product ID, title, brand, scoped categories | candidate catalogue |
| `sunscreen_prices.parquet` | product ID, lowest valid positive historical price, provenance | filter/display only |
| `sunscreen_comments_raw.parquet` | matching raw comment fields and source row | scoped audit baseline |
| `sunscreen_comments_canonical.parquet` | canonical comment plus IDs, selection provenance, cleaned text | retrieval/evidence |
| `manifest.json` | source fingerprints, counts, config, schema/version | reproducibility gate |
| `evaluation.jsonl` | fixed Persian cases and expected assertions | offline quality gate |

Historical-price display text is fixed to “قیمت تاریخی مشاهده‌شده” (or equivalent explicitly
historical wording). No UI code may reuse it as a current price field.

## Phased delivery plan

### 1. Scoped dataset builder — critical path

1. Define the sunscreen-only schema, output paths, thresholds, and manifest contract.
2. Load canonical products and select only the two fixed category labels; construct the scoped
   product-ID membership set.
3. Derive lowest valid positive historical price per scoped product from offers Parquet.
4. Stream raw comments once with `iter_exact_csv_batches`; reuse existing parsers/cleaners and
   retain only matching comment candidates.
5. Canonicalize matching comments deterministically in bounded memory, then write Parquet and
   manifest atomically.
6. Gate the phase on the fixed evidence counts in `spec.md`.

### 2. Deterministic baseline retrieval/ranking — critical path

1. Implement Persian-friendly deterministic normalization/token matching without a broad search
   framework.
2. Retrieve title/review matches from canonical scoped artifacts.
3. Apply brand and historical-price filters before ranking.
4. Produce a documented score breakdown and stable tie-break (`product_id`).

### 3. Evidence-backed comparison — critical path

1. Select bounded, relevant excerpts per candidate with product/comment IDs.
2. Produce strengths and weaknesses only from tagged review evidence; keep counts as aggregates.
3. Enforce unsupported-attribute and medical/current-price guardrails in the presentation layer.

### 4. Local demo — critical path

1. Add a minimal local CLI or single-process web UI chosen for the repository’s installed
   dependencies; avoid accounts, services, and deployment work.
2. Support one Persian query, filters, ranked results, and a two-product comparison.
3. Add a scripted presenter path using known evaluation cases.

### 5. Evaluation

1. Add a fixed, versioned Persian JSONL set with queries, filter inputs, expected product IDs or
   evidence constraints, and safety cases.
2. Run deterministic retrieval and comparison assertions offline.
3. Record aggregate pass/fail results and failures by case ID.

### 6. Documentation and final verification

1. Document local build, demo, data lineage, known limitations, and historical-price wording.
2. Run unit, integration, determinism, safety, and evaluation checks.
3. Capture a short presentation checklist and the final manifest counts.

## Test strategy

Use small synthetic fixtures for parser, filters, ranking, traceability, and safety behavior;
use one scoped integration fixture/manifest gate for the real evidence counts when source data is
available. Tests must not require a GPU, internet, or live prices.

The builder test will spy on the raw-comment iterator to prove a single scan. Determinism tests
will vary chunk size and input ordering where valid, then compare canonical records and evaluation
results. Snapshot tests will validate Persian labels and prohibit unsafe wording.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Persian lexical misses | Start with title + review token normalization; add embeddings only after evaluation exposes a measured gap. |
| Duplicate comment IDs/conflicts | Deterministic canonical rule plus audit artifact and fixture cases. |
| Unsupported product claims | Template only title/evidence-supported facts; default to unknown. |
| Deadline pressure | Ship through local demo on lexical baseline; drop optional embeddings/GPU work. |
| Large raw CSV | One exact sequential scan; retain only scoped candidates. |

## Optional follow-ons (not a prerequisite)

### 7. Hybrid Semantic RAG — completed artifact and retrieval checkpoint

1. Build a separate semantic artifact from published `sunscreen_products.parquet` and
   `sunscreen_comments_canonical.parquet` only; never rescan raw CSV.
2. Use CPU-only `intfloat/multilingual-e5-small` with `passage: ` corpus prefixes and persist
   normalized float32 vectors plus explicit vector-to-comment provenance.
3. Validate artifact checksums, row alignment, dimension, dtype, and norms before atomically
   publishing `_SUCCESS`.
4. Load only a successful artifact after checksum, manifest-property, mmap shape/dtype/norm, and
   metadata-alignment validation. Embed `query: ` on CPU with a process-cached model.
5. Retrieve exact normalized-vector dot-product comment hits, use each product's best bounded
   hits to avoid review-volume bias, and preserve comment/source provenance.
6. Fuse unchanged lexical and semantic product ranks with Reciprocal Rank Fusion
   `sum(1 / (60 + rank))`; stable ties use product ID and source provenance. Apply filters before
   channel ranking. Any semantic failure explicitly uses lexical fallback.
7. Extend offline cases with low-overlap Persian paraphrases and record three-channel
   comparison, evidence, latency, and deterministic repetitions. Groq sees bounded excerpts only.

## Optional follow-ons (not a prerequisite)

- Offline semantic artifact and later hybrid ranker, measured against the fixed evaluation set
- Kaggle GPU preparation notebook/script for experiments only
- Additional evidence facets if they preserve complete ID traceability
