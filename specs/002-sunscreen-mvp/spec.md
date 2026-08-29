# Sunscreen recommendation MVP

## Purpose

Deliver a local, presentation-ready Persian sunscreen recommendation demo for exactly this
catalogue scope:

- `Category1 = مراقبت پوست`
- `Category2 = کرم ضد آفتاب`

This is a sunscreen vertical slice, not a reusable multi-category platform.

## Evidence baseline

The scoped build must reproduce these source-inventory facts before recommendation work begins:

| Measure | Expected value |
|---|---:|
| Products | 1,048 |
| Raw matching comments | 53,522 |
| Products with comments | 892 |
| Brands | 175 |
| Historical-price coverage | 99.81% (1,046 / 1,048) |
| Buyer comments | 50,275 |
| Non-empty comment bodies | 53,517 |

“Historical price” means the lowest valid positive observed `price_raw` per product. It is
historical inferred data, never a current, live, or available-for-purchase price.

## Users and primary flows

1. A Persian-speaking presenter enters a sunscreen-focused query, optional historical-price
   range, and optional brand filter.
2. The system returns a deterministic ranked shortlist with transparent score components.
3. The user selects at least two products for a side-by-side comparison.
4. The comparison presents evidence-backed strengths, weaknesses, review counts, historical
   price (clearly labelled), and short review excerpts with product and review IDs.

## Functional requirements

### F1 — Scoped, reproducible data build

The builder shall create versioned Parquet outputs for scoped products, product-level historical
prices, raw matching comments, and canonical matching comments. It shall record source paths,
row counts, build parameters, and a content fingerprint in a manifest.

### F2 — One exact raw-comment scan

For one build invocation, the raw comments CSV shall be parsed sequentially exactly once with the
existing exact-token CSV semantics where practical. Only comments whose parsed product ID belongs
to the scoped product-ID set may enter the matching-comment candidate stream. The build shall not
use SQLite.

### F3 — Bounded-memory canonical comments

Only the approximately 53k matching candidates may be retained for deterministic
deduplication/canonicalization. Canonical selection must be stable across runs and input chunk
sizes, retain provenance (`comment_id`, `product_id`, source row), and emit audit/conflict data
for duplicate or conflicting IDs. The raw 6.15m-comment corpus must not be fully materialized.

### F4 — Product filters

The demo shall support deterministic filtering by brand and historical-price range. Products
without a valid historical price shall be excluded only when a price filter is active; otherwise
they remain eligible and show historical price as unknown.

### F5 — Baseline retrieval and ranking

For a Persian query, retrieval shall use a simple deterministic lexical baseline over product
titles and canonical review text. Ranking shall expose its components, at minimum query match,
review-evidence quantity/quality, and any active filters. Embeddings are not required for the
baseline.

### F6 — Evidence-backed comparison

The system shall compare two or more selected candidates. Each displayed strength or weakness
must cite one or more bounded excerpts and their `{product_id, comment_id}`. Aggregate review
counts must be shown separately from retrieved evidence.

### F7 — Local demo and evaluation

The repository shall provide a simple local presentation flow and a fixed Persian evaluation set
covering retrieval, comparison, price/brand filtering, unknown attributes, and safety behavior.

## Safety and truthfulness requirements

- Never call a historical price “current,” “today,” “live,” or imply stock or availability.
- Do not diagnose, treat, prescribe, or make medical recommendations.
- Do not guarantee suitability for any skin type.
- State SPF, skin type, texture, finish, white cast, or other product attributes only when the
  product title or retrieved review evidence supports them; otherwise omit them or label them
  `unknown`.
- Every generated or templated claim must remain traceable to a product ID and, when evidence is
  used, one or more comment IDs.
- Excerpts must be short and bounded; the UI must not dump full reviews or claim that evidence is
  representative of all users.

## Non-goals

- Any category other than `مراقبت پوست > کرم ضد آفتاب`
- A category-plugin or general recommendation-platform architecture
- Live prices, inventory, production deployment, user accounts, or personalization
- LLM fine-tuning
- Scanning/materializing all 6.15m comments after the scoped build
- Medical recommendation, diagnosis, or treatment guidance

## Measurable acceptance criteria

1. A clean build produces the four scoped Parquet datasets and manifest, and its manifest matches
   all seven evidence-baseline values above (price coverage rounded to two decimals).
2. Instrumented integration testing proves one raw-comments scan per build and records zero
   SQLite imports, files, or connections.
3. Rebuilding with at least two chunk sizes yields byte-identical canonical-comment identifiers,
   canonical selection, and product ranking for the fixed evaluation set.
4. Build memory stays bounded by the scoped candidate set: no structure contains all raw comments;
   the test fixture asserts candidate retention is limited to matching rows.
5. Every evaluation result has only scoped product IDs; every excerpt has a matching scoped
   `{product_id, comment_id}`; and no excerpt exceeds the configured character limit.
6. For every fixed evaluation case, repeated runs return the same ordered product IDs and score
   breakdowns.
7. Brand and historical-price filters have exact fixture-tested inclusion/exclusion behavior,
   including unknown historical prices.
8. The comparison accepts at least two products and displays, for each, review counts, strengths,
   weaknesses, historical-price label/value-or-unknown, and traceable excerpts.
9. Safety tests reject or safely reframe medical/treatment language, prohibit current-price
   wording, and mark unsupported attributes as unknown or absent.
10. The documented local demo starts from the generated scoped artifacts and completes a fixed
    two-product Persian comparison without network access or a GPU.

## Constraints and decisions

- Use existing parsing and cleaning semantics where practical, especially strict ID parsing,
  boolean/rating parsing, comment-text cleaning, and exact CSV streaming.
- Keep paths and thresholds in one small, sunscreen-specific configuration surface.
- Prefer deterministic templates for results and comparisons; an LLM, if later added, may only
  summarize supplied evidence and cannot introduce uncited facts.
- Optional embedding or Kaggle-GPU preparation must be isolated behind the completed lexical
  baseline. The deliverable demo must remain local and CPU-capable.
