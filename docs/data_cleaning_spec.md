# Phase 1 Data-Cleaning Specification

## 1. Purpose and scope

This document specifies a future, reproducible cleaning pipeline for the Digikala product,
seller-offer, and comment data. It defines behavior; it does not authorize generating cleaned
datasets yet.

The source files are:

- `data/raw/digikala-products.csv`: 1,283,496 rows and 12 columns
- `data/raw/digikala-comments.csv`: 6,156,289 rows and 15 columns

Both files are UTF-8 with a BOM and comma-delimited. They are immutable inputs. The cleaning
pipeline must open them read-only, never rewrite them, and record their paths, byte sizes, and
SHA-256 checksums in every run manifest.

The full EDA found important row semantics:

- `products.id` is a product grouping key, not a unique raw-row key. The file appears to mix
  stable product facts with seller offers.
- Products contain 323,129 repeated full rows, 335,144 duplicate-ID excess rows, 258,718
  duplicated unique IDs, 4,036 IDs with multiple sellers, 8,254 IDs with different prices, and
  3,989 IDs with conflicting core attributes.
- Comments contain 1,318 repeated full rows, 3,229 duplicate-ID excess rows, and 3,226 duplicated
  unique IDs.
- The full raw join found no comments whose `product_id` was absent from raw `products.id`.
  Nevertheless, future runs must enforce and audit this relationship.

## 2. Non-negotiable principles

1. Never modify either raw CSV.
2. Preserve the raw value in a quarantine or conflict record whenever interpretation is uncertain.
3. Count every exclusion, replacement, canonical selection, and other lossy transformation in a
   machine-readable audit.
4. Do not silently resolve conflicting product facts or conflicting duplicate comments.
5. Use bounded-memory, chunked processing. The implementation must be suitable for at least
   1.28 million product rows and 6.15 million comment rows without loading a complete source file.
6. Produce byte-for-byte deterministic tabular content for identical inputs, configuration, and
   dependency versions. Parquet container metadata may differ only where the selected writer
   cannot suppress nondeterministic metadata; this must be tested and documented.
7. Use `utf-8-sig` while reading both sources. Keep Persian text intact. Do not repair text after
   decoding and do not apply aggressive Arabic/Persian character normalization in Phase 1.

### 2.1 Data-state vocabulary

The implementation and audit must distinguish these states:

- **Invalid data:** violates an approved type or domain constraint, such as a negative price, an
  unrecognized boolean, or a required identifier that cannot be represented as `int64`. Apply the
  rule-specific nulling or quarantine action and count it.
- **Missing or unavailable data:** no usable value is available, but the source is not necessarily
  erroneous. Examples are `Rate=0, Rate_cnt=0`, `min_price_last_month=0`, and optional blank text.
- **Review-only outlier:** a valid value merits inspection but remains in the clean output. A price
  above the global positive-price 99.9th percentile is the initial example.
- **Unresolved business semantics:** the value may be technically valid, but its meaning is not
  authoritative. Source confirmation of price currency, offer recency, and price-history capture
  are examples. Do not turn these uncertainties into invalid-data rules.

### 2.2 Temporal and currency policy

- Treat the dataset as a historical snapshot approximately two years old. The exact snapshot date
  and product price capture date are unresolved metadata.
- Never describe an offer or price as current, latest, live, or representative of today's market.
  Do not infer the product price capture date solely from comment dates.
- Preserve parsed `Price` as `price_raw`. Human plausibility review strongly supports IRR/rial as
  the raw unit, although source metadata has not authoritatively confirmed it.
- Record `currency_status="IRR_inferred"` in the run manifest and audit.
- Emit `price_toman = price_raw / 10` for display and analysis while retaining `price_raw`
  unchanged. Never present `price_toman` as a current market price.
- Do not apply inflation adjustment in the core cleaning pipeline. Any future inflation-adjusted
  price must be a separate, explicitly estimated feature and must never be presented as a real
  current market price.
- `comments.created_at` is a date-only Jalali/Solar Hijri field in the validated source format
  `D <Persian Jalali month name> YYYY`. All 6,156,289 observed values matched and were valid.
- Preserve the exact text as `created_at_raw`, emit canonical numeric Jalali `YYYY-MM-DD` as
  `created_at_jalali`, and convert explicitly to date-only `created_at_gregorian`.
- Use a tested Jalali conversion implementation with known anchors and round-trip tests. Do not use
  ambiguous pandas automatic date parsing.
- Assign no timezone because the source has no time-of-day or timezone. The observed comment range
  is `1395-04-23` / `2016-07-13` through `1402-07-26` / `2023-10-18`.
- Never infer the dataset snapshot date or product/offer capture date from the maximum comment date.

### 2.3 Category-quality policy

Observed `sub_category` assignments include semantically surprising products. Preserve
`Category1`, `Category2`, and `sub_category` unchanged except for the general blank-to-null rule.
Do not silently reclassify products and do not treat `sub_category` alone as authoritative semantic
ground truth. Any future category correction must be a separate, documented, versioned enrichment
step that retains the source category values.

## 3. Planned outputs

Cleaned tables must use Parquet through a pinned `pyarrow` dependency:

- `data/processed/products_clean.parquet`
- `data/processed/offers_clean.parquet`
- `data/processed/comments_clean.parquet`

Audit artifacts:

- `reports/cleaning/cleaning_audit.json`: counts, invariants, configuration, and provenance
- `reports/cleaning/run_manifest.json`: source checksums, code version, dependency versions,
  timestamps, chunk size, and rule-set version
- `data/interim/quarantine/product_conflicts.parquet`
- `data/interim/quarantine/product_invalid_ratings.parquet`
- `data/interim/quarantine/offer_invalid_prices.parquet`
- `data/interim/quarantine/comment_conflicts.parquet`
- `data/interim/quarantine/comment_invalid_ratings.parquet`
- `data/interim/quarantine/comment_orphans.parquet`

Quarantine records must include `source_file`, one-based `source_row_number`, the original raw
values required to reproduce the decision, applicable rule IDs, and a reason. Conflict outputs
must also identify the selected canonical row. Tabular audit and quarantine outputs must also use
Parquet; JSON remains the format for the audit summary and run manifest.

## 4. Intended schemas

Source identifiers are parsed and stored as signed `int64`. Required identifiers that are missing,
non-numeric, fractional, or outside the signed 64-bit range are invalid and their rows must be
quarantined. Synthetic hashes such as `offer_id` remain strings. Numeric parsing must be strict and
locale-independent.

### 4.1 `products_clean`

One canonical row per `product_id`.

| Column | Logical type | Nullable | Meaning |
|---|---|---:|---|
| `product_id` | int64 | no | Strictly parsed raw `products.id`; canonical product key |
| `title_fa` | string | yes | Persian product title; whitespace-only becomes null |
| `category1` | string | yes | Raw `Category1` |
| `category2` | string | yes | Raw `Category2` |
| `brand` | string | yes | Raw `Brand` |
| `rate` | float64 | yes | Product rating on the 0–100 scale; unrated becomes null |
| `rate_count` | int64 | yes | Parsed raw `Rate_cnt`; must be non-negative |
| `sub_category` | string | yes | Raw `sub_category` |
| `is_unrated` | boolean | no | True only when raw `Rate=0` and `Rate_cnt=0` |
| `inconsistent_zero_rate` | boolean | no | True when raw `Rate=0` and `Rate_cnt>0` |
| `core_attribute_conflict` | boolean | no | At least one distinct core-attribute tuple exists for the ID |
| `canonical_source_row_number` | int64 | no | Source row selected by the canonical rule |

### 4.2 `offers_clean`

One row per distinct raw offer tuple after exact identical offers are deduplicated. A different
seller or price is not a duplicate and is not inherently an error.

| Column | Logical type | Nullable | Meaning |
|---|---|---:|---|
| `offer_id` | string | no | Deterministic SHA-256 of the length-prefixed raw offer tuple |
| `product_id` | int64 | no | Strictly parsed raw `products.id` |
| `seller` | string | yes | Raw `Seller`; whitespace-only becomes null |
| `price_raw` | decimal/int64 | yes | Parsed raw `Price`; operationally inferred IRR and always retained |
| `price_toman` | decimal/float64 | yes | Derived `price_raw / 10` for display and analysis |
| `is_fake` | boolean | yes | Strictly parsed raw `Is_Fake` |
| `min_price_last_month` | decimal/int64 | yes | Zero becomes null because history is unavailable |
| `missing_price_history` | boolean | no | True when raw `min_price_last_month=0` |
| `invalid_price` | boolean | no | True for a raw zero price retained as null |
| `high_price_review` | boolean | no | `price_raw` exceeds the global valid-positive 99.9th percentile |
| `source_row_number` | int64 | no | First raw row representing the exact offer tuple |

No timestamp identifies the current offer. `offers_clean` must not label any price as current,
latest, expired, or preferred. Version 1 does not emit an inflation-adjusted price.

### 4.3 `comments_clean`

One canonical row per `comment_id` after exact full-row deduplication and conflict handling.

| Column | Logical type | Nullable | Meaning |
|---|---|---:|---|
| `comment_id` | int64 | no | Strictly parsed raw `comments.id` |
| `product_id` | int64 | no | Foreign key to `products_clean.product_id` |
| `title` | string | yes | Optional comment title |
| `body` | string | yes | Primary review text; retained even when title is missing |
| `created_at_raw` | string | no | Exact source Jalali date text |
| `created_at_jalali` | string | no | Canonical numeric Jalali date `YYYY-MM-DD` |
| `created_at_gregorian` | date32/date | no | Converted Gregorian date-only value |
| `rate` | float64 | yes | Valid recorded rating in the inclusive 1–5 range |
| `is_unrated` | boolean | no | True when raw rate is exactly zero |
| `invalid_rate` | boolean | no | True when a non-null raw rate is outside 1–5 and not zero |
| `recommendation_status` | string | yes | Preserved independently from `rate` |
| `is_buyer` | boolean | yes | Strictly parsed buyer indicator |
| `advantages` | string | yes | Raw text/list representation; no semantic parsing yet |
| `disadvantages` | string | yes | Raw text/list representation; no semantic parsing yet |
| `likes` | int64 | yes | Strictly parsed, non-negative count |
| `dislikes` | int64 | yes | Strictly parsed, non-negative count |
| `seller_title` | string | yes | Optional seller title |
| `seller_code` | int64 | yes | Strictly parsed optional seller identifier |
| `true_to_size_rate` | string | yes | Optional size-fit category |
| `comment_id_conflict` | boolean | no | True when distinct rows shared this comment ID |
| `canonical_source_row_number` | int64 | no | Source row selected by the canonical rule |

Missing `title`, `advantages`, `disadvantages`, `recommendation_status`, seller fields, or
`true_to_size_rate` is not grounds for row removal. A missing body must be retained and audited;
Phase 1 does not invent text or substitute the title for the body. Preserve comments with missing
body in `comments_clean`. A later embedding stage may exclude a comment from embedding input only
when all usable text components (`body`, `title`, `advantages`, and `disadvantages`) are null or
blank; that is not a cleaning-table deletion rule.

## 5. Source-to-target column mapping

| Source | Raw column | Target table | Target column | Transformation |
|---|---|---|---|---|
| products | `id` | products, offers | `product_id` | Strict signed-`int64` parse; quarantine invalid required ID |
| products | `title_fa` | products | `title_fa` | Blank to null; preserve Persian text |
| products | `Category1` | products | `category1` | Blank to null; otherwise preserve without reclassification |
| products | `Category2` | products | `category2` | Blank to null; otherwise preserve without reclassification |
| products | `Brand` | products | `brand` | Blank to null |
| products | `Rate` | products | `rate` | Strict numeric parse and rating rules |
| products | `Rate_cnt` | products | `rate_count` | Strict non-negative integer parse |
| products | `sub_category` | products | `sub_category` | Blank to null; preserve as non-authoritative source label |
| products | `Seller` | offers | `seller` | Blank to null |
| products | `Price` | offers | `price_raw` | Strict numeric parse; retain original numeric value |
| products | `Price` | offers | `price_toman` | Derive `price_raw / 10`; status is operational IRR inference |
| products | `Is_Fake` | offers | `is_fake` | Accept only `True`/`False`; quarantine unrecognized values |
| products | `min_price_last_month` | offers | `min_price_last_month` | Strict numeric parse; zero to null as unavailable history |
| comments | `id` | comments | `comment_id` | Strict signed-`int64` parse; quarantine invalid required ID |
| comments | `product_id` | comments | `product_id` | Strict signed-`int64` parse; quarantine invalid required ID |
| comments | `title` | comments | `title` | Blank to null |
| comments | `body` | comments | `body` | Blank to null; primary text field |
| comments | `created_at` | comments | `created_at_raw` | Preserve the exact source value |
| comments | `created_at` | comments | `created_at_jalali` | Explicit month mapping to Jalali `YYYY-MM-DD` |
| comments | `created_at` | comments | `created_at_gregorian` | Explicit tested Jalali conversion to date-only ISO/Parquet date |
| comments | `rate` | comments | `rate` | Strict numeric parse; zero/out-of-range rules |
| comments | `recommendation_status` | comments | `recommendation_status` | Blank to null; no inference from rate |
| comments | `is_buyer` | comments | `is_buyer` | Accept only `True`/`False`; quarantine unrecognized values |
| comments | `advantages` | comments | `advantages` | Blank to null; otherwise preserve |
| comments | `disadvantages` | comments | `disadvantages` | Blank to null; otherwise preserve |
| comments | `likes` | comments | `likes` | Strict non-negative integer parse |
| comments | `dislikes` | comments | `dislikes` | Strict non-negative integer parse |
| comments | `seller_title` | comments | `seller_title` | Blank to null |
| comments | `seller_code` | comments | `seller_code` | Nullable signed-`int64` parse; flag unrecognized values |
| comments | `true_to_size_rate` | comments | `true_to_size_rate` | Blank to null; preserve category |

## 6. Deterministic canonical-selection rules

### 6.1 Products

After removing exact full-row duplicates, group rows by `product_id`. Compare the stable core tuple:

`(title_fa, Category1, Category2, Brand, Rate, Rate_cnt, sub_category)`.

If all tuples agree, emit one product without a conflict. If they differ, set
`core_attribute_conflict=true`, write every distinct candidate to `product_conflicts`, and select
one canonical product deterministically. The highest-`Rate_cnt` rule is approved as the primary
selection policy:

1. Exclude candidates quarantined for an out-of-range product rating or invalid `Rate_cnt`.
2. Choose the greatest parsed `Rate_cnt`; null sorts below every valid count.
3. On a tie, choose the candidate with the greatest number of non-null core attributes.
4. On a tie, choose the lexicographically smallest SHA-256 digest of the length-prefixed core tuple.
5. On the practically impossible digest tie, choose the lowest one-based source row number.

The conflict audit must include all candidates, their raw values, selection rank components, and
`selected_as_canonical`. Selecting a canonical row does not assert that its conflicting facts are
true. If no candidate survives validation, omit that product from `products_clean`, quarantine all
candidates, and count the omission.

### 6.2 Comments

Remove exact full-row duplicates, then group by `comment_id`. If distinct rows share an ID, mark a
conflict, write all alternatives to `comment_conflicts`, and choose the canonical row as follows:

1. Count non-null values across all mapped source fields except `comment_id`. Use blank-to-null
   values for this count but otherwise do not normalize the text.
2. Choose the row with the greatest completeness count.
3. On a tie, choose the lexicographically smallest SHA-256 digest of the length-prefixed mapped
   raw values in source-column order.
4. On the practically impossible digest tie, choose the lowest one-based source row number.

The audit records all alternatives, completeness scores, digests, and the selected row. A canonical
selection is lossy in the clean table even though every alternative remains recoverable from the
audit and immutable source.

## 7. Rule catalog

“Lossy” means the clean table drops a row/value or changes its information content. Audit or
quarantine preservation does not make the clean-table action lossless.

| Rule ID | Input condition | Action | Output flag/audit | Lossy? |
|---|---|---|---|---:|
| RAW-001 | Any source row | Read source only; record checksum and row number | Run manifest | No |
| ID-001 | Required ID is a whole number within signed `int64` range | Parse and retain as `int64` | Valid-ID count | No |
| ID-002 | Required ID is missing, non-numeric, fractional, or outside `int64` | Quarantine row | Invalid-required-ID count | Yes |
| ID-003 | Optional `seller_code` cannot be parsed as `int64` | Set null and flag; preserve raw value in audit | Invalid-optional-ID count | Yes |
| BOOL-001 | Boolean source value is exactly observed `True` or `False` | Parse to boolean | Boolean value counts | No |
| BOOL-002 | Required boolean has any other representation | Quarantine affected row and preserve raw value | Unrecognized-boolean count | Yes |
| TXT-001 | Text is empty or whitespace-only | Convert to null | `blank_text_to_null_count` per column | Yes |
| TXT-002 | Nonblank Persian text | Preserve decoded UTF-8 text; no aggressive normalization | Encoding validation | No |
| PRD-001 | Entire product raw row repeats | Keep first source occurrence; drop later occurrences | `products_exact_full_row_duplicates_removed` | Yes |
| PRD-002 | Multiple rows share `product_id` | Treat as a group, not duplicate rows | ID multiplicity counts | No |
| PRD-003 | Core tuples conflict within product ID | Preserve all in conflict audit; apply canonical rule | `core_attribute_conflict` | Yes |
| PRD-004 | `Rate` is outside 0–100 or non-numeric | Quarantine candidate; do not emit its rating | Product-rating quarantine count | Yes |
| PRD-005 | `Rate=0` and `Rate_cnt=0` | Set clean rate to null | `is_unrated=true` | Yes |
| PRD-006 | `Rate=0` and `Rate_cnt>0` | Preserve zero; flag inconsistency; do not infer meaning | `inconsistent_zero_rate=true` | No |
| PRD-007 | `Rate_cnt` is negative, fractional, or non-numeric | Quarantine candidate | Invalid rating-count audit | Yes |
| PRD-008 | Source category appears semantically surprising | Preserve unchanged; do not reclassify or treat as authoritative truth | Category-quality note/count | No |
| OFF-001 | Complete raw offer tuple repeats exactly | Keep first source occurrence only | `offers_exact_duplicates_removed` | Yes |
| OFF-002 | Seller or price differs for same product | Preserve distinct offer | Offer multiplicity counts | No |
| OFF-003 | `Price=0` | Retain offer, set `price_raw` to null | `invalid_price=true` | Yes |
| OFF-004 | Price is negative or non-numeric | Exclude offer from clean table; quarantine raw row | Invalid-price quarantine count | Yes |
| OFF-005 | Valid positive `price_raw` exceeds the global 99.9th percentile of valid positive prices | Retain unchanged; flag for review | `high_price_review=true`; threshold in audit/manifest | No |
| OFF-006 | Any valid price | Preserve as `price_raw`; derive `price_toman=price_raw/10`; never label current/latest | `currency_status="IRR_inferred"` | No |
| OFF-007 | `min_price_last_month=0` | Set history value to null; do not mark current price invalid | `missing_price_history=true` | Yes |
| COM-001 | Entire comment raw row repeats | Keep first source occurrence; drop later occurrences | `comments_exact_full_row_duplicates_removed` | Yes |
| COM-002 | Distinct rows share `comment_id` | Preserve all in conflict audit; apply canonical rule | `comment_id_conflict=true` | Yes |
| COM-003 | Optional comment fields are missing | Retain comment | Missing counts per column | No |
| COM-004 | `rate` is within 1–5 inclusive, including fractional values | Preserve numeric value | Valid-rate count | No |
| COM-005 | `rate=0` | Set clean rate to null; do not infer negative sentiment | `is_unrated=true`, zero-rate count | Yes |
| COM-006 | Non-null rate is outside 1–5 and is not zero | Set rate to null; preserve raw row/value in audit | `invalid_rate=true` | Yes |
| COM-007 | Rate is `2500` | Apply COM-006 explicitly | Invalid-rate audit includes raw `2500` | Yes |
| COM-008 | Recommendation is present or missing | Preserve independently; never derive from rate | Recommendation distribution audit | No |
| COM-009 | Body is missing | Retain comment; defer embedding eligibility to downstream text assembly | Missing-body count | No |
| DATE-001 | `created_at` matches `D <recognized Persian Jalali month> YYYY` | Preserve exact raw value and construct canonical Jalali `YYYY-MM-DD` | Date-format and range audit | No |
| DATE-002 | Valid canonical Jalali date | Convert with an explicit tested Jalali algorithm to Gregorian date-only value | Conversion implementation/version in manifest | No |
| DATE-003 | Date shape, month, or Jalali day is invalid in a future source | Quarantine row and preserve raw value; never use ambiguous automatic parsing | Invalid-date audit | Yes |
| DATE-004 | Any source comment date | Assign no timezone and infer no product/offer capture date | Temporal-policy assertion | No |
| JOIN-001 | Comment `product_id` exists in canonical products | Retain comment | Matched-comment count | No |
| JOIN-002 | Comment `product_id` is null or absent from canonical products | Exclude from clean comments; quarantine and count | Orphan/missing-FK audit | Yes |

## 8. Validation requirements

### 8.1 `products_clean`

- `product_id` is non-null and unique.
- Every required identifier is a signed `int64`; invalid required-ID rows are quarantined.
- There is at most one canonical product per raw product ID.
- `rate` is null or within 0–100.
- `rate_count` is null or a non-negative integer.
- `is_unrated=true` implies `rate is null` and raw `Rate=0, Rate_cnt=0` in provenance.
- `inconsistent_zero_rate=true` implies clean `rate=0` and `rate_count>0`.
- Every conflicting product ID has complete candidate coverage in `product_conflicts`.
- Source category values are preserved without silent reclassification; `sub_category` is not
  asserted to be authoritative semantic ground truth.
- Output row count reconciles with distinct IDs after invalid-only groups and missing IDs are
  quarantined.

### 8.2 `offers_clean`

- Every `offer_id` is non-null and unique.
- Every `product_id` matches a `products_clean.product_id`; offers for omitted products are
  quarantined and counted rather than silently dropped.
- Exact offer tuples are unique; distinct seller/price combinations remain.
- `price_raw` is null or positive. Zero maps to null with `invalid_price=true`.
- For non-null valid prices, `price_toman = price_raw / 10` exactly; `price_raw` remains unchanged.
- Negative and non-numeric raw prices appear only in quarantine.
- `high_price_review=true` exactly when a valid positive `price_raw` is greater than the global
  99.9th percentile threshold; it never causes removal. The threshold is recorded in both audit
  and manifest.
- Raw `min_price_last_month=0` maps to null with `missing_price_history=true` and does not set
  `invalid_price`.
- `is_fake` contains only parsed source `True`/`False`; unrecognized values are quarantined.
- No column or documentation claims that an offer is current or that source metadata authoritatively
  confirms the currency. Operational currency status is `IRR_inferred`.

### 8.3 `comments_clean`

- `comment_id` is non-null and unique.
- `product_id` is non-null and references `products_clean.product_id`.
- Required IDs are signed `int64`; invalid required-ID rows are quarantined.
- `rate` is null or within 1–5 inclusive.
- Raw zero rates map to null with `is_unrated=true` and `invalid_rate=false`.
- Other out-of-range/non-numeric rates map to null with `invalid_rate=true` and have an audit row.
- The known raw `2500` value is represented in the invalid-rate audit.
- Recommendation status is unchanged except blank-to-null and is never inferred from rate.
- All alternatives for conflicting comment IDs are present in `comment_conflicts`.
- Missing optional fields do not reduce row counts.
- Missing body does not remove a row from `comments_clean`.
- `is_buyer` contains only parsed source `True`/`False`; unrecognized values are quarantined.
- `created_at_raw` exactly preserves the source text.
- `created_at_jalali` matches `YYYY-MM-DD` and represents the same validated Jalali date.
- `created_at_gregorian` is a date-only Parquet date and round-trips through the tested conversion.
- The observed clean range is `1395-04-23` through `1402-07-26`, corresponding to `2016-07-13`
  through `2023-10-18`. All 6,156,289 observed source values matched the approved format and were
  valid in the completed diagnostic.
- No timezone is assigned, because the source contains neither time-of-day nor timezone data.
- Comment dates do not determine the dataset snapshot date or product/offer capture date.

### 8.4 Cross-output reconciliation

For each source, the audit must provide equations that reconcile raw rows to clean and quarantine
outcomes. Because categories can overlap, both exclusive disposition counts and non-exclusive rule
event counts are required. At minimum:

```text
raw rows = exact-duplicate rows removed
         + clean canonical/offer/comment rows
         + exclusively quarantined rows
         + noncanonical conflicting rows preserved in conflict audit
```

The implementation must define the exact mutually exclusive disposition ordering used by this
equation and separately report overlapping flags.

## 9. Machine-readable audit contract

`cleaning_audit.json` must be UTF-8 JSON with sorted keys and this top-level structure:

```json
{
  "spec_version": "1.0.0",
  "run_id": "deterministic hash of inputs, configuration, and code version",
  "sources": {},
  "products_clean": {
    "before_rows": 0,
    "after_rows": 0,
    "rule_event_counts": {},
    "exclusive_disposition_counts": {},
    "conflict_counts": {},
    "validation_failures": []
  },
  "offers_clean": {},
  "comments_clean": {},
  "join": {
    "matched_comments": 0,
    "missing_product_id_comments": 0,
    "orphan_comments": 0
  },
  "quarantine_outputs": {},
  "thresholds": {
    "high_price_review": {
      "definition": "global p99.9 of valid positive price_raw",
      "calculated_value": null
    }
  },
  "business_semantics": {
    "authoritative_source_currency_confirmation": "unresolved",
    "currency_status": "IRR_inferred",
    "offer_recency": "unknown",
    "dataset_age": "approximately two years old"
  },
  "unresolved_assumptions": [],
  "status": "success or failed"
}
```

Each rule in Section 7 must have a counter, including zero-valued counters. Each output entry must
include its row count and SHA-256 checksum. A run is successful only after all output files are
atomically finalized and every acceptance validation passes. Failed runs must not publish partial
files as final outputs.

`run_manifest.json` must repeat the calculated high-price threshold and nearest-rank parameters,
record the pinned `pyarrow` version, set `currency_status="IRR_inferred"`, state that authoritative
source confirmation and offer recency remain unresolved, and carry nullable fields for exact
dataset snapshot date and product price capture date. It must also record the Jalali conversion
implementation/version, its test-vector version, and the observed comment-date range.

## 10. Streaming and reproducibility design constraints

- Read CSVs with configurable chunks, defaulting to 100,000 rows or less if memory tests require.
- Use disk-backed keyed state, partitioning, or an embedded database for global deduplication,
  canonical grouping, and joins. Never retain all comments or their text in memory.
- Compute full-row and canonical hashes from explicit column order, type tags, null markers, UTF-8
  bytes, and length prefixes. Do not use Python's process-randomized `hash()`.
- Assign source row numbers before filtering. Chunk size must not change canonical choices.
- Sort final tables by their stable keys (`product_id`; then `product_id, offer_id`; `comment_id`)
  using an external/disk-backed sort if required.
- Pin and record parsing/writer versions, null conventions, decimal behavior, boolean vocabulary,
  and the rule-set version.
- Use a pinned `pyarrow` version for all Parquet outputs and record it in the manifest.
- Calculate the high-price threshold globally from all valid positive `price_raw` values before
  final offer publication. Define p99.9 by the nearest-rank method: after sorting `N` valid positive
  prices ascending, select rank `ceil(0.999 × N)` using one-based ranks. Flag values strictly
  greater than that threshold. Record `N`, rank, algorithm, and calculated threshold.
- Write to run-scoped temporary paths and atomically rename only after validation succeeds.
- On exceptions, close temporary databases/files and mark the audit failed. Never report success
  after incomplete processing.

## 11. Acceptance criteria for future implementation

1. Raw source checksums before and after a run are identical.
2. Unit tests cover exact duplicates across chunk boundaries, duplicate IDs with conflicts,
   canonical ties, blank text, every price class, every rating class, and orphan comments.
3. Integration tests use synthetic chunks small enough to force every global rule across a chunk
   boundary.
4. Tests prove canonical results and output ordering are unchanged for at least three chunk sizes.
5. Peak memory remains bounded by the configured chunk plus documented indexes/buffers; the
   implementation successfully processes datasets at the observed full sizes.
6. All three schemas and nullability constraints pass.
7. Every raw row has a reconciled exclusive disposition, and every lossy event has an audit count.
8. Product and comment conflict audits preserve every distinct alternative and identify the
   canonical row deterministically.
9. Product ratings, comment ratings, prices, and foreign keys satisfy Section 8.
10. The known EDA findings are accounted for, including the single `2500` comment rate, 223 zero
    product prices, and zero raw join orphans. Differences must fail validation unless explained by
    an approved source checksum or rule-set change.
11. No cleaned file is published if validation fails; rerunning identical inputs/configuration
    yields equivalent files and an identical deterministic `run_id`.
12. `pytest -q`, `ruff check .`, and `git diff --check` pass before release.
13. All cleaned and quarantine tables are readable by the pinned `pyarrow`; schemas match Section 4.
14. The audit and manifest contain the calculated global 99.9th-percentile price threshold,
    `currency_status="IRR_inferred"`, unresolved authoritative source confirmation, and temporal
    policy.
15. `price_toman` equals `price_raw / 10`, raw prices remain present, and no output exposes an
    inflation-adjusted price, current price, or latest offer.
16. Tests cover every Persian Jalali month, month-specific day bounds including leap Esfand,
    approved known conversion anchors, and Jalali/Gregorian round trips. Automatic pandas date
    inference is prohibited.
17. The implementation preserves category labels unchanged and treats future corrections as a
    separate versioned enrichment output.

## 12. Remaining unresolved questions

The following questions remain unresolved; all other policy choices previously listed for approval
are now approved in this specification.

1. **Authoritative source confirmation of currency:** Human plausibility review strongly supports
   IRR/rial and the operational status is `IRR_inferred`, but source metadata has not authoritatively
   confirmed the unit. Preserve `price_raw` regardless of future confirmation.
2. **Exact dataset snapshot date:** The dataset is treated as approximately two years old, but its
   authoritative snapshot date is unknown and must be supplied as metadata.
3. **Product price capture date:** The capture date for product and offer prices is unknown. It must
   not be inferred solely from comment dates.
4. **Offer temporal semantics:** It is unknown whether product rows represent concurrent offers,
   historical snapshots, or another collection process. Regardless of the answer, version 1 never
   labels them current or latest.
5. **Price-history metadata:** Although zero `min_price_last_month` is now defined as unavailable,
   the field's precise lookback window, capture method, and currency relationship to `Price` remain
   unverified.
