# Products v1 Cleaning Run: Engineering Report

## Executive summary

The Phase 1 products cleaning run completed successfully against all 1,283,496 raw product-file
records under cleaning specification 1.0.3. It published 948,352 canonical products and 957,753
distinct seller offers. The run preserved the raw source, removed exact duplicates, retained
distinct offers, exposed product-fact conflicts, applied deterministic row-level rules, and wrote
auditable Parquet outputs.

This report is an educational record of what the run did and, equally importantly, what its outputs
do and do not mean. The output is a reproducible historical dataset, not a statement about current
Digikala inventory or prices.

## 1. Purpose

The raw products CSV mixes two kinds of information:

- relatively stable product facts, such as title, brand, categories, rating, and rating count;
- seller-offer facts, such as seller, price, fake-status indicator, and prior-price field.

Raw `products.id` therefore acts as a product grouping key rather than a unique raw-row key. The
run separated those concepts into:

- `products_clean.parquet`: exactly one deterministically selected canonical row per valid
  `product_id`;
- `offers_clean.parquet`: distinct offer tuples, preserving different sellers and prices;
- conflict, quarantine, and row-audit outputs that retain evidence about nontrivial decisions.

This separation provides a stable product dimension for the future comments foreign-key check and
a distinct offer table for later price and seller analysis.

## 2. Parser failure and ingest hardening

An earlier full-run attempt terminated with SIGSEGV in pandas' native C-parser path while its
low-memory parsing machinery was concatenating internal chunks. A native segmentation fault is not
a normal Python exception, so the abandoned run could not publish a completed output.

The ingest layer was subsequently changed to the Python standard-library CSV reader. The hardened
reader streams logical CSV records, preserves exact field strings, handles the UTF-8 BOM and quoted
CSV content, validates the exact source header, and batches records without invoking pandas'
`read_csv` parser. This removed the native parsing path implicated by the crash without changing
deduplication, transformation, canonicalization, price, audit, or Parquet semantics.

The successful run log shows all 13 chunks completing, including progress beyond the earlier
failure boundary, followed by canonical-product selection, global price-threshold calculation,
Parquet materialization, source-immutability verification, and final publication.

## 3. Execution and completion evidence

| Property | Recorded value |
|---|---|
| Specification | 1.0.3 |
| Run ID | `34bc1b4b7c529af6f3f50cbb9a62fbedb192a7645fc0728494661403bd061774` |
| Input | `data/raw/digikala-products.csv` |
| Output | `data/processed/products_v1` |
| Chunk size | 100,000 records |
| Row limit | None; complete source |
| Wall-clock runtime | 23 minutes 57.18 seconds |
| User CPU time | 332.57 seconds |
| System CPU time | 74.94 seconds |
| Maximum resident memory | 388,776 KiB, approximately 379.7 MiB |
| Swap | 0 |
| Exit status | 0 |
| Audit status | `success` |
| Manifest status | `success` |

The execution enabled unbuffered output and Python faulthandler. The final log line records that the
pipeline published `data/processed/products_v1`; publication occurs only after all phases and the
source-immutability check complete.

The raw source SHA-256 was identical before and after processing:

```text
0f646e1b0ba5a07510a168d98ae397ae25a947b1423ccdaefc029170b7831acd
```

Recorded runtime versions were Python 3.13.13, pandas 2.3.3, PyArrow 21.0.0, and SQLite 3.53.1.
Pandas remains part of the recorded environment, but the hardened raw products ingest does not use
its CSV parser.

## 4. Raw-to-clean reconciliation

Every documented milestone-2 reconciliation check passed.

| Reconciliation | Substitution with run counts | Result |
|---|---|---|
| Raw-row identity | `1,283,496 = 323,129 + 960,367` | Pass |
| Product candidate disposition | `960,367 = 960,367 + 0` | Pass |
| Offer candidate disposition | `960,367 = 960,367 + 0` | Pass |
| Exact offer deduplication | `960,367 = 957,753 + 2,614` | Pass |
| Final referential-integrity disposition | `957,753 = 957,753 + 0` | Pass |

The terms are, respectively:

```text
input rows = exact raw duplicates + distinct raw rows
distinct raw rows = accepted product candidates + product quarantines
distinct raw rows = accepted offer candidates + offer-transform quarantines
accepted offer candidates = distinct offer candidates + exact offer duplicates
distinct offer candidates = final offers + OFF-009 missing-canonical-product quarantines
```

Canonicalization is a separate grouping step: 960,367 accepted product candidates produced
948,352 unique canonical products. Noncanonical candidates are not unexplained losses; they belong
to product groups represented by one canonical row, with conflicting core alternatives preserved
in the conflict output.

## 5. Products, offers, duplicates, and conflicts

| Result | Count | Interpretation |
|---|---:|---|
| Raw input records | 1,283,496 | Complete products CSV |
| Exact full-row duplicates removed | 323,129 | Later byte-equivalent field tuples removed under PRD-001 |
| Distinct raw rows retained | 960,367 | Candidates evaluated independently as product facts and offers |
| IDs with multiple accepted product candidates | 11,785 | Product grouping, not automatically an error |
| Canonical products | 948,352 | One row per valid `product_id` |
| Product IDs with core conflicts | 3,989 | At least two distinct stable-core tuples |
| Conflict alternative rows | 8,022 | All distinct conflicting alternatives retained for inspection |
| Accepted offer candidates | 960,367 | Offer-side row transformation succeeded |
| Exact offer duplicates removed | 2,614 | Identical offer tuples collapsed under OFF-001 |
| Distinct final offers | 957,753 | Different sellers or prices remain distinct |
| Offer multiplicity events | 9,401 | Additional distinct offers associated with product IDs |
| Offers without a canonical product, OFF-009 | 0 | Referential integrity required no offer quarantine |

Canonical products are sorted by `product_id`. Offers are sorted by `(product_id, offer_id)`. The
technical `offer_id` is a deterministic SHA-256 fingerprint of the exact raw offer tuple; it is not
a source-system or business identifier.

## 6. Data-quality transformations

### Product ratings

`products.Rate` uses a 0–100 scale.

| Rule/result | Count | Clean behavior |
|---|---:|---|
| `Rate=0` and `Rate_cnt=0` | 590,034 | Rating becomes null and `is_unrated=true` |
| `Rate=0` and `Rate_cnt>0` | 15 | Zero is preserved and `inconsistent_zero_rate=true` |
| Invalid/out-of-range `Rate` | 0 | No product-rating quarantine required |
| Invalid `Rate_cnt` | 0 | No rating-count quarantine required |

The 15 inconsistent zero ratings remain visible because silently deciding their meaning would be
less reliable than preserving and flagging them.

### Prices

| Rule/result | Count | Clean behavior |
|---|---:|---|
| Valid positive `price_raw` | 957,555 | Preserved; provisional `price_toman=price_raw/10` derived |
| Zero `Price` | 198 | Converted to null with `invalid_price=true` |
| Negative/non-numeric/fractional/overflow Price | 0 | No OFF-004 quarantine required |
| High-price review flags | 954 | Retained unchanged for review, never removed |

Zero prices are invalid values but do not invalidate the entire offer; the clean offer remains with
a null price and an explicit flag. High prices are valid review-only outliers and are not evidence
of corruption by themselves.

### Categories and price history

| Rule/result | Count | Clean behavior |
|---|---:|---|
| Blank `Category2` | 181,890 | Converted to null under the general blank-text rule |
| `min_price_last_month=0` | 900,650 | Converted to null with `missing_price_history=true` |
| Invalid nonzero price-history values | 0 | No OFF-008 quarantine required |

A blank category is missing data rather than an invalid product. Similarly, zero price history is
defined as unavailable history and is not treated as an invalid current price.

### Row-level audit and quarantine

`row_audit.parquet` contains 213 traceability records: the 198 zero-price transformations and 15
inconsistent zero-rating observations. Frequent expected transformations such as unrated products,
blank optional categories, and unavailable price history are counted in aggregate rather than
expanded into hundreds of thousands of row-level audit objects.

Both product and offer quarantine counts are zero. This means no candidate violated the rules that
require whole-row exclusion; it does not mean the data is complete, conflict-free, current, or
semantically perfect. The run still found:

- 323,129 exact duplicate raw rows;
- 3,989 product IDs with conflicting core facts;
- 590,034 unrated products;
- 15 inconsistent zero ratings;
- 198 invalid zero prices;
- 181,890 missing `Category2` values;
- 900,650 unavailable price-history values;
- 954 high-price review outliers.

Quarantine is therefore one narrow disposition category, not a global quality score.

## 7. Global p99.9 price review

The threshold population is the 957,555 valid positive `price_raw` values in final accepted
distinct offers after exact offer deduplication and removal of any OFF-009 offers. In this run,
OFF-009 was zero.

The specification's nearest-rank method is:

1. Sort the `N` population values in ascending order.
2. Calculate one-based rank `ceil(0.999 × N)`.
3. Select the price at that rank as the threshold.
4. Set `high_price_review=true` only when `price_raw` is strictly greater than the threshold.
   Equality is not flagged.

For this run:

| Quantity | Value |
|---|---:|
| Population `N` | 957,555 |
| One-based rank | 956,598 |
| Threshold, raw inferred IRR | 599,000,000 |
| Provisional threshold in Toman | 59,900,000 |
| Offers strictly above threshold | 954 |

The Toman value is derived as `price_raw / 10`. The currency status remains `IRR_inferred`, not
authoritatively confirmed, and neither representation should be presented as a current market
price.

## 8. Published artifacts

### Clean-table schemas

`products_clean.parquet` contains:

```text
product_id:int64, title_fa:string?, category1:string?, category2:string?,
brand:string?, rate:float64?, rate_count:int64?, sub_category:string?,
is_unrated:boolean, inconsistent_zero_rate:boolean,
core_attribute_conflict:boolean, canonical_source_row_number:int64
```

`offers_clean.parquet` contains:

```text
offer_id:string, product_id:int64, seller:string?, price_raw:exact numeric?,
price_toman:exact decimal?, is_fake:boolean?, min_price_last_month:exact numeric?,
missing_price_history:boolean, invalid_price:boolean,
high_price_review:boolean, source_row_number:int64
```

Conflict output stores each distinct conflicting core tuple, its candidate and canonical source-row
numbers, completeness and digest tie-break evidence, raw core JSON, and whether it was selected.
Quarantine records follow the shared quarantine schema with source file, source row, entity ID,
rule IDs, reason, and raw JSON. Row audits retain dataset, source row, entity ID, field, raw value,
rule, action, and severity.

### Rows and checksums

| File | Rows / role | SHA-256 |
|---|---:|---|
| `products_clean.parquet` | 948,352 | `267adcf7555b5396aa20f544314de94d9109d244297689fd2d28c0e79dc77808` |
| `offers_clean.parquet` | 957,753 | `b6863da74aca6d44fea42effa4a56b038ac2a0c8706aa7e3d7c3b518215d3f6d` |
| `product_conflicts.parquet` | 8,022 | `ca0efad40a8a3f35d32d22029b6ef1a12c86d87bf206d3e3e6d06fdd777a73e8` |
| `product_quarantine.parquet` | 0 | `e9557c57b6d2ac473eea8af6a66da5e165e932867148ecd083d4ced54dfa62b2` |
| `offer_quarantine.parquet` | 0 | `e9557c57b6d2ac473eea8af6a66da5e165e932867148ecd083d4ced54dfa62b2` |
| `row_audit.parquet` | 213 | `8122c4da115c34139f1af0c30cad80f508fb04cb753e213a1fe57bbc1a10e7df` |
| `cleaning_audit.json` | Audit summary | `fb5730a43bfa1ea03a1ae0fd01da219c84a6cb7b018133cf6b4d09fccd881e1d` |
| `run_manifest.json` | Run provenance; no self-checksum recorded | — |

The audit and manifest both record successful status, the source checksum, output metadata, price
threshold, specification version, and configuration. The manifest additionally records the
deterministic run ID and dependency versions.

## 9. Remaining limitations

The cleaned outputs deliberately preserve several unresolved business semantics:

1. **Currency confirmation:** IRR/rial is strongly inferred but not authoritatively confirmed by
   source metadata. `price_raw` must remain preserved even if this inference is revised.
2. **Historical prices:** this is an approximately two-year-old historical snapshot. Prices must
   never be described as current market prices, and the core pipeline performs no inflation
   adjustment.
3. **Offer recency:** there are no timestamps proving which offer was current, latest, concurrent,
   expired, or preferred. Distinct offers are preserved without making such claims.
4. **Price-history semantics:** zero `min_price_last_month` means unavailable history, but the exact
   lookback window, capture method, capture date, and currency relationship remain unverified.
5. **Category semantics:** source category values are preserved. They are not silently corrected
   and should not be treated as unquestionable semantic ground truth.

## 10. Preparation for comments and LLM stages

The canonical product table establishes the required `products_clean.product_id` key for the next
dataset-level milestone. The comments pipeline can now:

- validate each comment's `product_id` against a deterministic canonical product universe;
- quarantine and count future orphan comments instead of silently discarding them;
- attach stable product title, brand, and category context without multiplying comments through
  multiple seller offers;
- keep offer information separate when price or seller context is needed.

For later LLM work, this separation reduces ambiguity between product facts and commercial offers.
Product metadata can support retrieval filters, product context, evaluation slices, and prompt
grounding, while comments remain the primary review-text source. Conflict flags, unrated status,
invalid-price flags, and review-only price outliers provide explicit quality signals that downstream
feature engineering can use rather than rediscovering or silently guessing.

The cleaned tables are therefore a reproducible foundation, not a finished modeling dataset. The
comments cleaning, foreign-key validation, text eligibility policy, and later embedding or
retrieval construction remain separate auditable stages.

## Evidence sources

This report was prepared from:

- `data/processed/products_v1/cleaning_audit.json`
- `data/processed/products_v1/run_manifest.json`
- `/tmp/digikala-products-v1-runtime-2.txt`
- `/tmp/digikala-products-v1-run-2.log`
- `docs/data_cleaning_spec.md`, specification version 1.0.3

