"""Reproducible category-viability inventory for recommendation-MVP scoping."""

from __future__ import annotations

import argparse
import csv
import math
import resource
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from digikala_llm.cleaning import (
    _parse_comment_rate,
    clean_comment_text,
    parse_required_id,
    parse_strict_boolean,
)
from digikala_llm.comments_pipeline import COMMENT_CSV_FIELD_SIZE_LIMIT, COMMENT_SOURCE_COLUMNS
from digikala_llm.csv_stream import iter_exact_csv_batches

INVENTORY_COLUMNS = (
    "scope_level",
    "category1",
    "category2",
    "product_count",
    "comment_count",
    "products_with_comments",
    "products_without_comments",
    "product_comment_coverage_pct",
    "products_with_at_least_5_comments",
    "products_with_at_least_10_comments",
    "products_with_at_least_20_comments",
    "products_with_at_least_50_comments",
    "median_comments_per_commented_product",
    "p95_comments_per_commented_product",
    "non_empty_body_count",
    "buyer_comment_count",
    "valid_positive_rating_count",
    "unique_brand_count",
    "products_with_valid_historical_price",
    "historical_price_coverage_pct",
    "historical_price_p05",
    "historical_price_p50",
    "historical_price_p95",
)

SUNSCREEN_KEY = ("category1_category2", "مراقبت پوست", "کرم ضد آفتاب")
SUNSCREEN_EXPECTED = {
    "product_count": 1048,
    "comment_count": 53522,
    "products_with_comments": 892,
    "unique_brand_count": 175,
    "products_with_valid_historical_price": 1046,
    "products_with_at_least_5_comments": 685,
    "products_with_at_least_10_comments": 584,
    "products_with_at_least_20_comments": 459,
    "products_with_at_least_50_comments": 316,
}

# These broad source-category families are not useful comparison scopes for this MVP.
COHERENCE_EXCLUDED_CATEGORY1_TERMS = (
    "کتاب",
    "دفتر",
    "نوشت افزار",
    "اسباب بازی",
    "لباس",
    "اکسسوری",
    "زیورآلات",
    "نوزاد",
    "دخترانه",
    "پسرانه",
)


@dataclass
class ScopeStats:
    scope_level: str
    category1: str
    category2: str | None
    product_count: int = 0
    brands: set[str] = field(default_factory=set)
    prices: list[int] = field(default_factory=list)
    comment_count: int = 0
    non_empty_body_count: int = 0
    buyer_comment_count: int = 0
    valid_positive_rating_count: int = 0
    commented_counts: list[int] = field(default_factory=list)


def _quantile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    values.sort()
    position = (len(values) - 1) * percentile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(values[low])
    return values[low] + (values[high] - values[low]) * (position - low)


def _scope_keys(category1: str | None, category2: str | None) -> tuple[tuple[str, str, str | None], ...]:
    keys = []
    if category1 is not None:
        keys.append(("category1", category1, None))
    if category1 is not None and category2 is not None:
        keys.append(("category1_category2", category1, category2))
    return tuple(keys)


def _scope_stat(
    stats: dict[tuple[str, str, str | None], ScopeStats], key: tuple[str, str, str | None]
) -> ScopeStats:
    if key not in stats:
        stats[key] = ScopeStats(*key)
    return stats[key]


def _row(stat: ScopeStats) -> dict[str, Any]:
    commented = stat.commented_counts
    price_p05 = _quantile(stat.prices, 0.05)
    price_p50 = _quantile(stat.prices, 0.50)
    price_p95 = _quantile(stat.prices, 0.95)
    products_with_comments = len(commented)
    return {
        "scope_level": stat.scope_level,
        "category1": stat.category1,
        "category2": stat.category2,
        "product_count": stat.product_count,
        "comment_count": stat.comment_count,
        "products_with_comments": products_with_comments,
        "products_without_comments": stat.product_count - products_with_comments,
        "product_comment_coverage_pct": 100 * products_with_comments / stat.product_count,
        "products_with_at_least_5_comments": sum(value >= 5 for value in commented),
        "products_with_at_least_10_comments": sum(value >= 10 for value in commented),
        "products_with_at_least_20_comments": sum(value >= 20 for value in commented),
        "products_with_at_least_50_comments": sum(value >= 50 for value in commented),
        "median_comments_per_commented_product": _quantile(commented, 0.50),
        "p95_comments_per_commented_product": _quantile(commented, 0.95),
        "non_empty_body_count": stat.non_empty_body_count,
        "buyer_comment_count": stat.buyer_comment_count,
        "valid_positive_rating_count": stat.valid_positive_rating_count,
        "unique_brand_count": len(stat.brands),
        "products_with_valid_historical_price": len(stat.prices),
        "historical_price_coverage_pct": 100 * len(stat.prices) / stat.product_count,
        "historical_price_p05": price_p05,
        "historical_price_p50": price_p50,
        "historical_price_p95": price_p95,
    }


def build_inventory(
    products_path: Path,
    offers_path: Path,
    comments_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build category stats using one exact-token streaming pass over comments."""
    started = time.monotonic()
    stats: dict[tuple[str, str, str | None], ScopeStats] = {}
    product_scopes: dict[int, tuple[tuple[str, str, str | None], ...]] = {}
    product_columns = ("product_id", "category1", "category2", "brand")

    for batch in pq.ParquetFile(products_path).iter_batches(
        batch_size=65_536, columns=product_columns
    ):
        values = batch.to_pydict()
        for product_id, category1, category2, brand in zip(
            *(values[column] for column in product_columns)
        ):
            keys = _scope_keys(category1, category2)
            product_scopes[product_id] = keys
            for key in keys:
                stat = _scope_stat(stats, key)
                stat.product_count += 1
                if brand is not None:
                    stat.brands.add(brand)

    min_prices: dict[int, int] = {}
    for batch in pq.ParquetFile(offers_path).iter_batches(
        batch_size=65_536, columns=("product_id", "price_raw")
    ):
        product_ids, prices = batch.column(0).to_pylist(), batch.column(1).to_pylist()
        for product_id, price in zip(product_ids, prices):
            if product_id not in product_scopes or price is None or price <= 0:
                continue
            previous = min_prices.get(product_id)
            if previous is None or price < previous:
                min_prices[product_id] = price
    for product_id, price in min_prices.items():
        for key in product_scopes[product_id]:
            _scope_stat(stats, key).prices.append(price)

    comment_counts: Counter[int] = Counter()
    for batch_number, batch in enumerate(
        iter_exact_csv_batches(
            comments_path,
            COMMENT_SOURCE_COLUMNS,
            chunksize=100_000,
            max_rows=None,
            field_size_limit=COMMENT_CSV_FIELD_SIZE_LIMIT,
            dataset_name="comments",
        ),
        start=1,
    ):
        for _, raw in batch:
            parsed_product = parse_required_id(raw["product_id"])
            if not parsed_product.valid:
                continue
            keys = product_scopes.get(parsed_product.value)
            if not keys:
                continue
            comment_counts[parsed_product.value] += 1
            has_body = clean_comment_text(raw["body"], "body") is not None
            buyer = parse_strict_boolean(raw["is_buyer"])
            rate, _, invalid_rate, _ = _parse_comment_rate(raw["rate"])
            for key in keys:
                stat = _scope_stat(stats, key)
                stat.comment_count += 1
                stat.non_empty_body_count += has_body
                stat.buyer_comment_count += buyer.valid and buyer.value
                stat.valid_positive_rating_count += rate is not None and not invalid_rate
        if progress is not None and batch_number % 10 == 0:
            progress(f"processed {batch_number * 100_000:,} comment records")

    for product_id, keys in product_scopes.items():
        count = comment_counts.get(product_id, 0)
        if count:
            for key in keys:
                _scope_stat(stats, key).commented_counts.append(count)

    rows = [_row(stat) for _, stat in sorted(stats.items())]
    reconciliation = _reconcile(rows)
    if not all(reconciliation.values()):
        raise RuntimeError(f"inventory reconciliation failed: {reconciliation}")
    metadata = {
        "runtime_seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "product_mapping_count": len(product_scopes),
        "reconciliation": reconciliation,
    }
    return rows, metadata


def _reconcile(rows: Iterable[dict[str, Any]]) -> dict[str, bool]:
    checks = []
    for row in rows:
        checks.extend(
            (
                row["product_count"]
                == row["products_with_comments"] + row["products_without_comments"],
                row["products_with_at_least_50_comments"]
                <= row["products_with_at_least_20_comments"]
                <= row["products_with_at_least_10_comments"]
                <= row["products_with_at_least_5_comments"]
                <= row["products_with_comments"],
                row["non_empty_body_count"] <= row["comment_count"],
                row["buyer_comment_count"] <= row["comment_count"],
                row["valid_positive_rating_count"] <= row["comment_count"],
                row["products_with_valid_historical_price"] <= row["product_count"],
            )
        )
    return {"all_scope_invariants": all(checks)}


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_inventory(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=INVENTORY_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row[column]) for column in INVENTORY_COLUMNS})


def _number(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{value:,.0f}"


def _percentage(value: float) -> str:
    return f"{value:.2f}%"


def _is_coherent_pair(row: dict[str, Any]) -> bool:
    return (
        row["scope_level"] == "category1_category2"
        and row["product_count"] >= 150
        and row["comment_count"] >= 1_000
        and row["products_with_comments"] >= 75
        and row["products_with_at_least_10_comments"] >= 20
        and row["non_empty_body_count"] >= 800
        and row["unique_brand_count"] >= 8
        and row["historical_price_coverage_pct"] >= 60
        and not any(term in row["category1"] for term in COHERENCE_EXCLUDED_CATEGORY1_TERMS)
    )


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| Scope | Products | Comments | With comments | ≥5 / ≥10 / ≥20 / ≥50 | "
            "Median / p95 | Bodies | Brands | Historical-price coverage |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scope = row["category1"]
        if row["category2"] is not None:
            scope = f"{scope} > {row['category2']}"
        lines.append(
            "| "
            + " | ".join(
                (
                    scope,
                    _number(row["product_count"]),
                    _number(row["comment_count"]),
                    _number(row["products_with_comments"]),
                    " / ".join(
                        _number(row[column])
                        for column in (
                            "products_with_at_least_5_comments",
                            "products_with_at_least_10_comments",
                            "products_with_at_least_20_comments",
                            "products_with_at_least_50_comments",
                        )
                    ),
                    f"{_number(row['median_comments_per_commented_product'])} / {_number(row['p95_comments_per_commented_product'])}",
                    _number(row["non_empty_body_count"]),
                    _number(row["unique_brand_count"]),
                    _percentage(row["historical_price_coverage_pct"]),
                )
            )
            + " |"
        )
    return lines


def validate_sunscreen(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    index = {(row["scope_level"], row["category1"], row["category2"]): row for row in rows}
    sunscreen = index.get(SUNSCREEN_KEY)
    if sunscreen is None:
        raise RuntimeError("sunscreen scope is absent from the category inventory")
    mismatches = {
        column: (expected, sunscreen[column])
        for column, expected in SUNSCREEN_EXPECTED.items()
        if sunscreen[column] != expected
    }
    expected_coverage = 100 * 1046 / 1048
    if not math.isclose(sunscreen["historical_price_coverage_pct"], expected_coverage):
        mismatches["historical_price_coverage_pct"] = (
            expected_coverage,
            sunscreen["historical_price_coverage_pct"],
        )
    if mismatches:
        raise RuntimeError(f"sunscreen validation failed: {mismatches}")
    return sunscreen


def write_summary(
    rows: list[dict[str, Any]], metadata: dict[str, Any], output_path: Path
) -> None:
    sunscreen = validate_sunscreen(rows)
    category1_rows = [row for row in rows if row["scope_level"] == "category1"]
    category1_rows.sort(key=lambda row: (-row["products_with_at_least_10_comments"], row["category1"]))
    coherent = [row for row in rows if _is_coherent_pair(row)]
    coherent.sort(
        key=lambda row: (
            -row["products_with_at_least_10_comments"],
            -row["products_with_comments"],
            -row["comment_count"],
            row["category1"],
            row["category2"],
        )
    )
    named_keys = (
        SUNSCREEN_KEY,
        ("category1_category2", "شامپو و مراقبت مو", "شامپو مو"),
        ("category1_category2", "مراقبت پوست", "پاک کننده آرایش صورت"),
        ("category1_category2", "مراقبت پوست", "ماسک صورت و بدن"),
        ("category1_category2", "مراقبت پوست", "کرم مرطوب کننده و نرم کننده"),
    )
    index = {(row["scope_level"], row["category1"], row["category2"]): row for row in rows}
    named = [index[key] for key in named_keys]
    lines = [
        "# Category viability inventory",
        "",
        "## Methodology",
        "",
        "- Sources: canonical `products_clean.parquet`, `offers_clean.parquet`, and one sequential exact-token pass over `digikala-comments.csv`.",
        "- Authoritative scopes are `Category1` and hierarchical `Category1 > Category2`; `sub_category` is not used for selection.",
        "- Product-level price is the lowest valid positive historical `price_raw` offer for that canonical product. Percentiles are calculated over those product-level prices.",
        "- Comment text is never materialized as a dataset: only aggregate counters and a per-product comment counter are retained.",
        "- The top-pair table applies transparent evidence thresholds (at least 150 products, 1,000 comments, 75 commented products, 20 products with 10+ comments, 800 non-empty bodies, 8 brands, and 60% price coverage) and excludes broad book, stationery, toy, clothing, accessory, jewellery, and child-category families.",
        "",
        "## Top 20 coherent `Category1 > Category2` scopes",
        "",
        *_markdown_table(coherent[:20]),
        "",
        "## Category1 summary",
        "",
        *_markdown_table(category1_rows[:20]),
        "",
        "## Selected comparison scopes",
        "",
        *_markdown_table(named),
        "",
        "## Why sunscreen was selected",
        "",
        f"`مراقبت پوست > کرم ضد آفتاب` has {_number(sunscreen['product_count'])} products, {_number(sunscreen['comment_count'])} comments across {_number(sunscreen['products_with_comments'])} products, {_number(sunscreen['unique_brand_count'])} brands, and {_percentage(sunscreen['historical_price_coverage_pct'])} historical-price coverage. Its review-depth counts are {_number(sunscreen['products_with_at_least_5_comments'])} / {_number(sunscreen['products_with_at_least_10_comments'])} / {_number(sunscreen['products_with_at_least_20_comments'])} / {_number(sunscreen['products_with_at_least_50_comments'])} at the 5 / 10 / 20 / 50 thresholds.",
        "",
        "Sunscreen was selected as the best deadline-constrained MVP scope because it combines sufficient scale, strong review coverage, brand and price diversity, narrow semantic coherence, and a clear comparison-oriented use case. It is not claimed to have the highest raw product or comment count.",
        "",
        "## Rejected device scopes",
        "",
        "- Laptop: no strictly categorized laptop products; title mentions were false or ambiguous.",
        "- Mobile phone: no strict hardware category; the only mobile-labelled Category1 was training content, and title matches were non-phone items.",
        "- Tablet: `Category1 = تبلت` has 81 products and zero matching comments.",
        "",
        "## Limitations",
        "",
        "- Categories are source labels and may contain occasional semantic surprises; no category correction was applied.",
        "- Product prices are historical inferred IRR, not current, latest, or live prices. They must not be presented as current market prices.",
        "- Raw comment rows are counted here; this report does not deduplicate or canonicalize comments.",
        "",
        "## Run metadata",
        "",
        f"- Canonical product mapping: {_number(metadata['product_mapping_count'])} products",
        f"- Runtime: {metadata['runtime_seconds']:.2f} seconds",
        f"- Maximum RSS: {_number(metadata['max_rss_kib'])} KiB",
        f"- Reconciliation: {'passed' if all(metadata['reconciliation'].values()) else 'failed'}",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the category-viability inventory.")
    parser.add_argument("--products", type=Path, default=Path("data/processed/products_v1/products_clean.parquet"))
    parser.add_argument("--offers", type=Path, default=Path("data/processed/products_v1/offers_clean.parquet"))
    parser.add_argument("--comments", type=Path, default=Path("data/raw/digikala-comments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diagnostics"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, metadata = build_inventory(args.products, args.offers, args.comments, progress=print)
    write_inventory(rows, args.output_dir / "category_viability_inventory.csv")
    write_summary(rows, metadata, args.output_dir / "category_viability_summary.md")
    print(f"[category-viability] wrote {len(rows):,} scopes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
