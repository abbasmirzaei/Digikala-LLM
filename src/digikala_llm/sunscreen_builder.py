"""One-pass, sunscreen-only scoped dataset builder; intentionally no SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.cleaning import (
    COMMENTS_CLEAN_SCHEMA,
    PRODUCTS_CLEAN_SCHEMA,
    clean_comment_text,
    parse_required_id,
    parse_strict_boolean,
    transform_comment_row,
)
from digikala_llm.comments_pipeline import (
    COMMENT_CSV_FIELD_SIZE_LIMIT,
    COMMENT_MAPPED_FIELDS,
    COMMENT_SOURCE_COLUMNS,
)
from digikala_llm.csv_stream import iter_exact_csv_batches
from digikala_llm.products_pipeline import _exact_values

SPECIFICATION_VERSION = "002-sunscreen-mvp-1"
CATEGORY1 = "مراقبت پوست"
CATEGORY2 = "کرم ضد آفتاب"
HISTORICAL_PRICE_LABEL = "historical inferred IRR"
OUTPUT_FILENAMES = (
    "sunscreen_products.parquet",
    "sunscreen_prices.parquet",
    "sunscreen_comments_raw.parquet",
    "sunscreen_comments_canonical.parquet",
    "sunscreen_comment_audit.parquet",
)
PRICE_SCHEMA = pa.schema(
    [
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("historical_price_inferred_irr", pa.int64()),
        pa.field("valid_price_offer_count", pa.int64(), nullable=False),
        pa.field("historical_price_label", pa.string(), nullable=False),
    ]
)
RAW_COMMENT_SCHEMA = pa.schema(
    [pa.field("source_row_number", pa.int64(), nullable=False), pa.field("product_id", pa.int64(), nullable=False)]
    + [pa.field(f"raw_{column}", pa.string(), nullable=False) for column in COMMENT_SOURCE_COLUMNS]
)
AUDIT_SCHEMA = pa.schema(
    [
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("comment_id", pa.int64()),
        pa.field("product_id", pa.int64()),
        pa.field("event", pa.string(), nullable=False),
        pa.field("detail", pa.string(), nullable=False),
    ]
)
OUTPUT_SCHEMAS = {
    "sunscreen_products.parquet": PRODUCTS_CLEAN_SCHEMA,
    "sunscreen_prices.parquet": PRICE_SCHEMA,
    "sunscreen_comments_raw.parquet": RAW_COMMENT_SCHEMA,
    "sunscreen_comments_canonical.parquet": COMMENTS_CLEAN_SCHEMA,
    "sunscreen_comment_audit.parquet": AUDIT_SCHEMA,
}
EVIDENCE_GATE = {
    "selected_products": 1048,
    "raw_matching_comment_rows": 53522,
    "products_with_raw_comments": 892,
    "brands": 175,
    "buyer_comment_rows": 50275,
    "non_empty_body_rows": 53517,
    "products_with_historical_price": 1046,
}


def _write_parquet(path: Path, schema: pa.Schema, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    pq.write_table(pa.Table.from_pylist(materialized, schema=schema), path, compression="snappy")
    return len(materialized)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_stat(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _comment_clean_blob(candidate: dict[str, Any]) -> bytes:
    fields = (
        "product_id", "title", "body", "created_at_raw", "created_at_jalali",
        "created_at_gregorian", "rate", "is_unrated", "invalid_rate",
        "recommendation_status", "is_buyer", "advantages", "disadvantages", "likes",
        "dislikes", "seller_title", "seller_code", "true_to_size_rate",
    )
    values = []
    for field in fields:
        value = candidate[field]
        values.append(value.isoformat() if hasattr(value, "isoformat") else value)
    return _exact_values(values)


def _candidate_rank(candidate: dict[str, Any], raw: dict[str, str]) -> tuple[int, str, int]:
    mapped = tuple(clean_comment_text(raw[field], field) for field in COMMENT_MAPPED_FIELDS)
    digest = hashlib.sha256(_exact_values(mapped)).hexdigest()
    return (-sum(value is not None for value in mapped), digest, candidate["source_row_number"])


def _canonicalize(
    candidates: list[tuple[dict[str, Any], dict[str, str]]], audit: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id: dict[int, list[tuple[dict[str, Any], dict[str, str]]]] = defaultdict(list)
    for item in candidates:
        by_id[item[0]["comment_id"]].append(item)
    final: list[dict[str, Any]] = []
    metrics = Counter()
    for comment_id in sorted(by_id):
        group = by_id[comment_id]
        if len(group) > 1:
            metrics["repeated_comment_id_count"] += 1
            metrics["repeated_comment_id_excess_rows"] += len(group) - 1
        representatives: dict[bytes, tuple[dict[str, Any], dict[str, str]]] = {}
        for candidate, raw in group:
            clean_blob = _comment_clean_blob(candidate)
            current = representatives.get(clean_blob)
            if current is None or _candidate_rank(candidate, raw) < _candidate_rank(*current):
                representatives[clean_blob] = (candidate, raw)
        metrics["exact_clean_duplicate_rows_removed"] += len(group) - len(representatives)
        choices = list(representatives.values())
        selected, _selected_raw = min(choices, key=lambda item: _candidate_rank(*item))
        conflict = len(choices) > 1
        if conflict:
            metrics["conflicting_comment_id_count"] += 1
            metrics["conflict_alternative_count"] += len(choices) - 1
        for candidate, _ in choices:
            if candidate is not selected:
                audit.append(
                    {
                        "source_row_number": candidate["source_row_number"],
                        "comment_id": candidate["comment_id"],
                        "product_id": candidate["product_id"],
                        "event": "conflicting_comment_id_noncanonical",
                        "detail": f"canonical_source_row={selected['source_row_number']}",
                    }
                )
        final.append(
            {
                **{field.name: selected[field.name] for field in COMMENTS_CLEAN_SCHEMA if field.name not in {"comment_id_conflict", "canonical_source_row_number"}},
                "comment_id_conflict": conflict,
                "canonical_source_row_number": selected["source_row_number"],
            }
        )
    return final, dict(metrics)


def _validate_gate(metrics: dict[str, Any]) -> None:
    mismatches = {key: (expected, metrics[key]) for key, expected in EVIDENCE_GATE.items() if metrics[key] != expected}
    expected_coverage = 100 * EVIDENCE_GATE["products_with_historical_price"] / EVIDENCE_GATE["selected_products"]
    if round(metrics["historical_price_coverage_pct"], 2) != round(expected_coverage, 2):
        mismatches["historical_price_coverage_pct"] = (expected_coverage, metrics["historical_price_coverage_pct"])
    if mismatches:
        raise RuntimeError(f"sunscreen evidence gate failed: {mismatches}")


def build_sunscreen_dataset(
    products_path: Path | str,
    offers_path: Path | str,
    comments_path: Path | str,
    output_dir: Path | str,
    *,
    chunksize: int = 100_000,
    force: bool = False,
    validate_evidence: bool = True,
    failure_injector: Callable[[str], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Publish a versioned sunscreen-only artifact directory atomically."""
    products_path, offers_path, comments_path = (
        Path(value).resolve() for value in (products_path, offers_path, comments_path)
    )
    output_dir = Path(output_dir).resolve()
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if output_dir.exists() and not force:
        raise FileExistsError(output_dir)
    for path in (products_path, offers_path, comments_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    started = time.monotonic()
    try:
        product_rows = []
        for batch in pq.ParquetFile(products_path).iter_batches(batch_size=65_536):
            for row in pa.Table.from_batches([batch]).to_pylist():
                if row["category1"] == CATEGORY1 and row["category2"] == CATEGORY2:
                    product_rows.append(row)
        product_rows.sort(key=lambda row: row["product_id"])
        product_ids = {row["product_id"] for row in product_rows}
        if len(product_ids) != len(product_rows):
            raise RuntimeError("scoped products contain duplicate product_id")
        prices: dict[int, tuple[int, int]] = {}
        for batch in pq.ParquetFile(offers_path).iter_batches(batch_size=65_536, columns=["product_id", "price_raw"]):
            for product_id, price in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
                if product_id not in product_ids or price is None or price <= 0:
                    continue
                previous = prices.get(product_id)
                prices[product_id] = (price if previous is None else min(price, previous[0]), 1 if previous is None else previous[1] + 1)
        price_rows = [
            {"product_id": row["product_id"], "historical_price_inferred_irr": prices.get(row["product_id"], (None, 0))[0], "valid_price_offer_count": prices.get(row["product_id"], (None, 0))[1], "historical_price_label": HISTORICAL_PRICE_LABEL}
            for row in product_rows
        ]
        raw_rows: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], dict[str, str]]] = []
        audit: list[dict[str, Any]] = []
        exact_rows: set[bytes] = set()
        metrics: Counter[str] = Counter()
        commented_products: set[int] = set()
        source_before = _source_stat(comments_path)
        for batch_number, batch in enumerate(
            iter_exact_csv_batches(
                comments_path,
                COMMENT_SOURCE_COLUMNS,
                chunksize=chunksize,
                max_rows=None,
                field_size_limit=COMMENT_CSV_FIELD_SIZE_LIMIT,
                dataset_name="comments",
            ),
            start=1,
        ):
            for source_row, raw in batch:
                parsed_product = parse_required_id(raw["product_id"])
                if not parsed_product.valid or parsed_product.value not in product_ids:
                    continue
                product_id = parsed_product.value
                metrics["raw_matching_comment_rows"] += 1
                commented_products.add(product_id)
                buyer = parse_strict_boolean(raw["is_buyer"])
                metrics["buyer_comment_rows"] += buyer.valid and buyer.value
                metrics["non_empty_body_rows"] += clean_comment_text(raw["body"], "body") is not None
                raw_rows.append({"source_row_number": source_row, "product_id": product_id, **{f"raw_{column}": raw[column] for column in COMMENT_SOURCE_COLUMNS}})
                exact = _exact_values(raw[column] for column in COMMENT_SOURCE_COLUMNS)
                if exact in exact_rows:
                    metrics["exact_raw_duplicate_rows_removed"] += 1
                    audit.append({"source_row_number": source_row, "comment_id": None, "product_id": product_id, "event": "exact_raw_duplicate", "detail": "excluded before canonicalization"})
                    continue
                exact_rows.add(exact)
                transformed = transform_comment_row(raw, source_row)
                if transformed.quarantined:
                    metrics["transformation_quarantine_rows"] += 1
                    audit.append({"source_row_number": source_row, "comment_id": None, "product_id": product_id, "event": "transformation_quarantine", "detail": transformed.quarantine_record["reason"] if transformed.quarantine_record else "invalid comment"})
                    continue
                assert transformed.candidate_row is not None
                candidates.append((transformed.candidate_row, raw))
            if progress is not None:
                progress(f"scanned {batch_number * chunksize:,} raw comment records")
        if _source_stat(comments_path) != source_before:
            raise RuntimeError("comments source changed during its one-pass build")
        canonical_rows, canonical_metrics = _canonicalize(candidates, audit)
        metrics.update(canonical_metrics)
        metrics["selected_products"] = len(product_rows)
        metrics["products_with_raw_comments"] = len(commented_products)
        metrics["brands"] = len({row["brand"] for row in product_rows if row["brand"] is not None})
        metrics["products_with_historical_price"] = len(prices)
        metrics["historical_price_coverage_pct"] = 100 * len(prices) / len(product_rows)
        metrics["canonical_comment_rows"] = len(canonical_rows)
        metrics["matching_candidates_retained"] = len(candidates)
        if validate_evidence:
            _validate_gate(metrics)
        if failure_injector is not None:
            failure_injector("after_validation")
        output_counts = {
            "sunscreen_products.parquet": _write_parquet(staging / "sunscreen_products.parquet", PRODUCTS_CLEAN_SCHEMA, product_rows),
            "sunscreen_prices.parquet": _write_parquet(staging / "sunscreen_prices.parquet", PRICE_SCHEMA, price_rows),
            "sunscreen_comments_raw.parquet": _write_parquet(staging / "sunscreen_comments_raw.parquet", RAW_COMMENT_SCHEMA, raw_rows),
            "sunscreen_comments_canonical.parquet": _write_parquet(staging / "sunscreen_comments_canonical.parquet", COMMENTS_CLEAN_SCHEMA, sorted(canonical_rows, key=lambda row: row["comment_id"])),
            "sunscreen_comment_audit.parquet": _write_parquet(staging / "sunscreen_comment_audit.parquet", AUDIT_SCHEMA, sorted(audit, key=lambda row: (row["source_row_number"], row["event"]))),
        }
        if output_counts["sunscreen_comments_raw.parquet"] != metrics["raw_matching_comment_rows"]:
            raise RuntimeError("raw scoped comment count does not reconcile")
        for filename, schema in OUTPUT_SCHEMAS.items():
            if pq.read_schema(staging / filename) != schema:
                raise RuntimeError(f"schema round-trip failed for {filename}")
        manifest = {
            "specification_version": SPECIFICATION_VERSION,
            "scope": {"category1": CATEGORY1, "category2": CATEGORY2},
            "historical_price": {"label": HISTORICAL_PRICE_LABEL, "method": "lowest valid positive price_raw per product"},
            "sources": {"products": _source_stat(products_path), "offers": _source_stat(offers_path), "comments": source_before},
            "configuration": {"chunksize": chunksize, "one_raw_comment_scan": True, "sqlite_used": False},
            "rows": dict(sorted(metrics.items())),
            "schemas": {name: str(schema) for name, schema in OUTPUT_SCHEMAS.items()},
            "outputs": {name: {"rows": output_counts[name], "sha256": _sha256(staging / name)} for name in OUTPUT_FILENAMES},
            "runtime_seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "status": "success",
        }
        manifest["content_fingerprint"] = hashlib.sha256(json.dumps(manifest["outputs"], sort_keys=True).encode()).hexdigest()
        _write_json(staging / "manifest.json", manifest)
        (staging / "_SUCCESS").write_text("\n", encoding="utf-8")
        if output_dir.exists():
            if not force:
                raise FileExistsError(output_dir)
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                shutil.rmtree(backup)
            output_dir.rename(backup)
            try:
                staging.rename(output_dir)
            except Exception:
                backup.rename(output_dir)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the sunscreen-only scoped datasets.")
    parser.add_argument("--products", type=Path, default=Path("data/processed/products_v1/products_clean.parquet"))
    parser.add_argument("--offers", type=Path, default=Path("data/processed/products_v1/offers_clean.parquet"))
    parser.add_argument("--comments", type=Path, default=Path("data/raw/digikala-comments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sunscreen_mvp/v1"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_sunscreen_dataset(
        args.products,
        args.offers,
        args.comments,
        args.output_dir,
        force=args.force,
        progress=lambda message: print(f"[sunscreen-build] {message}", flush=True),
    )
    print(json.dumps({"rows": manifest["rows"], "runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
