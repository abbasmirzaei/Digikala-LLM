"""Bounded-memory dataset-level cleaning for Digikala products and offers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.cleaning import (
    OFFERS_CLEAN_SCHEMA,
    PRODUCT_CONFLICT_SCHEMA,
    PRODUCTS_CLEAN_SCHEMA,
    QUARANTINE_SCHEMA,
    ROW_AUDIT_SCHEMA,
    transform_offer_row,
    transform_product_row,
)
from digikala_llm.csv_stream import iter_exact_csv_batches

SPECIFICATION_VERSION = "1.0.6"
COMPLETION_MARKER = "_SUCCESS"
CORE_SOURCE_FIELDS = (
    "title_fa",
    "Category1",
    "Category2",
    "Brand",
    "Rate",
    "Rate_cnt",
    "sub_category",
)
REQUIRED_PRODUCT_COLUMNS = {
    "id",
    "title_fa",
    "Category1",
    "Category2",
    "Brand",
    "Rate",
    "Rate_cnt",
    "sub_category",
    "Seller",
    "Price",
    "Is_Fake",
    "min_price_last_month",
}
PRODUCT_SOURCE_COLUMNS = (
    "id",
    "title_fa",
    "Rate",
    "Rate_cnt",
    "Category1",
    "Category2",
    "Brand",
    "Price",
    "Seller",
    "Is_Fake",
    "min_price_last_month",
    "sub_category",
)
# Deliberately bounded well above observed fields; oversized records fail explicitly.
CSV_FIELD_SIZE_LIMIT = 16 * 1024 * 1024
OUTPUT_SCHEMAS = {
    "products_clean.parquet": PRODUCTS_CLEAN_SCHEMA,
    "offers_clean.parquet": OFFERS_CLEAN_SCHEMA,
    "product_conflicts.parquet": PRODUCT_CONFLICT_SCHEMA,
    "product_quarantine.parquet": QUARANTINE_SCHEMA,
    "offer_quarantine.parquet": QUARANTINE_SCHEMA,
    "row_audit.parquet": ROW_AUDIT_SCHEMA,
}
PRODUCT_OFFER_RULE_IDS = (
    "RAW-001",
    "ID-001",
    "ID-002",
    "BOOL-001",
    "BOOL-002",
    "TXT-001",
    "TXT-002",
    *(f"PRD-{number:03d}" for number in range(1, 9)),
    *(f"OFF-{number:03d}" for number in range(1, 10)),
)
ROW_METRIC_NAMES = (
    "input_rows",
    "exact_duplicate_rows_removed",
    "distinct_raw_rows_retained",
    "accepted_product_candidates",
    "product_quarantine_rows",
    "accepted_offer_candidates",
    "offer_transform_quarantine_rows",
    "distinct_offer_candidates",
    "exact_offer_duplicates_removed",
)


def _progress(message: str) -> None:
    print(f"[clean-products] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _exact_values(values: Iterable[object]) -> bytes:
    """Collision-safe canonical serialization with type and length markers."""
    serialized = bytearray()
    for value in values:
        if value is None:
            payload = b""
            marker = b"N"
        elif isinstance(value, bool):
            payload = b"1" if value else b"0"
            marker = b"B"
        elif isinstance(value, int):
            payload = str(value).encode("ascii")
            marker = b"I"
        elif isinstance(value, float):
            payload = repr(value).encode("ascii")
            marker = b"F"
        elif isinstance(value, Decimal):
            payload = str(value).encode("ascii")
            marker = b"D"
        else:
            payload = str(value).encode("utf-8")
            marker = b"S"
        serialized.extend(marker)
        serialized.extend(len(payload).to_bytes(8, "big"))
        serialized.extend(payload)
    return bytes(serialized)


def _raw_row_content(columns: list[str], raw: dict[str, object]) -> bytes:
    return _exact_values(raw.get(column) for column in columns)


def _raw_digest(content: bytes) -> str:
    """Return an informational fingerprint; exact bytes remain the equality key."""
    return hashlib.sha256(content).hexdigest()


def _core_content(candidate: dict[str, Any]) -> bytes:
    return _exact_values(
        candidate.get(field)
        for field in (
            "title_fa",
            "category1",
            "category2",
            "brand",
            "rate",
            "rate_count",
            "sub_category",
        )
    )


def _offer_content(raw: dict[str, object]) -> bytes:
    return _exact_values(
        raw.get(field)
        for field in ("id", "Seller", "Price", "Is_Fake", "min_price_last_month")
    )


def _remove_sqlite_files(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(
            """
        PRAGMA journal_mode=DELETE;
        PRAGMA temp_store=FILE;
        CREATE TABLE raw_rows (
            content BLOB PRIMARY KEY,
            digest TEXT NOT NULL,
            source_row INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE product_candidates (
            source_row INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            title_fa TEXT,
            category1 TEXT,
            category2 TEXT,
            brand TEXT,
            rate REAL,
            rate_count INTEGER,
            sub_category TEXT,
            is_unrated INTEGER NOT NULL,
            inconsistent_zero_rate INTEGER NOT NULL,
            completeness INTEGER NOT NULL,
            core_blob BLOB NOT NULL,
            core_digest TEXT NOT NULL,
            raw_core_json TEXT NOT NULL
        );
        CREATE INDEX product_candidates_grouping
            ON product_candidates(product_id, rate_count, completeness, core_digest, source_row);
        CREATE TABLE offers (
            offer_blob BLOB PRIMARY KEY,
            offer_id TEXT NOT NULL,
            product_id INTEGER NOT NULL,
            seller TEXT,
            price_raw INTEGER,
            price_toman TEXT,
            is_fake INTEGER,
            min_price_last_month INTEGER,
            missing_price_history INTEGER NOT NULL,
            invalid_price INTEGER NOT NULL,
            source_row INTEGER NOT NULL,
            raw_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX offers_sorting ON offers(product_id, offer_id);
        CREATE TABLE final_products (
            product_id INTEGER PRIMARY KEY,
            title_fa TEXT,
            category1 TEXT,
            category2 TEXT,
            brand TEXT,
            rate REAL,
            rate_count INTEGER,
            sub_category TEXT,
            is_unrated INTEGER NOT NULL,
            inconsistent_zero_rate INTEGER NOT NULL,
            core_attribute_conflict INTEGER NOT NULL,
            canonical_source_row INTEGER NOT NULL
        );
        CREATE TABLE conflicts (
            product_id INTEGER NOT NULL,
            title_fa TEXT,
            category1 TEXT,
            category2 TEXT,
            brand TEXT,
            rate REAL,
            rate_count INTEGER,
            sub_category TEXT,
            is_unrated INTEGER NOT NULL,
            inconsistent_zero_rate INTEGER NOT NULL,
            candidate_source_row INTEGER NOT NULL,
            canonical_source_row INTEGER NOT NULL,
            completeness INTEGER NOT NULL,
            core_digest TEXT NOT NULL,
            raw_core_json TEXT NOT NULL,
            selected INTEGER NOT NULL,
            PRIMARY KEY(product_id, core_blob_placeholder)
        );
            """.replace(
                "core_blob_placeholder", "core_digest, candidate_source_row"
            )
        )
        connection.executescript(
            """
        CREATE TABLE quarantine (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            side TEXT NOT NULL,
            record_json TEXT NOT NULL
        );
        CREATE TABLE row_audit (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            record_json TEXT NOT NULL
        );
            """
        )
    except Exception:
        connection.close()
        raise
    return connection


def _store_audits(connection: sqlite3.Connection, records: Iterable[dict[str, Any]]) -> None:
    connection.executemany(
        "INSERT INTO row_audit(record_json) VALUES (?)",
        ((_json_text(record),) for record in records),
    )


def _with_source_file(records: Iterable[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    adjusted = []
    for record in records:
        copy = dict(record)
        copy["source_file"] = str(source)
        adjusted.append(copy)
    return adjusted


def _store_quarantine(
    connection: sqlite3.Connection, side: str, record: dict[str, Any]
) -> None:
    connection.execute(
        "INSERT INTO quarantine(side, record_json) VALUES (?, ?)",
        (side, _json_text(record)),
    )


def _count_result_events(
    side: str,
    result: Any,
    aggregate_counts: Counter[str],
    rule_counts: Counter[str],
) -> None:
    for rule_id, field in result.aggregate_counter_keys:
        aggregate_counts[f"{side}:{rule_id}:{field}"] += 1
        rule_counts[rule_id] += 1
    for record in result.audit_records:
        rule_counts[record["rule_id"]] += 1


def _iter_product_csv_batches(
    input_path: Path,
    chunksize: int,
    max_rows: int | None,
) -> Iterator[list[tuple[int, dict[str, str]]]]:
    """Yield exact-token CSV records with logical, not physical-line, row numbers."""
    yield from iter_exact_csv_batches(
        input_path,
        PRODUCT_SOURCE_COLUMNS,
        chunksize=chunksize,
        max_rows=max_rows,
        field_size_limit=CSV_FIELD_SIZE_LIMIT,
        dataset_name="products",
    )


def _insert_product_candidate(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
    raw: dict[str, object],
) -> None:
    core_blob = _core_content(candidate)
    core_digest = hashlib.sha256(core_blob).hexdigest()
    core_fields = (
        candidate["title_fa"],
        candidate["category1"],
        candidate["category2"],
        candidate["brand"],
        candidate["rate"],
        candidate["rate_count"],
        candidate["sub_category"],
    )
    completeness = sum(value is not None for value in core_fields)
    raw_core = {field: raw.get(field) for field in CORE_SOURCE_FIELDS}
    connection.execute(
        """
        INSERT INTO product_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate["source_row_number"],
            candidate["product_id"],
            candidate["title_fa"],
            candidate["category1"],
            candidate["category2"],
            candidate["brand"],
            candidate["rate"],
            candidate["rate_count"],
            candidate["sub_category"],
            int(candidate["is_unrated"]),
            int(candidate["inconsistent_zero_rate"]),
            completeness,
            core_blob,
            core_digest,
            _json_text(raw_core),
        ),
    )


def _insert_offer_candidate(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
    raw: dict[str, object],
) -> bool:
    offer_blob = _offer_content(raw)
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO offers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            offer_blob,
            candidate["offer_id"],
            candidate["product_id"],
            candidate["seller"],
            candidate["price_raw"],
            str(candidate["price_toman"]) if candidate["price_toman"] is not None else None,
            int(candidate["is_fake"]) if candidate["is_fake"] is not None else None,
            candidate["min_price_last_month"],
            int(candidate["missing_price_history"]),
            int(candidate["invalid_price"]),
            candidate["source_row_number"],
            _json_text(raw),
        ),
    )
    return cursor.rowcount == 1


def _ingest(
    connection: sqlite3.Connection,
    input_path: Path,
    chunksize: int,
    max_rows: int | None,
    failure_injector: Callable[[str], None] | None,
) -> tuple[dict[str, int], Counter[str], Counter[str], Counter[str], Counter[str]]:
    metrics = Counter({name: 0 for name in ROW_METRIC_NAMES})
    aggregate_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter({rule_id: 0 for rule_id in PRODUCT_OFFER_RULE_IDS})
    product_quarantine_rules: Counter[str] = Counter()
    offer_quarantine_rules: Counter[str] = Counter()
    saw_rows = False
    columns = list(PRODUCT_SOURCE_COLUMNS)
    for chunk_number, batch in enumerate(
        _iter_product_csv_batches(input_path, chunksize, max_rows), start=1
    ):
        saw_rows = True
        for source_row, raw in batch:
            content = _raw_row_content(columns, raw)
            cursor = connection.execute(
                "INSERT OR IGNORE INTO raw_rows VALUES (?, ?, ?)",
                (content, _raw_digest(content), source_row),
            )
            if cursor.rowcount == 0:
                metrics["exact_duplicate_rows_removed"] += 1
                rule_counts["PRD-001"] += 1
                continue
            metrics["distinct_raw_rows_retained"] += 1

            product = transform_product_row(raw, source_row)
            _count_result_events("products", product, aggregate_counts, rule_counts)
            product_audits = _with_source_file(product.audit_records, input_path)
            _store_audits(connection, product_audits)
            if product.quarantined:
                metrics["product_quarantine_rows"] += 1
                assert product.quarantine_record is not None
                product_quarantine = dict(product.quarantine_record)
                product_quarantine["source_file"] = str(input_path)
                _store_quarantine(connection, "products", product_quarantine)
                for rule_id in product_quarantine["rule_ids"]:
                    product_quarantine_rules[rule_id] += 1
            else:
                metrics["accepted_product_candidates"] += 1
                assert product.candidate_row is not None
                _insert_product_candidate(connection, product.candidate_row, raw)

            offer = transform_offer_row(raw, source_row)
            _count_result_events("offers", offer, aggregate_counts, rule_counts)
            offer_audits = _with_source_file(offer.audit_records, input_path)
            _store_audits(connection, offer_audits)
            if offer.quarantined:
                metrics["offer_transform_quarantine_rows"] += 1
                assert offer.quarantine_record is not None
                offer_quarantine = dict(offer.quarantine_record)
                offer_quarantine["source_file"] = str(input_path)
                _store_quarantine(connection, "offers", offer_quarantine)
                for rule_id in offer_quarantine["rule_ids"]:
                    offer_quarantine_rules[rule_id] += 1
            else:
                metrics["accepted_offer_candidates"] += 1
                assert offer.candidate_row is not None
                if _insert_offer_candidate(connection, offer.candidate_row, raw):
                    metrics["distinct_offer_candidates"] += 1
                else:
                    metrics["exact_offer_duplicates_removed"] += 1
                    rule_counts["OFF-001"] += 1
        metrics["input_rows"] += len(batch)
        connection.commit()
        _progress(
            f"ingest chunk {chunk_number}: {metrics['input_rows']:,} input rows, "
            f"{metrics['distinct_raw_rows_retained']:,} distinct"
        )
        if failure_injector is not None:
            failure_injector("ingest_chunk")
    if not saw_rows:
        raise ValueError("products CSV has no rows")
    if metrics["input_rows"] != (
        metrics["exact_duplicate_rows_removed"] + metrics["distinct_raw_rows_retained"]
    ):
        raise RuntimeError("raw-row deduplication reconciliation failed")
    if metrics["distinct_raw_rows_retained"] != (
        metrics["accepted_product_candidates"] + metrics["product_quarantine_rows"]
    ):
        raise RuntimeError("product candidate reconciliation failed")
    if metrics["distinct_raw_rows_retained"] != (
        metrics["accepted_offer_candidates"] + metrics["offer_transform_quarantine_rows"]
    ):
        raise RuntimeError("offer transformation reconciliation failed")
    if metrics["accepted_offer_candidates"] != (
        metrics["distinct_offer_candidates"] + metrics["exact_offer_duplicates_removed"]
    ):
        raise RuntimeError("offer deduplication reconciliation failed")
    rule_counts["RAW-001"] = metrics["input_rows"]
    rule_counts["ID-001"] = metrics["distinct_raw_rows_retained"] - product_quarantine_rules[
        "ID-002"
    ]
    rule_counts["BOOL-001"] = metrics["distinct_raw_rows_retained"] - offer_quarantine_rules[
        "BOOL-002"
    ]
    return (
        dict(metrics),
        aggregate_counts,
        rule_counts,
        product_quarantine_rules,
        offer_quarantine_rules,
    )


_RANKED_PRODUCT_CTE = """
WITH marked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY product_id, core_blob ORDER BY source_row
    ) AS core_occurrence
    FROM product_candidates
), reps AS (
    SELECT * FROM marked WHERE core_occurrence = 1
), ranked AS (
    SELECT *,
        COUNT(*) OVER (PARTITION BY product_id) AS core_count,
        ROW_NUMBER() OVER (
            PARTITION BY product_id
            ORDER BY rate_count DESC, completeness DESC, core_digest ASC, source_row ASC
        ) AS selection_rank
    FROM reps
)
"""


def _canonicalize_products(
    connection: sqlite3.Connection,
) -> tuple[int, int, int]:
    connection.execute(
        _RANKED_PRODUCT_CTE
        + """
        INSERT INTO final_products
        SELECT product_id, title_fa, category1, category2, brand, rate, rate_count,
               sub_category, is_unrated, inconsistent_zero_rate,
               CAST(core_count > 1 AS INTEGER), source_row
        FROM ranked WHERE selection_rank = 1
        """
    )
    connection.execute(
        _RANKED_PRODUCT_CTE
        + """
        INSERT INTO conflicts
        SELECT ranked.product_id, ranked.title_fa, ranked.category1, ranked.category2,
               ranked.brand, ranked.rate, ranked.rate_count, ranked.sub_category,
               ranked.is_unrated, ranked.inconsistent_zero_rate, ranked.source_row,
               final_products.canonical_source_row, ranked.completeness,
               ranked.core_digest, ranked.raw_core_json,
               CAST(ranked.source_row = final_products.canonical_source_row AS INTEGER)
        FROM ranked
        JOIN final_products USING (product_id)
        WHERE ranked.core_count > 1
        """
    )
    connection.commit()
    unique_products = int(connection.execute("SELECT COUNT(*) FROM final_products").fetchone()[0])
    conflict_ids = int(
        connection.execute(
            "SELECT COUNT(*) FROM final_products WHERE core_attribute_conflict = 1"
        ).fetchone()[0]
    )
    conflict_alternatives = int(connection.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])
    return unique_products, conflict_ids, conflict_alternatives


def _quarantine_offers_without_products(
    connection: sqlite3.Connection,
    input_path: Path,
    offer_quarantine_rules: Counter[str],
    rule_counts: Counter[str],
) -> int:
    cursor = connection.execute(
        """
        SELECT * FROM offers
        WHERE NOT EXISTS (
            SELECT 1 FROM final_products WHERE final_products.product_id = offers.product_id
        )
        ORDER BY source_row
        """
    )
    omitted = 0
    for row in cursor:
        raw = json.loads(row["raw_json"])
        record = {
            "dataset": "offers",
            "source_file": str(input_path),
            "source_row_number": row["source_row"],
            "entity_id": row["product_id"],
            "rule_ids": ["OFF-009"],
            "reason": "no valid canonical product exists for product_id",
            "raw_record_json": _json_text(raw),
        }
        _store_quarantine(connection, "offers", record)
        omitted += 1
    if omitted:
        connection.execute(
            """
            DELETE FROM offers
            WHERE NOT EXISTS (
                SELECT 1 FROM final_products WHERE final_products.product_id = offers.product_id
            )
            """
        )
        offer_quarantine_rules["OFF-009"] += omitted
        rule_counts["OFF-009"] += omitted
    connection.commit()
    return omitted


def _price_threshold(connection: sqlite3.Connection) -> tuple[int, int | None, int | None]:
    population = int(
        connection.execute(
            "SELECT COUNT(*) FROM offers WHERE price_raw > 0"
        ).fetchone()[0]
    )
    if population == 0:
        return 0, None, None
    rank = (999 * population + 999) // 1000
    threshold = int(
        connection.execute(
            "SELECT price_raw FROM offers WHERE price_raw > 0 ORDER BY price_raw LIMIT 1 OFFSET ?",
            (rank - 1,),
        ).fetchone()[0]
    )
    return population, rank, threshold


def _product_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    for row in connection.execute("SELECT * FROM final_products ORDER BY product_id"):
        yield {
            "product_id": row["product_id"],
            "title_fa": row["title_fa"],
            "category1": row["category1"],
            "category2": row["category2"],
            "brand": row["brand"],
            "rate": row["rate"],
            "rate_count": row["rate_count"],
            "sub_category": row["sub_category"],
            "is_unrated": bool(row["is_unrated"]),
            "inconsistent_zero_rate": bool(row["inconsistent_zero_rate"]),
            "core_attribute_conflict": bool(row["core_attribute_conflict"]),
            "canonical_source_row_number": row["canonical_source_row"],
        }


def _offer_rows(
    connection: sqlite3.Connection, threshold: int | None
) -> Iterator[dict[str, Any]]:
    collisions = connection.execute(
        "SELECT offer_id FROM offers GROUP BY offer_id HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if collisions is not None:
        raise RuntimeError("technical offer fingerprint collision detected")
    for row in connection.execute("SELECT * FROM offers ORDER BY product_id, offer_id"):
        yield {
            "offer_id": row["offer_id"],
            "product_id": row["product_id"],
            "seller": row["seller"],
            "price_raw": row["price_raw"],
            "price_toman": Decimal(row["price_toman"]) if row["price_toman"] else None,
            "is_fake": bool(row["is_fake"]) if row["is_fake"] is not None else None,
            "min_price_last_month": row["min_price_last_month"],
            "missing_price_history": bool(row["missing_price_history"]),
            "invalid_price": bool(row["invalid_price"]),
            "high_price_review": bool(
                threshold is not None
                and row["price_raw"] is not None
                and row["price_raw"] > threshold
            ),
            "source_row_number": row["source_row"],
        }


def _conflict_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    query = "SELECT * FROM conflicts ORDER BY product_id, core_digest, candidate_source_row"
    for row in connection.execute(query):
        yield {
            "product_id": row["product_id"],
            "title_fa": row["title_fa"],
            "category1": row["category1"],
            "category2": row["category2"],
            "brand": row["brand"],
            "rate": row["rate"],
            "rate_count": row["rate_count"],
            "sub_category": row["sub_category"],
            "is_unrated": bool(row["is_unrated"]),
            "inconsistent_zero_rate": bool(row["inconsistent_zero_rate"]),
            "candidate_source_row_number": row["candidate_source_row"],
            "canonical_source_row_number": row["canonical_source_row"],
            "core_completeness": row["completeness"],
            "core_digest": row["core_digest"],
            "raw_core_json": row["raw_core_json"],
            "selected_as_canonical": bool(row["selected"]),
        }


def _json_rows(
    connection: sqlite3.Connection, table: str, where: str = ""
) -> Iterator[dict[str, Any]]:
    query = f"SELECT record_json FROM {table} {where} ORDER BY sequence"
    for row in connection.execute(query):
        yield json.loads(row["record_json"])


def _write_parquet(
    path: Path,
    schema: pa.Schema,
    rows: Iterable[dict[str, Any]],
    batch_size: int = 10_000,
) -> int:
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    count = 0
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) < batch_size:
                continue
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(path, schema, compression="snappy")
            writer.write_table(table)
            count += len(buffer)
            buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(path, schema, compression="snappy")
            writer.write_table(table)
            count += len(buffer)
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=schema), path, compression="snappy")
    finally:
        if writer is not None:
            writer.close()
    return count


def _publish(staging: Path, output_dir: Path, force: bool) -> None:
    backup: Path | None = None
    try:
        if output_dir.exists():
            if not force:
                raise FileExistsError(f"output directory already exists: {output_dir}")
            backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
            output_dir.rename(backup)
        staging.rename(output_dir)
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except Exception:
                output_dir.rename(staging)
                backup.rename(output_dir)
                raise
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise


def run_products_pipeline(
    input_products: Path | str,
    output_dir: Path | str,
    *,
    chunksize: int = 100_000,
    max_rows: int | None = None,
    temp_dir: Path | str | None = None,
    force: bool = False,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Clean products/offers with disk-backed global state and atomic publication."""
    input_path = Path(input_products).resolve()
    final_dir = Path(output_dir).resolve()
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if final_dir.exists() and not force:
        raise FileExistsError(f"output directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent)
    )
    try:
        selected_temp_dir = Path(temp_dir).resolve() if temp_dir is not None else None
        if selected_temp_dir is not None:
            selected_temp_dir.mkdir(parents=True, exist_ok=True)
        descriptor, database_name = tempfile.mkstemp(
            prefix="digikala-clean-products-",
            suffix=".sqlite3",
            dir=selected_temp_dir,
        )
        os.close(descriptor)
        database_path = Path(database_name)
    except Exception:
        shutil.rmtree(staging)
        raise
    connection: sqlite3.Connection | None = None
    published = False
    try:
        _progress("phase source checksum: start")
        source_checksum = _sha256_file(input_path)
        _progress("phase source checksum: complete")
        connection = _create_database(database_path)
        _progress("phase ingest and exact deduplication: start")
        (
            metrics,
            aggregate_counts,
            rule_counts,
            product_quarantine_rules,
            offer_quarantine_rules,
        ) = _ingest(
            connection, input_path, chunksize, max_rows, failure_injector
        )
        _progress("phase ingest and exact deduplication: complete")
        if failure_injector is not None:
            failure_injector("after_ingest")

        _progress("phase canonical products: start")
        unique_products, conflict_ids, conflict_alternatives = _canonicalize_products(connection)
        multiple_product_ids = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT product_id FROM product_candidates
                    GROUP BY product_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        rule_counts["PRD-002"] = multiple_product_ids
        rule_counts["PRD-003"] = conflict_ids
        _progress("phase canonical products: complete")
        if failure_injector is not None:
            failure_injector("after_canonical")

        omitted_offers = _quarantine_offers_without_products(
            connection, input_path, offer_quarantine_rules, rule_counts
        )
        metrics["offers_without_canonical_product"] = omitted_offers
        distinct_offers = metrics["distinct_offer_candidates"] - omitted_offers
        if metrics["distinct_offer_candidates"] != distinct_offers + omitted_offers:
            raise RuntimeError("final offer reconciliation failed")

        _progress("phase global price threshold: start")
        price_population, price_rank, threshold = _price_threshold(connection)
        high_price_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM offers WHERE price_raw > ?",
                (threshold if threshold is not None else sys.maxsize,),
            ).fetchone()[0]
        )
        offer_excess_rows = int(
            connection.execute(
                """
                SELECT COALESCE(SUM(n - 1), 0) FROM (
                    SELECT COUNT(*) AS n FROM offers GROUP BY product_id
                )
                """
            ).fetchone()[0]
        )
        rule_counts["OFF-002"] = offer_excess_rows
        rule_counts["OFF-005"] = high_price_count
        rule_counts["OFF-006"] = price_population
        _progress("phase global price threshold: complete")

        _progress("phase Parquet materialization: start")
        output_counts = {
            "products_clean.parquet": _write_parquet(
                staging / "products_clean.parquet",
                PRODUCTS_CLEAN_SCHEMA,
                _product_rows(connection),
            ),
            "offers_clean.parquet": _write_parquet(
                staging / "offers_clean.parquet",
                OFFERS_CLEAN_SCHEMA,
                _offer_rows(connection, threshold),
            ),
            "product_conflicts.parquet": _write_parquet(
                staging / "product_conflicts.parquet",
                PRODUCT_CONFLICT_SCHEMA,
                _conflict_rows(connection),
            ),
            "product_quarantine.parquet": _write_parquet(
                staging / "product_quarantine.parquet",
                QUARANTINE_SCHEMA,
                _json_rows(connection, "quarantine", "WHERE side='products'"),
            ),
            "offer_quarantine.parquet": _write_parquet(
                staging / "offer_quarantine.parquet",
                QUARANTINE_SCHEMA,
                _json_rows(connection, "quarantine", "WHERE side='offers'"),
            ),
            "row_audit.parquet": _write_parquet(
                staging / "row_audit.parquet",
                ROW_AUDIT_SCHEMA,
                _json_rows(connection, "row_audit"),
            ),
        }
        _progress("phase Parquet materialization: complete")
        if output_counts["products_clean.parquet"] != unique_products:
            raise RuntimeError("products output count does not match canonical count")
        if output_counts["offers_clean.parquet"] != distinct_offers:
            raise RuntimeError("offers output count does not match reconciled count")
        if output_counts["product_conflicts.parquet"] != conflict_alternatives:
            raise RuntimeError("conflict output count does not match alternatives")
        if output_counts["product_quarantine.parquet"] != metrics["product_quarantine_rows"]:
            raise RuntimeError("product quarantine output count does not reconcile")
        expected_offer_quarantine = metrics["offer_transform_quarantine_rows"] + omitted_offers
        if output_counts["offer_quarantine.parquet"] != expected_offer_quarantine:
            raise RuntimeError("offer quarantine output count does not reconcile")
        for filename, schema in OUTPUT_SCHEMAS.items():
            if pq.read_schema(staging / filename) != schema:
                raise RuntimeError(f"schema round-trip failed for {filename}")

        output_checksums = {
            filename: _sha256_file(staging / filename) for filename in OUTPUT_SCHEMAS
        }
        _progress("phase source immutability check: start")
        source_checksum_after = _sha256_file(input_path)
        if source_checksum_after != source_checksum:
            raise RuntimeError("source products CSV changed during the cleaning run")
        _progress("phase source immutability check: complete")
        audit = {
            "specification_version": SPECIFICATION_VERSION,
            "status": "success",
            "source": {
                "path": str(input_path),
                "sha256_before": source_checksum,
                "sha256_after": source_checksum_after,
                "unchanged": True,
            },
            "configuration": {"chunksize": chunksize, "max_rows": max_rows},
            "rows": {
                **metrics,
                "unique_product_count": unique_products,
                "conflict_product_id_count": conflict_ids,
                "conflict_alternative_row_count": conflict_alternatives,
                "distinct_offer_count": distinct_offers,
            },
            "quarantine_counts_by_rule": {
                "products": dict(sorted(product_quarantine_rules.items())),
                "offers": dict(sorted(offer_quarantine_rules.items())),
            },
            "aggregate_transformations": dict(sorted(aggregate_counts.items())),
            "rule_event_counts": dict(sorted(rule_counts.items())),
            "price_review": {
                "population": "final accepted distinct offers with valid positive price_raw",
                "valid_positive_price_population": price_population,
                "method": "nearest-rank ceil(0.999 * N), flag strictly greater",
                "rank": price_rank,
                "threshold": threshold,
                "currency_status": "IRR_inferred",
            },
            "offer_id_semantics": (
                "technical deterministic SHA-256 fingerprint of the exact raw offer tuple; "
                "not a business identifier"
            ),
            "deterministic_sort_keys": {
                "products_clean": ["product_id"],
                "offers_clean": ["product_id", "offer_id"],
                "product_conflicts": [
                    "product_id",
                    "core_digest",
                    "candidate_source_row_number",
                ],
            },
            "reconciliation": {
                "input_rows": "exact_duplicate_rows_removed + distinct_raw_rows_retained",
                "product_sides": (
                    "distinct_raw_rows_retained = accepted_product_candidates + "
                    "product_quarantine_rows"
                ),
                "offer_sides": (
                    "distinct_raw_rows_retained = accepted_offer_candidates + "
                    "offer_transform_quarantine_rows"
                ),
                "offer_dedup": (
                    "accepted_offer_candidates = distinct_offer_candidates + "
                    "exact_offer_duplicates_removed"
                ),
                "final_offers": (
                    "distinct_offer_candidates = distinct_offer_count + "
                    "offers_without_canonical_product"
                ),
                "checks_passed": {
                    "raw_dedup": metrics["input_rows"]
                    == metrics["exact_duplicate_rows_removed"]
                    + metrics["distinct_raw_rows_retained"],
                    "product_sides": metrics["distinct_raw_rows_retained"]
                    == metrics["accepted_product_candidates"]
                    + metrics["product_quarantine_rows"],
                    "offer_sides": metrics["distinct_raw_rows_retained"]
                    == metrics["accepted_offer_candidates"]
                    + metrics["offer_transform_quarantine_rows"],
                    "offer_dedup": metrics["accepted_offer_candidates"]
                    == metrics["distinct_offer_candidates"]
                    + metrics["exact_offer_duplicates_removed"],
                    "final_offers": metrics["distinct_offer_candidates"]
                    == distinct_offers + omitted_offers,
                },
            },
            "outputs": {
                filename: {
                    "rows": output_counts[filename],
                    "sha256": output_checksums[filename],
                }
                for filename in OUTPUT_SCHEMAS
            },
        }
        audit_path = staging / "cleaning_audit.json"
        audit_path.write_text(_json_text(audit) + "\n", encoding="utf-8")
        manifest = {
            "specification_version": SPECIFICATION_VERSION,
            "status": "success",
            "run_id": hashlib.sha256(
                _json_text(
                    {
                        "source_sha256": source_checksum,
                        "chunksize": chunksize,
                        "max_rows": max_rows,
                        "specification_version": SPECIFICATION_VERSION,
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "source": audit["source"],
            "configuration": audit["configuration"],
            "currency_status": "IRR_inferred",
            "offer_recency": "unknown",
            "price_review": audit["price_review"],
            "versions": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "sqlite": sqlite3.sqlite_version,
            },
            "outputs": {
                **audit["outputs"],
                "cleaning_audit.json": {
                    "sha256": _sha256_file(audit_path),
                },
            },
        }
        (staging / "run_manifest.json").write_text(
            _json_text(manifest) + "\n", encoding="utf-8"
        )
        if failure_injector is not None:
            failure_injector("before_completion")
        (staging / COMPLETION_MARKER).write_text("success\n", encoding="ascii")
        _publish(staging, final_dir, force)
        published = True
        _progress(f"complete: published {final_dir}")
        return audit
    finally:
        if connection is not None:
            connection.close()
        _remove_sqlite_files(database_path)
        if not published and staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Digikala products and seller offers with bounded memory."
    )
    parser.add_argument("input_products", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_products_pipeline(
        args.input_products,
        args.output_dir,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        temp_dir=args.temp_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
