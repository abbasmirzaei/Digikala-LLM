"""Bounded-memory dataset-level cleaning for Digikala comments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.cleaning import (
    COMMENT_CONFLICT_SCHEMA,
    COMMENTS_CLEAN_SCHEMA,
    QUARANTINE_SCHEMA,
    ROW_AUDIT_SCHEMA,
    clean_comment_text,
    transform_comment_row,
)
from digikala_llm.csv_stream import iter_exact_csv_batches
from digikala_llm.products_pipeline import (
    _exact_values,
    _json_text,
    _publish,
    _remove_sqlite_files,
    _sha256_file,
    _write_parquet,
)

SPECIFICATION_VERSION = "1.0.6"
COMPLETION_MARKER = "_SUCCESS"
COMMENT_SOURCE_COLUMNS = (
    "id",
    "title",
    "body",
    "created_at",
    "rate",
    "recommendation_status",
    "is_buyer",
    "product_id",
    "advantages",
    "disadvantages",
    "likes",
    "dislikes",
    "seller_title",
    "seller_code",
    "true_to_size_rate",
)
COMMENT_MAPPED_FIELDS = COMMENT_SOURCE_COLUMNS[1:]
# Reviews may be long; reject pathological individual fields above this explicit bound.
COMMENT_CSV_FIELD_SIZE_LIMIT = 64 * 1024 * 1024
OUTPUT_SCHEMAS = {
    "comments_clean.parquet": COMMENTS_CLEAN_SCHEMA,
    "comment_conflicts.parquet": COMMENT_CONFLICT_SCHEMA,
    "comment_quarantine.parquet": QUARANTINE_SCHEMA,
    "row_audit.parquet": ROW_AUDIT_SCHEMA,
}
COMMENT_RULE_IDS = (
    "RAW-001",
    "ID-001",
    "ID-002",
    "BOOL-001",
    "BOOL-002",
    "TXT-001",
    "TXT-002",
    *(f"COM-{number:03d}" for number in range(1, 22)),
    *(f"DATE-{number:03d}" for number in range(1, 5)),
    "JOIN-001",
    "JOIN-002",
)
ROW_METRIC_NAMES = (
    "input_rows",
    "exact_duplicate_rows_removed",
    "distinct_raw_rows_retained",
    "accepted_transformation_candidates",
    "transformation_quarantine_rows",
)
_CLEAN_CONTENT_FIELDS = (
    "product_id",
    "title",
    "body",
    "created_at_raw",
    "created_at_jalali",
    "created_at_gregorian",
    "rate",
    "is_unrated",
    "invalid_rate",
    "recommendation_status",
    "is_buyer",
    "advantages",
    "disadvantages",
    "likes",
    "dislikes",
    "seller_title",
    "seller_code",
    "true_to_size_rate",
)


def _progress(message: str) -> None:
    print(f"[clean-comments] {message}", flush=True)


def iter_comment_csv_batches(
    input_path: Path,
    chunksize: int,
    max_rows: int | None,
) -> Iterator[list[tuple[int, dict[str, str]]]]:
    """Yield exact comment records with logical CSV data-record source rows."""
    yield from iter_exact_csv_batches(
        input_path,
        COMMENT_SOURCE_COLUMNS,
        chunksize=chunksize,
        max_rows=max_rows,
        field_size_limit=COMMENT_CSV_FIELD_SIZE_LIMIT,
        dataset_name="comments",
    )


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA temp_store=FILE;
            CREATE TABLE product_ids (
                product_id INTEGER PRIMARY KEY
            );
            CREATE TABLE raw_rows (
                content BLOB PRIMARY KEY,
                digest TEXT NOT NULL,
                source_row INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE comment_candidates (
                source_row INTEGER PRIMARY KEY,
                comment_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                title TEXT,
                body TEXT,
                created_at_raw TEXT NOT NULL,
                created_at_jalali TEXT NOT NULL,
                created_at_gregorian TEXT NOT NULL,
                rate TEXT,
                is_unrated INTEGER NOT NULL,
                invalid_rate INTEGER NOT NULL,
                recommendation_status TEXT,
                is_buyer INTEGER,
                advantages TEXT,
                disadvantages TEXT,
                likes INTEGER,
                dislikes INTEGER,
                seller_title TEXT,
                seller_code TEXT,
                true_to_size_rate TEXT,
                completeness INTEGER NOT NULL,
                canonical_blob BLOB NOT NULL,
                canonical_digest TEXT NOT NULL,
                clean_blob BLOB NOT NULL,
                raw_json TEXT NOT NULL,
                raw_mapped_json TEXT NOT NULL
            );
            CREATE INDEX comment_candidates_grouping
                ON comment_candidates(comment_id, completeness, canonical_digest, source_row);
            CREATE TABLE final_comments (
                comment_id INTEGER PRIMARY KEY,
                product_id INTEGER NOT NULL,
                title TEXT,
                body TEXT,
                created_at_raw TEXT NOT NULL,
                created_at_jalali TEXT NOT NULL,
                created_at_gregorian TEXT NOT NULL,
                rate TEXT,
                is_unrated INTEGER NOT NULL,
                invalid_rate INTEGER NOT NULL,
                recommendation_status TEXT,
                is_buyer INTEGER,
                advantages TEXT,
                disadvantages TEXT,
                likes INTEGER,
                dislikes INTEGER,
                seller_title TEXT,
                seller_code TEXT,
                true_to_size_rate TEXT,
                comment_id_conflict INTEGER NOT NULL,
                canonical_source_row INTEGER NOT NULL,
                raw_json TEXT NOT NULL
            );
            CREATE TABLE quarantine (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
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


def _with_source_file(records: Iterable[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    adjusted = []
    for record in records:
        copy = dict(record)
        copy["source_file"] = str(source)
        adjusted.append(copy)
    return adjusted


def _store_audits(connection: sqlite3.Connection, records: Iterable[dict[str, Any]]) -> None:
    connection.executemany(
        "INSERT INTO row_audit(record_json) VALUES (?)",
        ((_json_text(record),) for record in records),
    )


def _store_quarantine(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO quarantine(record_json) VALUES (?)",
        (_json_text(record),),
    )


def _count_result_events(
    result: Any,
    aggregate_counts: Counter[str],
    rule_counts: Counter[str],
) -> None:
    for rule_id, field in result.aggregate_counter_keys:
        aggregate_counts[f"comments:{rule_id}:{field}"] += 1
        rule_counts[rule_id] += 1
    for record in result.audit_records:
        rule_counts[record["rule_id"]] += 1


def _mapped_values(raw: dict[str, object]) -> tuple[object, ...]:
    """Mapped source values after exact field-specific missing-value policy."""
    return tuple(clean_comment_text(raw.get(field), field) for field in COMMENT_MAPPED_FIELDS)


def _clean_values(candidate: dict[str, Any]) -> tuple[object, ...]:
    values = []
    for field in _CLEAN_CONTENT_FIELDS:
        value = candidate.get(field)
        values.append(value.isoformat() if hasattr(value, "isoformat") else value)
    return tuple(values)


def _canonical_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _insert_candidate(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
    raw: dict[str, object],
) -> None:
    mapped_values = _mapped_values(raw)
    canonical_blob = _exact_values(mapped_values)
    clean_blob = _exact_values(_clean_values(candidate))
    parsed_date = candidate["created_at_gregorian"]
    connection.execute(
        """
        INSERT INTO comment_candidates VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        (
            candidate["source_row_number"],
            candidate["comment_id"],
            candidate["product_id"],
            candidate["title"],
            candidate["body"],
            candidate["created_at_raw"],
            candidate["created_at_jalali"],
            parsed_date.isoformat(),
            str(candidate["rate"]) if candidate["rate"] is not None else None,
            int(candidate["is_unrated"]),
            int(candidate["invalid_rate"]),
            candidate["recommendation_status"],
            int(candidate["is_buyer"]) if candidate["is_buyer"] is not None else None,
            candidate["advantages"],
            candidate["disadvantages"],
            candidate["likes"],
            candidate["dislikes"],
            candidate["seller_title"],
            candidate["seller_code"],
            candidate["true_to_size_rate"],
            sum(value is not None for value in mapped_values),
            canonical_blob,
            _canonical_digest(canonical_blob),
            clean_blob,
            _json_text(raw),
            _json_text(dict(zip(COMMENT_MAPPED_FIELDS, mapped_values, strict=True))),
        ),
    )


def _load_product_ids(connection: sqlite3.Connection, products_path: Path) -> int:
    schema = pq.read_schema(products_path)
    field = schema.field("product_id")
    if field.type != pa.int64() or field.nullable:
        raise ValueError("products_clean product_id must be non-null signed int64")
    count = 0
    parquet = pq.ParquetFile(products_path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=["product_id"]):
        values = batch.column(0).to_pylist()
        for value in values:
            if value is None:
                raise ValueError("products_clean contains a null product_id")
            cursor = connection.execute("INSERT OR IGNORE INTO product_ids VALUES (?)", (value,))
            if cursor.rowcount != 1:
                raise ValueError(f"products_clean contains duplicate product_id {value}")
            count += 1
        connection.commit()
    if count == 0:
        raise ValueError("products_clean contains no product IDs")
    return count


def _ingest(
    connection: sqlite3.Connection,
    input_path: Path,
    chunksize: int,
    max_rows: int | None,
    failure_injector: Callable[[str], None] | None,
) -> tuple[dict[str, int], Counter[str], Counter[str], Counter[str]]:
    metrics = Counter({name: 0 for name in ROW_METRIC_NAMES})
    aggregate_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter({rule_id: 0 for rule_id in COMMENT_RULE_IDS})
    quarantine_rules: Counter[str] = Counter()
    saw_rows = False
    columns = list(COMMENT_SOURCE_COLUMNS)
    for chunk_number, batch in enumerate(
        iter_comment_csv_batches(input_path, chunksize, max_rows), start=1
    ):
        saw_rows = True
        for source_row, raw in batch:
            content = _exact_values(raw[column] for column in columns)
            cursor = connection.execute(
                "INSERT OR IGNORE INTO raw_rows VALUES (?, ?, ?)",
                (content, hashlib.sha256(content).hexdigest(), source_row),
            )
            if cursor.rowcount == 0:
                metrics["exact_duplicate_rows_removed"] += 1
                rule_counts["COM-001"] += 1
                continue
            metrics["distinct_raw_rows_retained"] += 1
            result = transform_comment_row(raw, source_row)
            _count_result_events(result, aggregate_counts, rule_counts)
            _store_audits(connection, _with_source_file(result.audit_records, input_path))
            if result.quarantined:
                metrics["transformation_quarantine_rows"] += 1
                assert result.quarantine_record is not None
                record = dict(result.quarantine_record)
                record["source_file"] = str(input_path)
                _store_quarantine(connection, record)
                for rule_id in record["rule_ids"]:
                    quarantine_rules[rule_id] += 1
            else:
                metrics["accepted_transformation_candidates"] += 1
                assert result.candidate_row is not None
                _insert_candidate(connection, result.candidate_row, raw)
                rule_counts["COM-008"] += 1
                rule_counts["DATE-001"] += 1
                rule_counts["DATE-002"] += 1
                rule_counts["DATE-004"] += 1
                if result.candidate_row["rate"] is not None:
                    rule_counts["COM-004"] += 1
                if raw["rate"].strip() == "2500":
                    rule_counts["COM-007"] += 1
        metrics["input_rows"] += len(batch)
        connection.commit()
        _progress(
            f"ingest chunk {chunk_number}: {metrics['input_rows']:,} input rows, "
            f"{metrics['distinct_raw_rows_retained']:,} distinct"
        )
        if failure_injector is not None:
            failure_injector("ingest_chunk")
    if not saw_rows:
        raise ValueError("comments CSV has no rows")
    if metrics["input_rows"] != (
        metrics["exact_duplicate_rows_removed"] + metrics["distinct_raw_rows_retained"]
    ):
        raise RuntimeError("raw-row deduplication reconciliation failed")
    if metrics["distinct_raw_rows_retained"] != (
        metrics["accepted_transformation_candidates"]
        + metrics["transformation_quarantine_rows"]
    ):
        raise RuntimeError("comment transformation reconciliation failed")
    rule_counts["RAW-001"] = metrics["input_rows"]
    rule_counts["ID-001"] = metrics["distinct_raw_rows_retained"] - quarantine_rules["ID-002"]
    rule_counts["BOOL-001"] = (
        metrics["distinct_raw_rows_retained"] - quarantine_rules["BOOL-002"]
    )
    return dict(metrics), aggregate_counts, rule_counts, quarantine_rules


_RANKED_COMMENT_CTE = """
WITH clean_ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY comment_id, clean_blob
        ORDER BY completeness DESC, canonical_digest ASC, source_row ASC
    ) AS clean_rank
    FROM comment_candidates
), reps AS (
    SELECT * FROM clean_ranked WHERE clean_rank = 1
), ranked AS (
    SELECT *,
        COUNT(*) OVER (PARTITION BY comment_id) AS clean_count,
        ROW_NUMBER() OVER (
            PARTITION BY comment_id
            ORDER BY completeness DESC, canonical_digest ASC, source_row ASC
        ) AS selection_rank
    FROM reps
)
"""


def _canonicalize(connection: sqlite3.Connection) -> dict[str, int]:
    repeated_ids, repeated_excess = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(n - 1), 0) FROM (
            SELECT COUNT(*) AS n FROM comment_candidates GROUP BY comment_id HAVING n > 1
        )
        """
    ).fetchone()
    clean_representatives = int(
        connection.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM comment_candidates GROUP BY comment_id, clean_blob)"
        ).fetchone()[0]
    )
    accepted = int(connection.execute("SELECT COUNT(*) FROM comment_candidates").fetchone()[0])
    identical_removed = accepted - clean_representatives
    identical_ids = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT comment_id FROM comment_candidates
                GROUP BY comment_id HAVING COUNT(*) > COUNT(DISTINCT clean_blob)
            )
            """
        ).fetchone()[0]
    )
    connection.execute(
        _RANKED_COMMENT_CTE
        + """
        INSERT INTO final_comments
        SELECT comment_id, product_id, title, body, created_at_raw, created_at_jalali,
               created_at_gregorian, rate, is_unrated, invalid_rate,
               recommendation_status, is_buyer, advantages, disadvantages, likes,
               dislikes, seller_title, seller_code, true_to_size_rate,
               CAST(clean_count > 1 AS INTEGER), source_row, raw_json
        FROM ranked WHERE selection_rank = 1
        """
    )
    connection.commit()
    canonical = int(connection.execute("SELECT COUNT(*) FROM final_comments").fetchone()[0])
    conflicts = int(
        connection.execute(
            "SELECT COUNT(*) FROM final_comments WHERE comment_id_conflict = 1"
        ).fetchone()[0]
    )
    alternatives = int(
        connection.execute(
            _RANKED_COMMENT_CTE + "SELECT COUNT(*) FROM ranked WHERE clean_count > 1"
        ).fetchone()[0]
    )
    return {
        "repeated_comment_id_count": int(repeated_ids),
        "repeated_comment_id_excess_rows": int(repeated_excess),
        "identical_clean_duplicate_id_count": identical_ids,
        "identical_clean_duplicate_id_rows_removed": identical_removed,
        "conflicting_comment_id_count": conflicts,
        "conflict_alternative_count": alternatives,
        "canonical_comment_count_before_fk": canonical,
    }


def _quarantine_orphans(
    connection: sqlite3.Connection,
    source: Path,
    quarantine_rules: Counter[str],
    rule_counts: Counter[str],
) -> tuple[int, int, list[int]]:
    rows = connection.execute(
        """
        SELECT f.* FROM final_comments f
        LEFT JOIN product_ids p USING (product_id)
        WHERE p.product_id IS NULL
        ORDER BY f.product_id, f.comment_id
        """
    ).fetchall()
    product_ids = sorted({int(row["product_id"]) for row in rows})
    for row in rows:
        record = {
            "dataset": "comments",
            "source_file": str(source),
            "source_row_number": row["canonical_source_row"],
            "entity_id": row["comment_id"],
            "rule_ids": ["JOIN-002"],
            "reason": "canonical comment product_id is absent from products_clean",
            "raw_record_json": row["raw_json"],
        }
        _store_quarantine(connection, record)
    connection.commit()
    quarantine_rules["JOIN-002"] += len(rows)
    rule_counts["JOIN-002"] += len(rows)
    return len(rows), len(product_ids), product_ids[:20]


def _comment_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    query = """
        SELECT f.* FROM final_comments f JOIN product_ids p USING (product_id)
        ORDER BY f.comment_id
    """
    for row in connection.execute(query):
        yield {
            "comment_id": row["comment_id"],
            "product_id": row["product_id"],
            "title": row["title"],
            "body": row["body"],
            "created_at_raw": row["created_at_raw"],
            "created_at_jalali": row["created_at_jalali"],
            "created_at_gregorian": date.fromisoformat(row["created_at_gregorian"]),
            "rate": Decimal(row["rate"]) if row["rate"] is not None else None,
            "is_unrated": bool(row["is_unrated"]),
            "invalid_rate": bool(row["invalid_rate"]),
            "recommendation_status": row["recommendation_status"],
            "is_buyer": bool(row["is_buyer"]) if row["is_buyer"] is not None else None,
            "advantages": row["advantages"],
            "disadvantages": row["disadvantages"],
            "likes": row["likes"],
            "dislikes": row["dislikes"],
            "seller_title": row["seller_title"],
            "seller_code": row["seller_code"],
            "true_to_size_rate": row["true_to_size_rate"],
            "comment_id_conflict": bool(row["comment_id_conflict"]),
            "canonical_source_row_number": row["canonical_source_row"],
        }


def _conflict_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    query = (
        _RANKED_COMMENT_CTE
        + """
        SELECT ranked.*, final_comments.canonical_source_row
        FROM ranked JOIN final_comments USING (comment_id)
        WHERE ranked.clean_count > 1
        ORDER BY ranked.comment_id, ranked.canonical_digest, ranked.source_row
        """
    )
    for row in connection.execute(query):
        yield {
            "comment_id": row["comment_id"],
            "product_id": row["product_id"],
            "title": row["title"],
            "body": row["body"],
            "created_at_raw": row["created_at_raw"],
            "created_at_jalali": row["created_at_jalali"],
            "created_at_gregorian": date.fromisoformat(row["created_at_gregorian"]),
            "rate": Decimal(row["rate"]) if row["rate"] is not None else None,
            "is_unrated": bool(row["is_unrated"]),
            "invalid_rate": bool(row["invalid_rate"]),
            "recommendation_status": row["recommendation_status"],
            "is_buyer": bool(row["is_buyer"]) if row["is_buyer"] is not None else None,
            "advantages": row["advantages"],
            "disadvantages": row["disadvantages"],
            "likes": row["likes"],
            "dislikes": row["dislikes"],
            "seller_title": row["seller_title"],
            "seller_code": row["seller_code"],
            "true_to_size_rate": row["true_to_size_rate"],
            "candidate_source_row_number": row["source_row"],
            "canonical_source_row_number": row["canonical_source_row"],
            "completeness": row["completeness"],
            "canonical_digest": row["canonical_digest"],
            "raw_mapped_json": row["raw_mapped_json"],
            "selected_as_canonical": row["source_row"] == row["canonical_source_row"],
        }


def _json_rows(connection: sqlite3.Connection, table: str) -> Iterator[dict[str, Any]]:
    for row in connection.execute(f"SELECT record_json FROM {table} ORDER BY sequence"):
        yield json.loads(row["record_json"])


def _date_range(connection: sqlite3.Connection) -> dict[str, str | None]:
    row = connection.execute(
        """
        SELECT MIN(created_at_jalali), MAX(created_at_jalali),
               MIN(created_at_gregorian), MAX(created_at_gregorian)
        FROM final_comments f JOIN product_ids p USING (product_id)
        """
    ).fetchone()
    return {
        "jalali_min": row[0],
        "jalali_max": row[1],
        "gregorian_min": row[2],
        "gregorian_max": row[3],
    }


def run_comments_pipeline(
    comments_csv: Path | str,
    products_clean: Path | str,
    output_dir: Path | str,
    *,
    chunksize: int = 100_000,
    max_rows: int | None = None,
    temp_dir: Path | str | None = None,
    force: bool = False,
    failure_injector: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Clean comments with exact disk-backed state and atomic publication."""
    comments_path = Path(comments_csv).resolve()
    products_path = Path(products_clean).resolve()
    final_dir = Path(output_dir).resolve()
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not comments_path.is_file():
        raise FileNotFoundError(comments_path)
    if not products_path.is_file():
        raise FileNotFoundError(products_path)
    if final_dir.exists() and not force:
        raise FileExistsError(f"output directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent)
    )
    try:
        selected_temp = Path(temp_dir).resolve() if temp_dir is not None else None
        if selected_temp is not None:
            selected_temp.mkdir(parents=True, exist_ok=True)
        descriptor, database_name = tempfile.mkstemp(
            prefix="digikala-clean-comments-", suffix=".sqlite3", dir=selected_temp
        )
        os.close(descriptor)
        database_path = Path(database_name)
    except Exception:
        shutil.rmtree(staging)
        raise
    connection: sqlite3.Connection | None = None
    published = False
    try:
        _progress("phase source checksums: start")
        comments_checksum = _sha256_file(comments_path)
        products_checksum = _sha256_file(products_path)
        _progress("phase source checksums: complete")
        connection = _create_database(database_path)
        _progress("phase product ID index: start")
        product_id_count = _load_product_ids(connection, products_path)
        _progress(f"phase product ID index: complete ({product_id_count:,} IDs)")
        if failure_injector is not None:
            failure_injector("after_product_index")
        _progress("phase ingest and exact deduplication: start")
        metrics, aggregate_counts, rule_counts, quarantine_rules = _ingest(
            connection, comments_path, chunksize, max_rows, failure_injector
        )
        _progress("phase ingest and exact deduplication: complete")
        if failure_injector is not None:
            failure_injector("after_ingest")
        _progress("phase comment canonicalization: start")
        canonical_metrics = _canonicalize(connection)
        rule_counts["COM-002"] = canonical_metrics["conflicting_comment_id_count"]
        _progress("phase comment canonicalization: complete")
        if failure_injector is not None:
            failure_injector("after_canonical")
        _progress("phase referential integrity: start")
        orphan_rows, orphan_ids, orphan_sample = _quarantine_orphans(
            connection, comments_path, quarantine_rules, rule_counts
        )
        final_count = canonical_metrics["canonical_comment_count_before_fk"] - orphan_rows
        rule_counts["JOIN-001"] = final_count
        _progress("phase referential integrity: complete")

        reconciliation_checks = {
            "raw_dedup": metrics["input_rows"]
            == metrics["exact_duplicate_rows_removed"]
            + metrics["distinct_raw_rows_retained"],
            "transformations": metrics["distinct_raw_rows_retained"]
            == metrics["accepted_transformation_candidates"]
            + metrics["transformation_quarantine_rows"],
            "canonicalization": metrics["accepted_transformation_candidates"]
            == canonical_metrics["identical_clean_duplicate_id_rows_removed"]
            + (
                canonical_metrics["conflict_alternative_count"]
                - canonical_metrics["conflicting_comment_id_count"]
            )
            + canonical_metrics["canonical_comment_count_before_fk"],
            "foreign_keys": canonical_metrics["canonical_comment_count_before_fk"]
            == final_count + orphan_rows,
        }
        if not all(reconciliation_checks.values()):
            raise RuntimeError("comment reconciliation failed")

        _progress("phase Parquet materialization: start")
        output_counts = {
            "comments_clean.parquet": _write_parquet(
                staging / "comments_clean.parquet", COMMENTS_CLEAN_SCHEMA, _comment_rows(connection)
            ),
            "comment_conflicts.parquet": _write_parquet(
                staging / "comment_conflicts.parquet",
                COMMENT_CONFLICT_SCHEMA,
                _conflict_rows(connection),
            ),
            "comment_quarantine.parquet": _write_parquet(
                staging / "comment_quarantine.parquet",
                QUARANTINE_SCHEMA,
                _json_rows(connection, "quarantine"),
            ),
            "row_audit.parquet": _write_parquet(
                staging / "row_audit.parquet", ROW_AUDIT_SCHEMA, _json_rows(connection, "row_audit")
            ),
        }
        _progress("phase Parquet materialization: complete")
        expected_quarantine = metrics["transformation_quarantine_rows"] + orphan_rows
        expected_counts = {
            "comments_clean.parquet": final_count,
            "comment_conflicts.parquet": canonical_metrics["conflict_alternative_count"],
            "comment_quarantine.parquet": expected_quarantine,
        }
        for filename, expected in expected_counts.items():
            if output_counts[filename] != expected:
                raise RuntimeError(f"{filename} row count does not reconcile")
        for filename, schema in OUTPUT_SCHEMAS.items():
            if pq.read_schema(staging / filename) != schema:
                raise RuntimeError(f"schema round-trip failed for {filename}")

        comments_after = _sha256_file(comments_path)
        products_after = _sha256_file(products_path)
        if comments_after != comments_checksum or products_after != products_checksum:
            raise RuntimeError("an input changed during comments cleaning")
        output_checksums = {
            filename: _sha256_file(staging / filename) for filename in OUTPUT_SCHEMAS
        }
        row_audit_counts = Counter(
            row["rule_id"] for row in _json_rows(connection, "row_audit")
        )
        date_range = _date_range(connection)
        audit = {
            "specification_version": SPECIFICATION_VERSION,
            "status": "success",
            "sources": {
                "comments_csv": {
                    "path": str(comments_path),
                    "sha256_before": comments_checksum,
                    "sha256_after": comments_after,
                    "unchanged": True,
                },
                "products_clean": {
                    "path": str(products_path),
                    "sha256_before": products_checksum,
                    "sha256_after": products_after,
                    "unchanged": True,
                    "product_id_count": product_id_count,
                },
            },
            "configuration": {"chunksize": chunksize, "max_rows": max_rows},
            "rows": {
                **metrics,
                **canonical_metrics,
                "orphan_comment_rows": orphan_rows,
                "distinct_orphan_product_ids": orphan_ids,
                "final_comments_clean_count": final_count,
            },
            "quarantine_counts_by_rule": dict(sorted(quarantine_rules.items())),
            "aggregate_transformations": dict(sorted(aggregate_counts.items())),
            "rule_event_counts": dict(sorted(rule_counts.items())),
            "row_audit_counts_by_rule": dict(sorted(row_audit_counts.items())),
            "field_quality_counts": {
                "missing_or_blank_title": aggregate_counts["comments:TXT-001:title"],
                "missing_or_blank_body": aggregate_counts["comments:COM-009:body"],
                "missing_or_blank_recommendation_status": aggregate_counts[
                    "comments:TXT-001:recommendation_status"
                ],
                "seller_code_missing": aggregate_counts["comments:COM-011:seller_code"],
                "seller_code_sentinel_to_null": aggregate_counts[
                    "comments:COM-013:seller_code"
                ],
                "seller_title_sentinel_to_null": aggregate_counts[
                    "comments:COM-014:seller_title"
                ],
                **{
                    f"{field}_sentinel_to_null": aggregate_counts[
                        f"comments:{rule_id}:{field}"
                    ]
                    for field, rule_id in (
                        ("title", "COM-015"),
                        ("body", "COM-016"),
                        ("recommendation_status", "COM-017"),
                        ("advantages", "COM-018"),
                        ("disadvantages", "COM-019"),
                        ("true_to_size_rate", "COM-020"),
                    )
                },
                "zero_ratings": aggregate_counts["comments:COM-005:rate"],
                "invalid_ratings": row_audit_counts["COM-006"] + row_audit_counts["COM-021"],
                "over_scale_ratings": row_audit_counts["COM-021"],
            },
            "orphan_product_id_sample": orphan_sample,
            "accepted_date_range": date_range,
            "canonical_digest": {
                "algorithm": "SHA-256 over collision-safe length-prefixed mapped values",
                "fields_in_source_order": list(COMMENT_MAPPED_FIELDS),
                "blank_policy": (
                    "blank and whitespace-only values, plus exact lowercase nan in the "
                    "eight confirmed optional fields, map to null; otherwise exact"
                ),
            },
            "deterministic_sort_keys": {
                "comments_clean": ["comment_id"],
                "comment_conflicts": [
                    "comment_id",
                    "canonical_digest",
                    "candidate_source_row_number",
                ],
            },
            "reconciliation": {
                "input_rows": "exact_duplicate_rows_removed + distinct_raw_rows_retained",
                "distinct_raw_rows": (
                    "accepted_transformation_candidates + transformation_quarantine_rows"
                ),
                "accepted_candidates": (
                    "identical_clean_duplicate_id_rows_removed + conflict_noncanonical_rows + "
                    "canonical_comment_count_before_fk"
                ),
                "canonical_comments": "final_comments_clean_count + orphan_comment_rows",
                "checks_passed": reconciliation_checks,
            },
            "outputs": {
                filename: {"rows": output_counts[filename], "sha256": output_checksums[filename]}
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
                        "comments_sha256": comments_checksum,
                        "products_sha256": products_checksum,
                        "chunksize": chunksize,
                        "max_rows": max_rows,
                        "specification_version": SPECIFICATION_VERSION,
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "sources": audit["sources"],
            "configuration": audit["configuration"],
            "canonical_digest": audit["canonical_digest"],
            "deterministic_sort_keys": audit["deterministic_sort_keys"],
            "versions": {
                "python": sys.version.split()[0],
                "pandas": importlib.metadata.version("pandas"),
                "pyarrow": pa.__version__,
                "sqlite": sqlite3.sqlite_version,
            },
            "outputs": {
                **audit["outputs"],
                "cleaning_audit.json": {"sha256": _sha256_file(audit_path)},
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
        description="Clean Digikala comments with bounded memory and product referential integrity."
    )
    parser.add_argument("comments_csv", type=Path)
    parser.add_argument("--products-clean", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_comments_pipeline(
        args.comments_csv,
        args.products_clean,
        args.output_dir,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        temp_dir=args.temp_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
