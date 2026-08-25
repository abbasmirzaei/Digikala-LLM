"""Chunked, reproducible EDA for the Digikala product and comment datasets."""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp1256", "windows-1252", "latin1")
TEXT_COLUMN_NAMES = {"title", "body", "description"}
ID_COLUMN_NAMES = {"id", "product_id"}
REQUIRED_DISTRIBUTIONS = {"rate", "recommendation_status", "is_buyer"}
MAX_CATEGORICAL_UNIQUES = 100
PRODUCT_CORE_ATTRIBUTES = (
    "title_fa",
    "Category1",
    "Category2",
    "Brand",
    "Rate",
    "Rate_cnt",
    "sub_category",
)


class DiskUniqueIndex:
    """Exact, disk-backed uniqueness tracking with bounded Python memory."""

    def __init__(self) -> None:
        descriptor, name = tempfile.mkstemp(prefix="digikala-eda-", suffix=".sqlite3")
        os.close(descriptor)
        self.path = Path(name)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE identifiers (value TEXT PRIMARY KEY)")
        self.connection.execute("CREATE TABLE duplicated_values (value TEXT PRIMARY KEY)")

    def add_many(self, values: list[str]) -> int:
        counts = Counter(values)
        unique_values = list(counts)
        already_seen: set[str] = set()
        for start in range(0, len(unique_values), 500):
            batch = unique_values[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT value FROM identifiers WHERE value IN ({placeholders})", batch
            )
            already_seen.update(row[0] for row in rows)
        duplicated_values = already_seen | {
            value for value, count in counts.items() if count > 1
        }
        self.connection.executemany(
            "INSERT OR IGNORE INTO duplicated_values(value) VALUES (?)",
            ((value,) for value in duplicated_values),
        )
        before = self.connection.total_changes
        self.connection.executemany(
            "INSERT OR IGNORE INTO identifiers(value) VALUES (?)",
            ((value,) for value in values),
        )
        duplicate_count = len(values) - (self.connection.total_changes - before)
        self.connection.commit()
        return duplicate_count

    def duplicated_unique_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM duplicated_values").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ProductVariationIndex:
    """Disk-backed product-offer grouping for seller and fact consistency checks."""

    def __init__(self) -> None:
        descriptor, name = tempfile.mkstemp(prefix="digikala-products-", suffix=".sqlite3")
        os.close(descriptor)
        self.path = Path(name)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE offers (product_id TEXT, seller TEXT, price TEXT, core_facts TEXT)"
        )
        self.connection.execute("CREATE INDEX offers_product_id ON offers(product_id)")

    def add_chunk(self, chunk: pd.DataFrame) -> None:
        core_columns = [name for name in PRODUCT_CORE_ATTRIBUTES if name in chunk.columns]
        rows = []
        for _, row in chunk.iterrows():
            product_id = _normalise_id(row.get("id"))
            if product_id is None:
                continue
            seller = _normalise_group_value(row.get("Seller"))
            price = _normalise_group_value(row.get("Price"))
            core_facts = _fingerprint_values(row.get(name) for name in core_columns)
            rows.append((product_id, seller, price, core_facts))
        self.connection.executemany("INSERT INTO offers VALUES (?, ?, ?, ?)", rows)
        self.connection.commit()

    def summary(self, exact_duplicate_offers: int) -> dict[str, Any]:
        def varying_ids(column: str) -> int:
            row = self.connection.execute(
                f"SELECT COUNT(*) FROM ("
                f"SELECT product_id FROM offers GROUP BY product_id "
                f"HAVING COUNT(DISTINCT {column}) > 1)"
            ).fetchone()
            return int(row[0])

        return {
            "row_semantics": "products.id groups a product and may have multiple seller offers",
            "exact_duplicate_offer_count": exact_duplicate_offers,
            "product_ids_with_multiple_sellers": varying_ids("seller"),
            "product_ids_with_different_prices": varying_ids("price"),
            "product_ids_with_conflicting_core_attributes": varying_ids("core_facts"),
            "core_attributes": list(PRODUCT_CORE_ATTRIBUTES),
        }

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[EDA] {message}", flush=True)


def detect_format(path: Path) -> tuple[str, str]:
    """Return a usable (encoding, delimiter) pair from a bounded byte sample."""
    with path.open("rb") as source:
        sample = source.read(100_000)
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            # The fixed-size sample can end halfway through a multibyte character.
            # An incremental decoder validates all complete bytes without treating
            # that incomplete trailing code point as an encoding failure.
            decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
            text = decoder.decode(sample, final=False)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
            return encoding, dialect.delimiter
        except csv.Error:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            return encoding, delimiter
    raise ValueError(f"Could not decode {path}") from last_error


def _normalise_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    result = str(value).strip()
    return result or None


def _normalise_group_value(value: object) -> str:
    normalized = _normalise_id(value)
    return normalized if normalized is not None else "<MISSING>"


def _fingerprint_values(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = _normalise_group_value(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _row_fingerprints(chunk: pd.DataFrame) -> list[str]:
    """Create stable full-row fingerprints without retaining row contents."""
    fingerprints: list[str] = []
    for row in chunk.itertuples(index=False, name=None):
        digest = hashlib.sha256()
        for value in row:
            if pd.isna(value):
                digest.update(b"N")
                continue
            encoded = str(value).encode("utf-8")
            digest.update(b"V")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        fingerprints.append(digest.hexdigest())
    return fingerprints


@dataclass
class ColumnAccumulator:
    missing: int = 0
    non_missing: int = 0
    numeric_count: int = 0
    numeric_invalid: int = 0
    numeric_sum: float = 0.0
    numeric_min: float | None = None
    numeric_max: float | None = None
    values: Counter[str] | None = field(default_factory=Counter)

    def update(self, series: pd.Series, keep_distribution: bool = False) -> None:
        self.missing += int(series.isna().sum())
        present = series.dropna().astype(str).str.strip()
        self.missing += int((present == "").sum())
        present = present[present != ""]
        self.non_missing += len(present)
        numeric = pd.to_numeric(present, errors="coerce")
        valid = numeric.dropna()
        self.numeric_count += len(valid)
        self.numeric_invalid += int(numeric.isna().sum())
        if not valid.empty:
            chunk_min, chunk_max = float(valid.min()), float(valid.max())
            self.numeric_sum += float(valid.sum())
            self.numeric_min = chunk_min if self.numeric_min is None else min(self.numeric_min, chunk_min)
            self.numeric_max = chunk_max if self.numeric_max is None else max(self.numeric_max, chunk_max)
        if self.values is not None:
            self.values.update(present)
            if not keep_distribution and len(self.values) > MAX_CATEGORICAL_UNIQUES:
                self.values = None


def _distribution(counter: Counter[str] | None) -> dict[str, int]:
    return dict(counter.most_common()) if counter is not None else {}


def _infer_dataset_type(path: Path, columns: Iterable[str]) -> str:
    lowered = {column.lower() for column in columns}
    name = path.name.lower()
    if "product_id" in lowered or "comment" in name:
        return "comments"
    if "price" in lowered or "product" in name:
        return "products"
    return "generic"


def _csv_chunks(
    path: Path,
    chunksize: int,
    usecols: list[str] | None = None,
    max_rows: int | None = None,
) -> Iterable[pd.DataFrame]:
    encoding, delimiter = detect_format(path)
    return pd.read_csv(
        path,
        encoding=encoding,
        sep=delimiter,
        dtype=str,
        usecols=usecols,
        chunksize=chunksize,
        nrows=max_rows,
        keep_default_na=True,
    )


def _outside_range_count(
    path: Path,
    column: str,
    low: float,
    high: float,
    chunksize: int,
    max_rows: int | None,
) -> int:
    count = 0
    for chunk in _csv_chunks(path, chunksize, [column], max_rows):
        values = pd.to_numeric(chunk[column], errors="coerce")
        count += int(((values < low) | (values > high)).sum())
    return count


def _price_validation(path: Path, chunksize: int, max_rows: int | None) -> dict[str, int]:
    result = {"missing_count": 0, "zero_count": 0, "negative_count": 0, "non_numeric_count": 0}
    for chunk in _csv_chunks(path, chunksize, ["Price"], max_rows):
        source = chunk["Price"]
        result["missing_count"] += int(source.isna().sum())
        present = source.dropna().astype(str).str.strip()
        empty = present == ""
        result["missing_count"] += int(empty.sum())
        numeric = pd.to_numeric(present[~empty], errors="coerce")
        result["non_numeric_count"] += int(numeric.isna().sum())
        result["zero_count"] += int((numeric == 0).sum())
        result["negative_count"] += int((numeric < 0).sum())
    return result


def profile_dataset(
    path: Path,
    chunksize: int = 100_000,
    *,
    max_rows: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Profile a CSV incrementally; no DataFrame larger than ``chunksize`` is created."""
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be greater than zero")
    _progress(progress, f"{path.name}: detecting encoding and delimiter")
    encoding, delimiter = detect_format(path)
    _progress(progress, f"{path.name}: format detected ({encoding=}, {delimiter=})")
    rows = 0
    column_names: list[str] = []
    accumulators: dict[str, ColumnAccumulator] = {}
    dataset_type = "generic"
    seen_ids: set[str] = set()
    duplicate_id_excess_rows = 0
    exact_full_row_duplicates = 0
    missing_ids = 0
    product_variation_summary = None

    _progress(progress, f"{path.name}: starting chunked primary scan")
    with ExitStack() as stack:
        id_index = stack.enter_context(DiskUniqueIndex())
        row_index = stack.enter_context(DiskUniqueIndex())
        product_variations: ProductVariationIndex | None = None
        for chunk_number, chunk in enumerate(
            _csv_chunks(path, chunksize, max_rows=max_rows), start=1
        ):
            if not column_names:
                column_names = [str(name) for name in chunk.columns]
                dataset_type = _infer_dataset_type(path, column_names)
                accumulators = {name: ColumnAccumulator() for name in column_names}
                if dataset_type == "products":
                    product_variations = stack.enter_context(ProductVariationIndex())
                _progress(progress, f"{path.name}: initialized {dataset_type} accumulators")
            rows += len(chunk)
            exact_full_row_duplicates += row_index.add_many(_row_fingerprints(chunk))
            if product_variations is not None:
                product_variations.add_chunk(chunk)
            for name in column_names:
                accumulators[name].update(
                    chunk[name], keep_distribution=name.lower() in REQUIRED_DISTRIBUTIONS
                )
            if "id" in chunk.columns:
                identifiers = [_normalise_id(value) for value in chunk["id"]]
                missing_ids += sum(identifier is None for identifier in identifiers)
                present_ids = [identifier for identifier in identifiers if identifier is not None]
                duplicate_id_excess_rows += id_index.add_many(present_ids)
                if dataset_type == "products":
                    seen_ids.update(present_ids)
            if chunk_number == 1 or chunk_number % 10 == 0:
                _progress(progress, f"{path.name}: primary scan reached {rows:,} rows")
        duplicated_unique_ids = id_index.duplicated_unique_count()
        if product_variations is not None:
            _progress(progress, f"{path.name}: summarizing product offer variation")
            product_variation_summary = product_variations.summary(exact_full_row_duplicates)
    _progress(progress, f"{path.name}: primary scan complete ({rows:,} rows); temp indexes removed")
    columns: list[dict[str, Any]] = []
    numeric_statistics: dict[str, dict[str, float | None]] = {}
    top_value_counts: dict[str, dict[str, int]] = {}
    for name in column_names:
        accumulator = accumulators[name]
        numeric = accumulator.numeric_count > 0 and (
            accumulator.numeric_invalid == 0 or name.lower() in {"price", "rate"}
        )
        columns.append(
            {
                "name": name,
                "missing_count": accumulator.missing,
                "missing_percent": round(accumulator.missing / rows * 100, 2) if rows else 0.0,
            }
        )
        if numeric:
            numeric_statistics[name] = {
                "min": accumulator.numeric_min,
                "max": accumulator.numeric_max,
                "mean": accumulator.numeric_sum / accumulator.numeric_count,
            }
        suitable = (
            accumulator.values is not None
            and name.lower() not in TEXT_COLUMN_NAMES | ID_COLUMN_NAMES
            and not numeric
        )
        if suitable:
            top_value_counts[name] = dict(accumulator.values.most_common(20))

    validation: dict[str, Any] = {}
    if dataset_type in {"products", "comments"}:
        validation["id"] = {
            "missing_count": missing_ids,
            "exact_full_row_duplicate_count": exact_full_row_duplicates,
            "duplicate_id_excess_row_count": duplicate_id_excess_rows,
            "duplicated_unique_id_count": duplicated_unique_ids,
            "is_unique": duplicate_id_excess_rows == 0,
        }
    if dataset_type == "products":
        validation["product_offer_variation"] = product_variation_summary
        if "Rate" in accumulators:
            _progress(progress, f"{path.name}: starting Rate range validation")
            validation["Rate"] = {
                "distribution": _distribution(accumulators["Rate"].values),
                "outside_0_100_count": _outside_range_count(
                    path, "Rate", 0, 100, chunksize, max_rows
                ),
            }
            _progress(progress, f"{path.name}: Rate range validation complete")
        if "Price" in accumulators:
            _progress(progress, f"{path.name}: starting Price validation")
            validation["Price"] = _price_validation(path, chunksize, max_rows)
            _progress(progress, f"{path.name}: Price validation complete")
    elif dataset_type == "comments":
        if "product_id" in accumulators:
            validation["missing_product_id"] = accumulators["product_id"].missing
        if "rate" in accumulators:
            _progress(progress, f"{path.name}: starting rate range validation")
            validation["rate"] = {
                "distribution": _distribution(accumulators["rate"].values),
                "outside_0_5_count": _outside_range_count(
                    path, "rate", 0, 5, chunksize, max_rows
                ),
            }
            _progress(progress, f"{path.name}: rate range validation complete")
        for name in ("recommendation_status", "is_buyer"):
            if name in accumulators:
                validation[name] = {"distribution": _distribution(accumulators[name].values)}

    _progress(progress, f"{path.name}: profile complete")
    return {
        "source_file": str(path),
        "dataset_type": dataset_type,
        "encoding": encoding,
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "rows": rows,
        "column_names": column_names,
        "columns_count": len(column_names),
        "columns": columns,
        "numeric_statistics": numeric_statistics,
        "top_value_counts": top_value_counts,
        "validation": validation,
        "_ids": seen_ids,
    }


def _join_validation(
    comments_path: Path,
    product_ids: set[str],
    chunksize: int,
    max_rows: int | None,
    progress: bool,
) -> dict[str, Any]:
    _progress(progress, f"{comments_path.name}: starting product join validation")
    orphan_comment_count = 0
    orphan_sample: set[str] = set()
    for chunk in _csv_chunks(comments_path, chunksize, ["product_id"], max_rows):
        for value in chunk["product_id"]:
            identifier = _normalise_id(value)
            if identifier is not None and identifier not in product_ids:
                orphan_comment_count += 1
                if len(orphan_sample) < 20:
                    orphan_sample.add(identifier)
    _progress(progress, f"{comments_path.name}: product join validation complete")
    return {
        "relationship": "products.id <- comments.product_id",
        "orphan_comment_count": orphan_comment_count,
        "orphan_product_ids_sample": sorted(orphan_sample),
    }


def profile_datasets(
    paths: list[Path],
    chunksize: int = 100_000,
    *,
    max_rows: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Profile datasets and validate the intended product/comment join when both exist."""
    reports = [
        profile_dataset(path, chunksize, max_rows=max_rows, progress=progress) for path in paths
    ]
    products = next((report for report in reports if report["dataset_type"] == "products"), None)
    comments = next((report for report in reports if report["dataset_type"] == "comments"), None)
    join_validation: dict[str, Any] | None = None
    if products is not None and comments is not None:
        join_validation = _join_validation(
            Path(comments["source_file"]),
            products["_ids"],
            chunksize,
            max_rows,
            progress,
        )
    for report in reports:
        report.pop("_ids")
    return {
        "chunksize": chunksize,
        "max_rows": max_rows,
        "datasets": reports,
        "join_validation": join_validation,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Phase 1 EDA report", "", f"- Chunk size: **{report['chunksize']:,} rows**"]
    for dataset in report["datasets"]:
        lines.extend(
            [
                "",
                f"## {dataset['dataset_type'].title()}: `{dataset['source_file']}`",
                "",
                f"- Shape: **{dataset['rows']:,} rows × {dataset['columns_count']} columns**",
                f"- Encoding: `{dataset['encoding']}`",
                f"- Delimiter: `{dataset['delimiter']}`",
                "",
                "| Column | Missing | Missing % |",
                "|---|---:|---:|",
            ]
        )
        for column in dataset["columns"]:
            lines.append(
                f"| {column['name']} | {column['missing_count']:,} | {column['missing_percent']}% |"
            )
        lines.extend(
            ["", "### Numeric statistics", "", "| Column | Min | Max | Mean |", "|---|---:|---:|---:|"]
        )
        for name, stats in dataset["numeric_statistics"].items():
            lines.append(f"| {name} | {stats['min']} | {stats['max']} | {stats['mean']} |")
        lines.extend(["", "### Top categorical values", ""])
        for name, counts in dataset["top_value_counts"].items():
            values = ", ".join(f"`{key}` ({value:,})" for key, value in counts.items())
            lines.append(f"- **{name}:** {values}")
        lines.extend(
            [
                "",
                "### Dataset validation",
                "",
                "```json",
                json.dumps(dataset["validation"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    if report["join_validation"] is not None:
        lines.extend(
            [
                "",
                "## Join validation",
                "",
                "```json",
                json.dumps(report["join_validation"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_profile.json"
    markdown_path = output_dir / "dataset_profile.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a chunked Phase 1 dataset profile.")
    parser.add_argument("dataset", type=Path, nargs="+", help="One or more CSV/TSV files")
    parser.add_argument("--output", type=Path, default=Path("reports/eda"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Diagnostic row limit per dataset; omit for a complete profile",
    )
    args = parser.parse_args()
    missing = [path for path in args.dataset if not path.is_file()]
    if missing:
        parser.error(f"Dataset not found: {missing[0]}")
    if args.chunksize <= 0:
        parser.error("--chunksize must be greater than zero")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be greater than zero")
    report = profile_datasets(
        args.dataset,
        args.chunksize,
        max_rows=args.max_rows,
        progress=True,
    )
    _progress(True, "writing JSON and Markdown reports")
    json_path, markdown_path = write_report(report, args.output)
    _progress(True, "report writing complete")
    print(
        "; ".join(f"{item['dataset_type']}: {item['rows']:,} rows" for item in report["datasets"]),
        flush=True,
    )
    print(f"Reports: {json_path}, {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
