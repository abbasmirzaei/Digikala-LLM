"""Exact-token, bounded-memory CSV ingestion shared by cleaning pipelines."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path


def iter_exact_csv_batches(
    input_path: Path,
    expected_columns: Sequence[str],
    *,
    chunksize: int,
    max_rows: int | None,
    field_size_limit: int,
    dataset_name: str,
) -> Iterator[list[tuple[int, dict[str, str]]]]:
    """Yield exact decoded CSV fields with logical data-record source row numbers."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    csv.field_size_limit(field_size_limit)
    expected = list(expected_columns)
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=",", restkey=None, restval=None)
        if reader.fieldnames != expected:
            raise ValueError(
                f"{dataset_name} CSV header mismatch: expected {expected}, got {reader.fieldnames}"
            )
        batch: list[tuple[int, dict[str, str]]] = []
        records_read = 0
        while max_rows is None or records_read < max_rows:
            try:
                raw = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                raise ValueError(
                    f"malformed {dataset_name} CSV data record {records_read + 1}: {exc}"
                ) from exc
            source_row = records_read + 2
            if None in raw:
                raise ValueError(
                    f"{dataset_name} CSV data record {records_read + 1} has extra fields: "
                    f"{raw[None]!r}"
                )
            missing = [column for column in expected if raw[column] is None]
            if missing:
                raise ValueError(
                    f"{dataset_name} CSV data record {records_read + 1} has missing fields: "
                    f"{missing}"
                )
            batch.append((source_row, {column: raw[column] for column in expected}))
            records_read += 1
            if len(batch) == chunksize:
                yield batch
                batch = []
        if batch:
            yield batch
