import csv
import hashlib
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from digikala_llm.cleaning import (
    COMMENT_CONFLICT_SCHEMA,
    COMMENT_RATE_TYPE,
    COMMENTS_CLEAN_SCHEMA,
    PRODUCTS_CLEAN_SCHEMA,
    QUARANTINE_SCHEMA,
    ROW_AUDIT_SCHEMA,
)
from digikala_llm.comments_pipeline import (
    COMMENT_SOURCE_COLUMNS,
    COMPLETION_MARKER,
    build_parser,
    iter_comment_csv_batches,
    run_comments_pipeline,
)


def raw_comment(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "1",
        "title": "عنوان",
        "body": "متن نظر",
        "created_at": "23 تیر 1395",
        "rate": "4",
        "recommendation_status": "recommended",
        "is_buyer": "True",
        "product_id": "1",
        "advantages": "خوب",
        "disadvantages": "",
        "likes": "2",
        "dislikes": "0",
        "seller_title": "فروشنده",
        "seller_code": "001A",
        "true_to_size_rate": "",
    }
    row.update(updates)
    return row


def write_comments(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=COMMENT_SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_products(path: Path, product_ids: list[int]) -> None:
    rows = [
        {
            "product_id": product_id,
            "title_fa": f"محصول {product_id}",
            "category1": None,
            "category2": None,
            "brand": None,
            "rate": None,
            "rate_count": 0,
            "sub_category": None,
            "is_unrated": True,
            "inconsistent_zero_rate": False,
            "core_attribute_conflict": False,
            "canonical_source_row_number": product_id + 1,
        }
        for product_id in product_ids
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=PRODUCTS_CLEAN_SCHEMA), path)


def read_rows(output: Path, filename: str) -> list[dict[str, object]]:
    return pq.read_table(output / filename).to_pylist()


def integration_rows() -> list[dict[str, object]]:
    exact = raw_comment(id="1")
    return [
        exact,
        dict(exact),
        raw_comment(id="2", title=""),
        raw_comment(id="2", title=" "),
        raw_comment(id="3", title=""),
        raw_comment(id="3", title="کامل"),
        raw_comment(id="4", title="الف"),
        raw_comment(id="4", title="ب"),
        raw_comment(id="5", rate="0"),
        raw_comment(id="6", rate="2500"),
        raw_comment(
            id="7", title="", body="", advantages="", disadvantages=""
        ),
        raw_comment(id="8", seller_code="nan", seller_title="nan"),
        raw_comment(id="9", seller_code="NAN", seller_title="NAN"),
        raw_comment(id="10", likes="not-an-integer"),
        raw_comment(id="11", is_buyer="yes"),
        raw_comment(id="bad"),
        raw_comment(id="12", product_id="99"),
        raw_comment(id="13", product_id="1", title="نسخه یک"),
        raw_comment(id="13", product_id="2", title="نسخه دو"),
    ]


def test_comments_pipeline_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    write_comments(source, integration_rows())
    write_products(products, [1, 2])

    audit = run_comments_pipeline(source, products, output, chunksize=3)
    clean = read_rows(output, "comments_clean.parquet")
    conflicts = read_rows(output, "comment_conflicts.parquet")
    quarantine = read_rows(output, "comment_quarantine.parquet")
    row_audit = read_rows(output, "row_audit.parquet")

    assert audit["specification_version"] == "1.0.6"
    assert (output / COMPLETION_MARKER).read_text() == "success\n"
    assert audit["rows"]["input_rows"] == 19
    assert audit["rows"]["exact_duplicate_rows_removed"] == 1
    assert audit["rows"]["distinct_raw_rows_retained"] == 18
    assert audit["rows"]["accepted_transformation_candidates"] == 16
    assert audit["rows"]["transformation_quarantine_rows"] == 2
    assert audit["rows"]["identical_clean_duplicate_id_rows_removed"] == 1
    assert audit["rows"]["conflicting_comment_id_count"] == 3
    assert audit["rows"]["conflict_alternative_count"] == 6
    assert audit["rows"]["canonical_comment_count_before_fk"] == 12
    assert audit["rows"]["orphan_comment_rows"] == 1
    assert audit["rows"]["final_comments_clean_count"] == 11
    assert all(audit["reconciliation"]["checks_passed"].values())
    assert [row["comment_id"] for row in clean] == sorted(row["comment_id"] for row in clean)
    assert {row["entity_id"] for row in quarantine} == {11, 12, None}
    assert audit["orphan_product_id_sample"] == [99]
    assert audit["rule_event_counts"]["JOIN-002"] == 1

    selected_three = next(row for row in clean if row["comment_id"] == 3)
    assert selected_three["title"] == "کامل"
    comment_four = [row for row in conflicts if row["comment_id"] == 4]
    assert next(row for row in comment_four if row["selected_as_canonical"])[
        "canonical_digest"
    ] == min(row["canonical_digest"] for row in comment_four)
    assert {row["product_id"] for row in conflicts if row["comment_id"] == 13} == {1, 2}

    zero = next(row for row in clean if row["comment_id"] == 5)
    assert zero["rate"] is None and zero["is_unrated"] and not zero["invalid_rate"]
    invalid = next(row for row in clean if row["comment_id"] == 6)
    assert invalid["rate"] is None and invalid["invalid_rate"]
    assert any(row["rule_id"] == "COM-006" and row["raw_value"] == "2500" for row in row_audit)
    missing_text = next(row for row in clean if row["comment_id"] == 7)
    assert all(missing_text[field] is None for field in ("title", "body", "advantages", "disadvantages"))
    sentinel = next(row for row in clean if row["comment_id"] == 8)
    assert sentinel["seller_code"] is None and sentinel["seller_title"] is None
    uppercase = next(row for row in clean if row["comment_id"] == 9)
    assert uppercase["seller_code"] == "NAN" and uppercase["seller_title"] == "NAN"
    invalid_likes = next(row for row in clean if row["comment_id"] == 10)
    assert invalid_likes["likes"] is None
    assert invalid_likes["created_at_gregorian"].isoformat() == "2016-07-13"
    assert audit["field_quality_counts"]["seller_code_sentinel_to_null"] == 1
    assert audit["field_quality_counts"]["invalid_ratings"] == 1
    assert audit["accepted_date_range"] == {
        "jalali_min": "1395-04-23",
        "jalali_max": "1395-04-23",
        "gregorian_min": "2016-07-13",
        "gregorian_max": "2016-07-13",
    }


def test_comment_parser_preserves_csv_features_and_logical_rows(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    rows = [
        raw_comment(body='comma, quote "value" and\nembedded newline', title="", seller_code="nan"),
        raw_comment(id="2", body="متن", seller_code="NAN", recommendation_status="NA"),
    ]
    write_comments(source, rows)
    batches = list(iter_comment_csv_batches(source, chunksize=1, max_rows=None))
    assert [number for batch in batches for number, _ in batch] == [2, 3]
    assert [row for batch in batches for _, row in batch] == rows


def test_comment_parser_exact_header_and_field_count_validation(tmp_path: Path) -> None:
    wrong_header = tmp_path / "header.csv"
    with wrong_header.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(list(reversed(COMMENT_SOURCE_COLUMNS)))
    with pytest.raises(ValueError, match="header mismatch"):
        list(iter_comment_csv_batches(wrong_header, 2, None))

    for delta, message in ((-1, "missing fields"), (1, "extra fields")):
        malformed = tmp_path / f"fields-{delta}.csv"
        with malformed.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(COMMENT_SOURCE_COLUMNS)
            writer.writerow(["x"] * (len(COMMENT_SOURCE_COLUMNS) + delta))
        with pytest.raises(ValueError, match=message):
            list(iter_comment_csv_batches(malformed, 2, None))


def test_comment_parser_long_text_chunks_and_exact_max_rows(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    rows = [raw_comment(id=str(number), body="x" * 200_000) for number in range(1, 7)]
    write_comments(source, rows)
    batches = list(iter_comment_csv_batches(source, chunksize=2, max_rows=5))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [number for batch in batches for number, _ in batch] == [2, 3, 4, 5, 6]
    assert all(len(row["body"]) == 200_000 for batch in batches for _, row in batch)


def test_comments_pipeline_does_not_call_pandas_read_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    write_comments(source, [raw_comment()])
    write_products(products, [1])

    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected read_csv")),
    )
    audit = run_comments_pipeline(source, products, output, chunksize=1)
    assert audit["rows"]["final_comments_clean_count"] == 1


def test_comment_decimal_ratings_round_trip_through_sqlite_and_parquet(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    raw_rates = ("0.05", "0.15", "0.25", "0.5", "0.75", "1", "3.33", "4.85", "5")
    write_comments(
        source,
        [raw_comment(id=str(index), rate=rate) for index, rate in enumerate(raw_rates, start=1)],
    )
    write_products(products, list(range(1, len(raw_rates) + 1)))

    run_comments_pipeline(source, products, output, chunksize=2)
    table = pq.read_table(output / "comments_clean.parquet")
    assert table.schema.field("rate").type == COMMENT_RATE_TYPE
    assert table.column("rate").to_pylist() == [Decimal(rate) for rate in raw_rates]
    rows = read_rows(output, "comments_clean.parquet")
    assert [row["rate"] for row in rows] == [Decimal(rate) for rate in raw_rates]
    assert all(isinstance(row["rate"], Decimal) for row in rows)


def test_over_scale_comment_decimal_is_audited_without_whole_run_failure(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    temporary = tmp_path / "temporary"
    write_comments(source, [raw_comment(rate="0.001")])
    write_products(products, [1])

    audit = run_comments_pipeline(source, products, output, chunksize=1, temp_dir=temporary)
    clean = read_rows(output, "comments_clean.parquet")
    row_audit = read_rows(output, "row_audit.parquet")

    assert clean[0]["rate"] is None and clean[0]["invalid_rate"]
    assert audit["row_audit_counts_by_rule"]["COM-021"] == 1
    assert audit["field_quality_counts"]["invalid_ratings"] == 1
    assert audit["field_quality_counts"]["over_scale_ratings"] == 1
    assert row_audit[0]["rule_id"] == "COM-021"
    assert (output / COMPLETION_MARKER).read_text() == "success\n"
    assert not list(tmp_path.glob(".output.staging-*"))
    assert not list(temporary.glob("digikala-clean-comments-*"))


def test_sentinels_correct_completeness_and_cleaned_content_identity(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    rows = [
        raw_comment(id="1", title="nan"),
        raw_comment(id="1", title=""),
        raw_comment(id="2", title="nan"),
        raw_comment(id="2", title="recorded title"),
    ]
    write_comments(source, rows)
    write_products(products, [1])

    audit = run_comments_pipeline(source, products, output, chunksize=1)
    clean = read_rows(output, "comments_clean.parquet")
    conflicts = read_rows(output, "comment_conflicts.parquet")
    row_audit = read_rows(output, "row_audit.parquet")

    first = next(row for row in clean if row["comment_id"] == 1)
    second = next(row for row in clean if row["comment_id"] == 2)
    assert first["title"] is None and not first["comment_id_conflict"]
    assert second["title"] == "recorded title" and second["comment_id_conflict"]
    assert [row["comment_id"] for row in conflicts] == [2, 2]
    assert next(row for row in conflicts if row["selected_as_canonical"])["completeness"] == 12
    assert audit["rows"]["identical_clean_duplicate_id_rows_removed"] == 1
    assert audit["field_quality_counts"]["title_sentinel_to_null"] == 2
    assert not any(row["rule_id"] == "COM-015" for row in row_audit)


def test_comment_source_row_is_final_tie_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    write_comments(source, [raw_comment(title="الف"), raw_comment(title="ب")])
    write_products(products, [1])
    monkeypatch.setattr(
        "digikala_llm.comments_pipeline._canonical_digest", lambda _content: "0" * 64
    )
    run_comments_pipeline(source, products, output, chunksize=1)
    clean = read_rows(output, "comments_clean.parquet")
    assert clean[0]["canonical_source_row_number"] == 2
    assert clean[0]["title"] == "الف"


def test_comments_outputs_are_deterministic_across_chunksizes(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_comments(source, integration_rows())
    write_products(products, [1, 2])
    audit_one = run_comments_pipeline(source, products, first, chunksize=1)
    audit_two = run_comments_pipeline(source, products, second, chunksize=5)
    for filename in (
        "comments_clean.parquet",
        "comment_conflicts.parquet",
        "comment_quarantine.parquet",
        "row_audit.parquet",
    ):
        assert read_rows(first, filename) == read_rows(second, filename)
        assert hashlib.sha256((first / filename).read_bytes()).digest() == hashlib.sha256(
            (second / filename).read_bytes()
        ).digest()
    for key in (
        "rows",
        "rule_event_counts",
        "aggregate_transformations",
        "field_quality_counts",
        "accepted_date_range",
    ):
        assert audit_one[key] == audit_two[key]


def test_comments_parquet_schemas_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    write_comments(source, integration_rows())
    write_products(products, [1, 2])
    run_comments_pipeline(source, products, output, chunksize=4)
    expected = {
        "comments_clean.parquet": COMMENTS_CLEAN_SCHEMA,
        "comment_conflicts.parquet": COMMENT_CONFLICT_SCHEMA,
        "comment_quarantine.parquet": QUARANTINE_SCHEMA,
        "row_audit.parquet": ROW_AUDIT_SCHEMA,
    }
    for filename, schema in expected.items():
        table = pq.read_table(output / filename)
        assert table.schema == schema
        assert pa.Table.from_pylist(table.to_pylist(), schema=schema).schema == schema


def test_existing_output_force_and_failed_force_preserves_completed_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    write_comments(source, [raw_comment()])
    write_products(products, [1])
    run_comments_pipeline(source, products, output, chunksize=1)
    original = (output / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_comments_pipeline(source, products, output, chunksize=1)

    def fail(phase: str) -> None:
        if phase == "before_completion":
            raise RuntimeError("injected forced failure")

    with pytest.raises(RuntimeError, match="injected forced failure"):
        run_comments_pipeline(
            source, products, output, chunksize=1, force=True, failure_injector=fail
        )
    assert (output / "run_manifest.json").read_bytes() == original
    write_comments(source, [raw_comment(id="2")])
    run_comments_pipeline(source, products, output, chunksize=1, force=True)
    assert read_rows(output, "comments_clean.parquet")[0]["comment_id"] == 2


def test_force_publish_failure_rolls_back_completed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    write_comments(source, [raw_comment()])
    write_products(products, [1])
    run_comments_pipeline(source, products, output, chunksize=1)
    original = (output / "run_manifest.json").read_bytes()
    original_rmtree = __import__("shutil").rmtree

    def fail_backup_removal(path: object, *args: object, **kwargs: object) -> None:
        if ".backup-" in str(path):
            raise OSError("injected backup removal failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("digikala_llm.products_pipeline.shutil.rmtree", fail_backup_removal)
    with pytest.raises(OSError, match="injected backup removal failure"):
        run_comments_pipeline(source, products, output, chunksize=1, force=True)
    assert (output / "run_manifest.json").read_bytes() == original
    assert (output / COMPLETION_MARKER).read_text() == "success\n"
    assert not list(tmp_path.glob(".output.staging-*"))
    assert not list(tmp_path.glob(".output.backup-*"))


def test_mid_run_failure_cleans_all_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "comments.csv"
    products = tmp_path / "products.parquet"
    output = tmp_path / "output"
    temporary = tmp_path / "temporary"
    write_comments(source, integration_rows())
    write_products(products, [1, 2])

    def fail(phase: str) -> None:
        if phase == "after_ingest":
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        run_comments_pipeline(
            source,
            products,
            output,
            chunksize=2,
            temp_dir=temporary,
            failure_injector=fail,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".output.staging-*"))
    assert not list(temporary.glob("digikala-clean-comments-*"))
    assert not list(tmp_path.rglob(COMPLETION_MARKER))


def test_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["comments.csv", "--products-clean", "products.parquet", "--output-dir", "output"]
    )
    assert args.chunksize == 100_000
    assert args.max_rows is None
    assert not args.force
