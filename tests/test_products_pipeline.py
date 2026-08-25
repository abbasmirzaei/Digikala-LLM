import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from digikala_llm.cleaning import (
    OFFERS_CLEAN_SCHEMA,
    PRODUCT_CONFLICT_SCHEMA,
    PRODUCTS_CLEAN_SCHEMA,
    QUARANTINE_SCHEMA,
    ROW_AUDIT_SCHEMA,
)
from digikala_llm.products_pipeline import (
    COMPLETION_MARKER,
    build_parser,
    run_products_pipeline,
)

PRODUCT_COLUMNS = [
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
]


def raw_product(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "1",
        "title_fa": "محصول یک",
        "Category1": "دسته یک",
        "Category2": "دسته دو",
        "Brand": "برند",
        "Rate": "80",
        "Rate_cnt": "10",
        "sub_category": "test",
        "Seller": "فروشنده یک",
        "Price": "100",
        "Is_Fake": "False",
        "min_price_last_month": "90",
    }
    row.update(updates)
    return row


def write_products(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=PRODUCT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(output: Path, filename: str) -> list[dict[str, object]]:
    return pq.read_table(output / filename).to_pylist()


def synthetic_rows() -> list[dict[str, object]]:
    duplicate = raw_product()
    return [
        duplicate,
        dict(duplicate),
        raw_product(Seller="فروشنده دو", Price="110"),
        raw_product(Seller="فروشنده یک", Price="120"),
        raw_product(id="2", title_fa="رتبه کم", Rate_cnt="5", Seller="S2"),
        raw_product(id="2", title_fa="رتبه زیاد", Rate_cnt="20", Seller="S2"),
        raw_product(id="3", title_fa="ناقص", Brand="", Rate_cnt="8", Seller="S3"),
        raw_product(id="3", title_fa="کامل", Brand="برند", Rate_cnt="8", Seller="S4"),
        raw_product(id="4", title_fa="هش الف", Rate_cnt="7", Seller="S5"),
        raw_product(id="4", title_fa="هش ب", Rate_cnt="7", Seller="S6"),
        raw_product(id="5", title_fa="محصول معتبر", Price="bad", Seller="S7"),
        raw_product(id="6", title_fa="محصول نامعتبر", Rate="101", Price="600", Seller="S8"),
        raw_product(id="7", title_fa="قیمت صفر", Price="0", Seller="nan"),
    ]


def test_products_pipeline_end_to_end_and_independent_sides(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    write_products(source, synthetic_rows())
    audit = run_products_pipeline(source, output, chunksize=3)
    assert audit["specification_version"] == "1.0.3"

    assert (output / COMPLETION_MARKER).read_text() == "success\n"
    products = read_rows(output, "products_clean.parquet")
    offers = read_rows(output, "offers_clean.parquet")
    conflicts = read_rows(output, "product_conflicts.parquet")
    product_quarantine = read_rows(output, "product_quarantine.parquet")
    offer_quarantine = read_rows(output, "offer_quarantine.parquet")

    assert [row["product_id"] for row in products] == [1, 2, 3, 4, 5, 7]
    assert next(row for row in products if row["product_id"] == 2)["title_fa"] == "رتبه زیاد"
    assert next(row for row in products if row["product_id"] == 3)["title_fa"] == "کامل"
    assert next(row for row in products if row["product_id"] == 1)[
        "canonical_source_row_number"
    ] == 2
    assert all(row["product_id"] != 6 for row in products)

    product_four = [row for row in conflicts if row["product_id"] == 4]
    assert len(product_four) == 2
    selected = next(row for row in product_four if row["selected_as_canonical"])
    assert selected["core_digest"] == min(row["core_digest"] for row in product_four)
    assert {row["seller"] for row in offers if row["product_id"] == 1} == {
        "فروشنده یک",
        "فروشنده دو",
    }
    assert {row["price_raw"] for row in offers if row["product_id"] == 1} == {100, 110, 120}
    product_two_offer = next(row for row in offers if row["product_id"] == 2)
    assert product_two_offer["source_row_number"] == 6
    assert all(row["product_id"] != 5 for row in offers)
    assert all(row["product_id"] != 6 for row in offers)
    zero = next(row for row in offers if row["product_id"] == 7)
    assert zero["price_raw"] is None and zero["invalid_price"]
    assert zero["seller"] == "nan"

    assert len(product_quarantine) == 1
    assert product_quarantine[0]["entity_id"] == 6
    assert {row["entity_id"] for row in offer_quarantine} == {5, 6}
    rules = {rule for row in offer_quarantine for rule in row["rule_ids"]}
    assert {"OFF-004", "OFF-009"} <= rules

    rows = audit["rows"]
    assert rows["input_rows"] == len(synthetic_rows())
    assert rows["exact_duplicate_rows_removed"] == 1
    assert rows["distinct_raw_rows_retained"] == len(synthetic_rows()) - 1
    assert rows["distinct_offer_count"] == len(offers)
    assert rows["unique_product_count"] == len(products)
    assert rows["conflict_product_id_count"] == 3
    assert rows["exact_offer_duplicates_removed"] == 1
    assert all(audit["reconciliation"]["checks_passed"].values())
    assert audit["rule_event_counts"]["OFF-008"] == 0
    assert audit["rule_event_counts"]["OFF-009"] == 1
    assert audit["rule_event_counts"]["PRD-001"] == 1
    assert audit["rule_event_counts"]["RAW-001"] == len(synthetic_rows())
    assert audit["rule_event_counts"]["ID-001"] == rows["distinct_raw_rows_retained"]
    assert audit["rule_event_counts"]["BOOL-001"] == rows["distinct_raw_rows_retained"]
    persisted_audit = json.loads((output / "cleaning_audit.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert persisted_audit == audit
    assert audit["source"]["sha256_before"] == audit["source"]["sha256_after"]
    assert audit["source"]["unchanged"] is True
    assert manifest["status"] == "success"
    assert manifest["currency_status"] == "IRR_inferred"
    assert "not a business identifier" in audit["offer_id_semantics"]
    for filename, metadata in audit["outputs"].items():
        digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert metadata["sha256"] == digest
        assert metadata["rows"] == pq.read_metadata(output / filename).num_rows
    assert audit["status"] == "success"


def test_nearest_rank_p999_uses_final_distinct_positive_offers(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    output = tmp_path / "prices-output"
    rows = [
        raw_product(id="99", Seller=f"S{number}", Price=str(number))
        for number in range(1, 1002)
    ]
    rows.append(raw_product(id="99", title_fa="conflicting facts", Seller="S1", Price="1"))
    write_products(source, rows)
    audit = run_products_pipeline(source, output, chunksize=73)
    price_review = audit["price_review"]
    assert price_review["valid_positive_price_population"] == 1001
    assert price_review["rank"] == 1000
    assert price_review["threshold"] == 1000
    assert audit["rows"]["exact_offer_duplicates_removed"] == 1
    offers = read_rows(output, "offers_clean.parquet")
    flagged = [row for row in offers if row["high_price_review"]]
    assert len(flagged) == 1
    assert flagged[0]["price_raw"] == 1001


def test_logical_outputs_and_counts_are_chunk_size_independent(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_products(source, synthetic_rows())
    audit_one = run_products_pipeline(source, first, chunksize=1)
    audit_two = run_products_pipeline(source, second, chunksize=5)
    for filename in (
        "products_clean.parquet",
        "offers_clean.parquet",
        "product_conflicts.parquet",
        "product_quarantine.parquet",
        "offer_quarantine.parquet",
        "row_audit.parquet",
    ):
        assert read_rows(first, filename) == read_rows(second, filename)
        assert hashlib.sha256((first / filename).read_bytes()).hexdigest() == hashlib.sha256(
            (second / filename).read_bytes()
        ).hexdigest()
    assert audit_one["rows"] == audit_two["rows"]
    assert audit_one["rule_event_counts"] == audit_two["rule_event_counts"]
    assert audit_one["aggregate_transformations"] == audit_two["aggregate_transformations"]
    assert audit_one["price_review"] == audit_two["price_review"]


def test_all_pipeline_parquet_schemas_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    write_products(source, synthetic_rows())
    run_products_pipeline(source, output, chunksize=4)
    expected = {
        "products_clean.parquet": PRODUCTS_CLEAN_SCHEMA,
        "offers_clean.parquet": OFFERS_CLEAN_SCHEMA,
        "product_conflicts.parquet": PRODUCT_CONFLICT_SCHEMA,
        "product_quarantine.parquet": QUARANTINE_SCHEMA,
        "offer_quarantine.parquet": QUARANTINE_SCHEMA,
        "row_audit.parquet": ROW_AUDIT_SCHEMA,
    }
    for filename, schema in expected.items():
        table = pq.read_table(output / filename)
        assert table.schema == schema
        assert pa.Table.from_pylist(table.to_pylist(), schema=schema).schema == schema


def test_existing_output_refusal_and_force_replacement(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    write_products(source, [raw_product()])
    run_products_pipeline(source, output, chunksize=1)
    original_manifest = json.loads((output / "run_manifest.json").read_text())
    with pytest.raises(FileExistsError):
        run_products_pipeline(source, output, chunksize=1)
    write_products(source, [raw_product(id="2")])
    run_products_pipeline(source, output, chunksize=1, force=True)
    assert read_rows(output, "products_clean.parquet")[0]["product_id"] == 2
    replaced_manifest = json.loads((output / "run_manifest.json").read_text())
    assert original_manifest["source"]["sha256_before"] != replaced_manifest["source"][
        "sha256_before"
    ]


def test_mid_run_exception_cleans_staging_and_sqlite(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    temporary = tmp_path / "temporary"
    write_products(source, synthetic_rows())

    def fail(phase: str) -> None:
        if phase == "after_ingest":
            raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        run_products_pipeline(
            source,
            output,
            chunksize=3,
            temp_dir=temporary,
            failure_injector=fail,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".cleaned.staging-*"))
    assert not list(temporary.glob("digikala-clean-products-*"))
    assert not list(tmp_path.rglob(COMPLETION_MARKER))


def test_success_cleans_temporary_sqlite_files(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    temporary = tmp_path / "temporary"
    write_products(source, [raw_product()])
    run_products_pipeline(source, output, chunksize=1, temp_dir=temporary)
    assert not list(temporary.glob("digikala-clean-products-*"))


def test_exact_row_dedup_does_not_trust_sha256_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    write_products(source, [raw_product(id="1"), raw_product(id="2")])

    monkeypatch.setattr(
        "digikala_llm.products_pipeline._raw_digest", lambda _content: "0" * 64
    )
    audit = run_products_pipeline(source, output, chunksize=1)
    assert audit["rows"]["exact_duplicate_rows_removed"] == 0
    assert audit["rows"]["distinct_raw_rows_retained"] == 2
    assert [row["product_id"] for row in read_rows(output, "products_clean.parquet")] == [1, 2]


def test_max_rows_is_an_exact_bound(tmp_path: Path) -> None:
    source = tmp_path / "products.csv"
    output = tmp_path / "cleaned"
    write_products(source, [raw_product(id=str(number)) for number in range(1, 8)])
    audit = run_products_pipeline(source, output, chunksize=2, max_rows=5)
    assert audit["rows"]["input_rows"] == 5
    assert [row["product_id"] for row in read_rows(output, "products_clean.parquet")] == [
        1, 2, 3, 4, 5
    ]


def test_cli_parser_defaults_and_required_output() -> None:
    parser = build_parser()
    args = parser.parse_args(["products.csv", "--output-dir", "output"])
    assert args.chunksize == 100_000
    assert args.max_rows is None
    assert not args.force
