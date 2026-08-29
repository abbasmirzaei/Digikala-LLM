import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from digikala_llm import sunscreen_builder as builder
from digikala_llm.cleaning import PRODUCTS_CLEAN_SCHEMA
from digikala_llm.comments_pipeline import COMMENT_SOURCE_COLUMNS


def _product(product_id: int, category2: str = "کرم ضد آفتاب", brand: str | None = "A") -> dict[str, object]:
    return {
        "product_id": product_id,
        "title_fa": f"Product {product_id}",
        "category1": "مراقبت پوست",
        "category2": category2,
        "brand": brand,
        "rate": None,
        "rate_count": 0,
        "sub_category": None,
        "is_unrated": True,
        "inconsistent_zero_rate": False,
        "core_attribute_conflict": False,
        "canonical_source_row_number": product_id,
    }


def _comment(comment_id: int, product_id: int, body: str = "good", buyer: str = "True") -> list[str]:
    return [
        str(comment_id), "title", body, "1 فروردین 1400", "5", "recommended", buyer,
        str(product_id), "", "", "1", "0", "seller", "1", "",
    ]


def _inputs(tmp_path: Path, comments: list[list[str]]) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    products, offers, comments_path = tmp_path / "products.parquet", tmp_path / "offers.parquet", tmp_path / "comments.csv"
    pq.write_table(pa.Table.from_pylist([_product(1), _product(2, "other"), _product(3, brand=None)], schema=PRODUCTS_CLEAN_SCHEMA), products)
    pq.write_table(pa.table({"product_id": [1, 1, 2, 3], "price_raw": [100, 90, 30, None]}), offers)
    with comments_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(COMMENT_SOURCE_COLUMNS)
        writer.writerows(comments)
    return products, offers, comments_path


def _build(tmp_path: Path, comments: list[list[str]], **kwargs: object) -> tuple[dict[str, object], Path]:
    products, offers, comments_path = _inputs(tmp_path, comments)
    output = tmp_path / "out"
    manifest = builder.build_sunscreen_dataset(products, offers, comments_path, output, validate_evidence=False, **kwargs)
    return manifest, output


def test_scoped_builder_selects_exact_categories_and_excludes_unrelated_comments(tmp_path: Path) -> None:
    manifest, output = _build(tmp_path, [_comment(1, 1), _comment(2, 2), _comment(3, 3)])
    assert manifest["rows"]["selected_products"] == 2
    assert manifest["rows"]["raw_matching_comment_rows"] == 2
    assert pq.read_table(output / "sunscreen_products.parquet").column("product_id").to_pylist() == [1, 3]
    assert pq.read_table(output / "sunscreen_comments_raw.parquet").column("product_id").to_pylist() == [1, 3]


def test_one_pass_deduplication_canonicalization_and_traceability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = builder.iter_exact_csv_batches

    def counting(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(builder, "iter_exact_csv_batches", counting)
    duplicate = _comment(1, 1)
    conflict = _comment(1, 1, body="better")
    manifest, output = _build(tmp_path, [duplicate, duplicate, conflict, _comment(9, 2)])
    assert calls == 1
    assert manifest["rows"]["raw_matching_comment_rows"] == 3
    assert manifest["rows"]["exact_raw_duplicate_rows_removed"] == 1
    assert manifest["rows"]["conflicting_comment_id_count"] == 1
    canonical = pq.read_table(output / "sunscreen_comments_canonical.parquet").to_pylist()
    assert len(canonical) == 1
    assert canonical[0]["comment_id"] == 1
    assert canonical[0]["comment_id_conflict"] is True
    assert canonical[0]["canonical_source_row_number"] in {2, 4}
    events = pq.read_table(output / "sunscreen_comment_audit.parquet").column("event").to_pylist()
    assert "exact_raw_duplicate" in events
    assert "conflicting_comment_id_noncanonical" in events


def test_output_is_deterministic_and_schemas_reconcile(tmp_path: Path) -> None:
    comments = [_comment(1, 1), _comment(2, 3, body="nan", buyer="False")]
    manifest_a, output_a = _build(tmp_path / "a", comments, chunksize=1)
    _manifest_b, output_b = _build(tmp_path / "b", comments, chunksize=10)
    for name, schema in builder.OUTPUT_SCHEMAS.items():
        assert pq.read_schema(output_a / name) == schema
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()
    assert manifest_a["rows"]["products_with_historical_price"] == 1
    assert manifest_a["rows"]["historical_price_coverage_pct"] == 50
    assert (output_a / "_SUCCESS").is_file()
    assert json.loads((output_a / "manifest.json").read_text(encoding="utf-8"))["configuration"]["sqlite_used"] is False


def test_atomic_failure_cleans_staging_and_does_not_write_success(tmp_path: Path) -> None:
    products, offers, comments = _inputs(tmp_path, [_comment(1, 1)])

    def fail(phase: str) -> None:
        assert phase == "after_validation"
        raise RuntimeError("injected")

    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="injected"):
        builder.build_sunscreen_dataset(products, offers, comments, output, validate_evidence=False, failure_injector=fail)
    assert not output.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_evidence_gate_reconciles_exact_counts() -> None:
    metrics = dict(builder.EVIDENCE_GATE)
    metrics["historical_price_coverage_pct"] = 100 * 1046 / 1048
    builder._validate_gate(metrics)
    metrics["buyer_comment_rows"] = 1
    with pytest.raises(RuntimeError, match="buyer_comment_rows"):
        builder._validate_gate(metrics)
