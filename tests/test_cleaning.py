from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from digikala_llm.cleaning import (
    COMMENT_ROW_CANDIDATE_SCHEMA,
    COMMENTS_CLEAN_SCHEMA,
    INT64_MAX,
    MAX_JALALI_YEAR,
    MIN_JALALI_YEAR,
    MONEY_TYPE,
    OFFER_ROW_CANDIDATE_SCHEMA,
    OFFERS_CLEAN_SCHEMA,
    PRODUCT_ROW_CANDIDATE_SCHEMA,
    PRODUCTS_CLEAN_SCHEMA,
    QUARANTINE_SCHEMA,
    ROW_AUDIT_SCHEMA,
    RULE_CATALOG,
    RowTransformResult,
    clean_text,
    parse_jalali_date,
    parse_required_id,
    parse_strict_boolean,
    transform_comment_row,
    transform_offer_row,
    transform_product_row,
)


def product_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "12", "title_fa": "گوشی نمونه", "Category1": "دیجیتال",
        "Category2": "موبایل", "Brand": "نمونه", "Rate": "87.5",
        "Rate_cnt": "8", "sub_category": "mobile", "Seller": "فروشنده",
        "Price": "101", "Is_Fake": "False", "min_price_last_month": "90",
    }
    row.update(updates)
    return row


def comment_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "20", "product_id": "12", "title": "عنوان", "body": "متن نظر",
        "created_at": "23 تیر 1395", "rate": "4",
        "recommendation_status": "recommended", "is_buyer": "True",
        "advantages": "خوب", "disadvantages": "", "likes": "2",
        "dislikes": "0", "seller_title": "فروشنده", "seller_code": "9",
        "true_to_size_rate": "",
    }
    row.update(updates)
    return row


@pytest.mark.parametrize("value", [0, "0", INT64_MAX, str(INT64_MAX), "0012", "1.0"])
def test_required_id_accepts_nonnegative_integral_int64(value: object) -> None:
    result = parse_required_id(value)
    assert result.valid
    assert result.value == int(Decimal(str(value)))


@pytest.mark.parametrize(
    "value", [None, "", "   ", "abc", "1.5", -1, str(INT64_MAX + 1), "NaN", True]
)
def test_required_id_rejects_invalid_values(value: object) -> None:
    result = parse_required_id(value)
    assert not result.valid
    assert result.value is None


@pytest.mark.parametrize(
    ("value", "expected"), [(True, True), (False, False), ("True", True), ("False", False)]
)
def test_strict_boolean_accepts_only_observed_forms(value: object, expected: bool) -> None:
    result = parse_strict_boolean(value)
    assert result.valid
    assert result.value is expected


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", " TRUE ", "", "yes"])
def test_strict_boolean_rejects_truthiness_and_unrecognized_forms(value: object) -> None:
    assert not parse_strict_boolean(value).valid


def test_clean_text_only_nulls_blank_values_and_preserves_unicode() -> None:
    assert clean_text(None) is None
    assert clean_text("  \t") is None
    original = "  يک  متن فارسی ك  "
    assert clean_text(original) == original


@pytest.mark.parametrize(
    ("rate", "count", "expected_rate", "unrated", "inconsistent"),
    [("0", "0", None, True, False), ("0", "2", 0.0, False, True),
     ("100", "1", 100.0, False, False), ("72.5", "4", 72.5, False, False)],
)
def test_product_rating_branches(
    rate: str, count: str, expected_rate: float | None, unrated: bool, inconsistent: bool
) -> None:
    result = transform_product_row(product_row(Rate=rate, Rate_cnt=count), 7)
    assert not result.quarantined
    assert result.candidate_row is not None
    assert result.candidate_row["rate"] == expected_rate
    assert result.candidate_row["is_unrated"] is unrated
    assert result.candidate_row["inconsistent_zero_rate"] is inconsistent


@pytest.mark.parametrize(
    ("updates", "rule_id"),
    [({"Rate": "101"}, "PRD-004"), ({"Rate": "x"}, "PRD-004"),
     ({"Rate_cnt": "-1"}, "PRD-007"), ({"Rate_cnt": "1.2"}, "PRD-007")],
)
def test_invalid_product_rating_fields_quarantine(
    updates: dict[str, object], rule_id: str
) -> None:
    result = transform_product_row(product_row(**updates), 7)
    assert result.quarantined
    assert result.quarantine_record is not None
    assert rule_id in result.quarantine_record["rule_ids"]


@pytest.mark.parametrize(
    ("raw_rate", "rate", "unrated", "invalid"),
    [("1", 1.0, False, False), ("5", 5.0, False, False),
     ("3.5", 3.5, False, False), ("0", None, True, False),
     ("2500", None, False, True), ("-1", None, False, True),
     ("bad", None, False, True), (None, None, False, False)],
)
def test_comment_rating_branches(
    raw_rate: object, rate: float | None, unrated: bool, invalid: bool
) -> None:
    result = transform_comment_row(comment_row(rate=raw_rate), 3)
    assert not result.quarantined
    assert result.candidate_row is not None
    assert result.candidate_row["rate"] == rate
    assert result.candidate_row["is_unrated"] is unrated
    assert result.candidate_row["invalid_rate"] is invalid
    assert result.candidate_row["recommendation_status"] == "recommended"


def test_offer_price_conversion_is_exact_and_raw_is_preserved() -> None:
    result = transform_offer_row(product_row(Price="101"), 11)
    assert result.candidate_row is not None
    assert result.candidate_row["price_raw"] == 101
    assert result.candidate_row["price_toman"] == Decimal("10.1")
    pa.Table.from_pylist([result.candidate_row], schema=OFFER_ROW_CANDIDATE_SCHEMA)


def test_zero_price_is_null_and_flagged_without_quarantine() -> None:
    result = transform_offer_row(product_row(Price="0"), 11)
    assert not result.quarantined
    assert result.candidate_row is not None
    assert result.candidate_row["price_raw"] is None
    assert result.candidate_row["price_toman"] is None
    assert result.candidate_row["invalid_price"] is True


@pytest.mark.parametrize("price", ["-1", "bad", "1.5", str(INT64_MAX + 1), None])
def test_invalid_prices_are_quarantined(price: object) -> None:
    assert transform_offer_row(product_row(Price=price), 11).quarantined


@pytest.mark.parametrize(
    ("history", "stored", "missing"),
    [("0", None, True), (None, None, False), ("75", 75, False)],
)
def test_price_history_rules(history: object, stored: int | None, missing: bool) -> None:
    result = transform_offer_row(product_row(min_price_last_month=history), 11)
    assert not result.quarantined
    assert result.candidate_row is not None
    assert result.candidate_row["min_price_last_month"] == stored
    assert result.candidate_row["missing_price_history"] is missing


@pytest.mark.parametrize(
    ("raw", "jalali", "gregorian"),
    [("23 تیر 1395", "1395-04-23", date(2016, 7, 13)),
     ("26 مهر 1402", "1402-07-26", date(2023, 10, 18)),
     ("1 فروردین 1397", "1397-01-01", date(2018, 3, 21)),
     ("1 فروردین 1402", "1402-01-01", date(2023, 3, 21)),
     ("30 اسفند 1399", "1399-12-30", date(2021, 3, 20))],
)
def test_jalali_known_anchors(raw: str, jalali: str, gregorian: date) -> None:
    result = parse_jalali_date(raw)
    assert result.valid and result.value is not None
    assert result.value.raw == raw
    assert result.value.canonical_jalali == jalali
    assert result.value.gregorian == gregorian


@pytest.mark.parametrize(
    ("year", "gregorian"),
    [(1397, date(2018, 3, 21)), (1398, date(2019, 3, 21)),
     (1399, date(2020, 3, 20)), (1400, date(2021, 3, 21)),
     (1401, date(2022, 3, 21)), (1402, date(2023, 3, 21)),
     (1403, date(2024, 3, 20))],
)
def test_diagnostic_nowruz_anchors(year: int, gregorian: date) -> None:
    result = parse_jalali_date(f"1 فروردین {year}")
    assert result.valid and result.value is not None
    assert result.value.gregorian == gregorian


@pytest.mark.parametrize(
    ("month", "gregorian"),
    [("فروردین", date(2023, 3, 21)), ("اردیبهشت", date(2023, 4, 21)),
     ("خرداد", date(2023, 5, 22)), ("تیر", date(2023, 6, 22)),
     ("مرداد", date(2023, 7, 23)), ("شهریور", date(2023, 8, 23)),
     ("مهر", date(2023, 9, 23)), ("آبان", date(2023, 10, 23)),
     ("آذر", date(2023, 11, 22)), ("دی", date(2023, 12, 22)),
     ("بهمن", date(2024, 1, 21)), ("اسفند", date(2024, 2, 20))],
)
def test_all_twelve_persian_month_names(month: str, gregorian: date) -> None:
    result = parse_jalali_date(f"1 {month} 1402")
    assert result.valid and result.value is not None
    assert result.value.gregorian == gregorian


@pytest.mark.parametrize(
    ("month", "last_day", "gregorian"),
    [("فروردین", 31, date(2023, 4, 20)), ("اردیبهشت", 31, date(2023, 5, 21)),
     ("خرداد", 31, date(2023, 6, 21)), ("تیر", 31, date(2023, 7, 22)),
     ("مرداد", 31, date(2023, 8, 22)), ("شهریور", 31, date(2023, 9, 22)),
     ("مهر", 30, date(2023, 10, 22)), ("آبان", 30, date(2023, 11, 21)),
     ("آذر", 30, date(2023, 12, 21)), ("دی", 30, date(2024, 1, 20)),
     ("بهمن", 30, date(2024, 2, 19)), ("اسفند", 29, date(2024, 3, 19))],
)
def test_all_twelve_jalali_month_end_boundaries(
    month: str, last_day: int, gregorian: date
) -> None:
    valid = parse_jalali_date(f"{last_day} {month} 1402")
    invalid = parse_jalali_date(f"{last_day + 1} {month} 1402")
    assert valid.valid and valid.value is not None
    assert valid.value.gregorian == gregorian
    assert not invalid.valid


@pytest.mark.parametrize(
    "raw",
    ["30 اسفند 1400", "32 فروردین 1402", "31 مهر 1402", "0 تیر 1395",
     "23 Foo 1395", "23 تیر 95", "۲۳ تیر ۱۳۹۵", " 23 تیر 1395",
     "23  تیر 1395", "23 تیر 1395 ", "30 اسفند 9999", "", None],
)
def test_jalali_rejects_invalid_or_noncanonical_input(raw: object) -> None:
    assert not parse_jalali_date(raw).valid


def test_comment_date_outputs_and_raw_text_preservation() -> None:
    original_body = "  يک  نظر كاربردی  "
    result = transform_comment_row(comment_row(body=original_body), 4)
    assert result.candidate_row is not None
    assert result.candidate_row["body"] == original_body
    assert result.candidate_row["created_at_raw"] == "23 تیر 1395"
    assert result.candidate_row["created_at_jalali"] == "1395-04-23"
    assert result.candidate_row["created_at_gregorian"] == date(2016, 7, 13)
    pa.Table.from_pylist([result.candidate_row], schema=COMMENT_ROW_CANDIDATE_SCHEMA)


def test_invalid_required_comment_fields_are_quarantined() -> None:
    cases = (("id", "1.5"), ("product_id", "-1"),
             ("is_buyer", "yes"), ("created_at", "bad"))
    for field, value in cases:
        result = transform_comment_row(comment_row(**{field: value}), 4)
        assert result.quarantined, field


def test_schema_types_and_required_nullability() -> None:
    assert PRODUCTS_CLEAN_SCHEMA.field("product_id").type == pa.int64()
    assert not PRODUCTS_CLEAN_SCHEMA.field("product_id").nullable
    assert OFFERS_CLEAN_SCHEMA.field("price_raw").type == pa.int64()
    assert OFFERS_CLEAN_SCHEMA.field("price_toman").type == MONEY_TYPE
    assert COMMENTS_CLEAN_SCHEMA.field("created_at_gregorian").type == pa.date32()
    assert not COMMENTS_CLEAN_SCHEMA.field("created_at_gregorian").nullable
    assert QUARANTINE_SCHEMA.field("rule_ids").type == pa.list_(pa.string())
    assert not ROW_AUDIT_SCHEMA.field("is_lossy").nullable
    assert {field.name for field in PRODUCTS_CLEAN_SCHEMA if not field.nullable} == {
        "product_id", "is_unrated", "inconsistent_zero_rate",
        "core_attribute_conflict", "canonical_source_row_number",
    }
    assert {field.name for field in OFFERS_CLEAN_SCHEMA if not field.nullable} == {
        "offer_id", "product_id", "missing_price_history", "invalid_price",
        "high_price_review", "source_row_number",
    }
    assert {field.name for field in COMMENTS_CLEAN_SCHEMA if not field.nullable} == {
        "comment_id", "product_id", "created_at_raw", "created_at_jalali",
        "created_at_gregorian", "is_unrated", "invalid_rate",
        "comment_id_conflict", "canonical_source_row_number",
    }


def test_clean_schema_signatures_exactly_match_specification() -> None:
    def signature(schema: pa.Schema) -> list[tuple[str, pa.DataType, bool]]:
        return [(field.name, field.type, field.nullable) for field in schema]

    assert signature(PRODUCTS_CLEAN_SCHEMA) == [
        ("product_id", pa.int64(), False), ("title_fa", pa.string(), True),
        ("category1", pa.string(), True), ("category2", pa.string(), True),
        ("brand", pa.string(), True), ("rate", pa.float64(), True),
        ("rate_count", pa.int64(), True), ("sub_category", pa.string(), True),
        ("is_unrated", pa.bool_(), False),
        ("inconsistent_zero_rate", pa.bool_(), False),
        ("core_attribute_conflict", pa.bool_(), False),
        ("canonical_source_row_number", pa.int64(), False),
    ]
    assert signature(OFFERS_CLEAN_SCHEMA) == [
        ("offer_id", pa.string(), False), ("product_id", pa.int64(), False),
        ("seller", pa.string(), True), ("price_raw", pa.int64(), True),
        ("price_toman", MONEY_TYPE, True), ("is_fake", pa.bool_(), True),
        ("min_price_last_month", pa.int64(), True),
        ("missing_price_history", pa.bool_(), False),
        ("invalid_price", pa.bool_(), False),
        ("high_price_review", pa.bool_(), False),
        ("source_row_number", pa.int64(), False),
    ]
    assert signature(COMMENTS_CLEAN_SCHEMA) == [
        ("comment_id", pa.int64(), False), ("product_id", pa.int64(), False),
        ("title", pa.string(), True), ("body", pa.string(), True),
        ("created_at_raw", pa.string(), False),
        ("created_at_jalali", pa.string(), False),
        ("created_at_gregorian", pa.date32(), False),
        ("rate", pa.float64(), True), ("is_unrated", pa.bool_(), False),
        ("invalid_rate", pa.bool_(), False),
        ("recommendation_status", pa.string(), True),
        ("is_buyer", pa.bool_(), True), ("advantages", pa.string(), True),
        ("disadvantages", pa.string(), True), ("likes", pa.int64(), True),
        ("dislikes", pa.int64(), True), ("seller_title", pa.string(), True),
        ("seller_code", pa.string(), True),
        ("true_to_size_rate", pa.string(), True),
        ("comment_id_conflict", pa.bool_(), False),
        ("canonical_source_row_number", pa.int64(), False),
    ]


def test_valid_rows_conform_to_arrow_schemas() -> None:
    product = transform_product_row(product_row(), 1)
    offer = transform_offer_row(product_row(), 1)
    comment = transform_comment_row(comment_row(), 1)
    assert product.candidate_row is not None
    assert offer.candidate_row is not None
    assert comment.candidate_row is not None
    pa.Table.from_pylist([product.candidate_row], schema=PRODUCT_ROW_CANDIDATE_SCHEMA)
    pa.Table.from_pylist([offer.candidate_row], schema=OFFER_ROW_CANDIDATE_SCHEMA)
    pa.Table.from_pylist([comment.candidate_row], schema=COMMENT_ROW_CANDIDATE_SCHEMA)


def test_quarantine_preserves_raw_unicode_record() -> None:
    result = transform_offer_row(product_row(Price="نامعتبر"), 5)
    assert result.quarantine_record is not None
    assert result.quarantine_record["source_file"] == "data/raw/digikala-products.csv"
    assert "نامعتبر" in result.quarantine_record["raw_record_json"]
    pa.Table.from_pylist([result.quarantine_record], schema=QUARANTINE_SCHEMA)
    pa.Table.from_pylist(list(result.audit_records), schema=ROW_AUDIT_SCHEMA)


def test_expected_frequent_transformations_use_aggregate_counters() -> None:
    product = transform_product_row(product_row(Rate="0", Rate_cnt="0"), 2)
    offer = transform_offer_row(product_row(min_price_last_month="0"), 3)
    comment = transform_comment_row(comment_row(rate="0", title=" "), 4)
    assert product.audit_records == ()
    assert product.aggregate_counter_keys == (("PRD-005", "Rate"),)
    assert offer.audit_records == ()
    assert offer.aggregate_counter_keys == (("OFF-007", "min_price_last_month"),)
    assert comment.audit_records == ()
    assert set(comment.aggregate_counter_keys) == {
        ("COM-005", "rate"), ("TXT-001", "title"),
        ("TXT-001", "disadvantages"), ("TXT-001", "true_to_size_rate"),
        ("COM-010", "seller_code"),
    }


def test_invalid_values_keep_row_level_traceability_and_stable_rule_metadata() -> None:
    result = transform_comment_row(comment_row(likes="invalid"), 42)
    assert result.candidate_row is not None
    assert result.quarantine_record is None
    assert len(result.audit_records) == 1
    audit = result.audit_records[0]
    rule = RULE_CATALOG["COM-012"]
    assert audit["rule_id"] == "COM-012"
    assert audit["raw_value"] == "invalid"
    assert audit["source_row_number"] == 42
    assert audit["action"] == rule.action
    assert audit["severity"] == rule.severity
    assert audit["is_lossy"] == rule.is_lossy


@pytest.mark.parametrize("code", ["AB12", "00123", "123", "NAN", "کد۱۲", "  AB12  "])
def test_seller_code_is_preserved_as_exact_opaque_text(code: str) -> None:
    result = transform_comment_row(comment_row(seller_code=code), 42)
    assert result.candidate_row is not None
    assert result.candidate_row["seller_code"] == code
    assert result.audit_records == ()
    assert ("COM-010", "seller_code") in result.aggregate_counter_keys


@pytest.mark.parametrize(
    ("code", "counter"),
    [(None, "COM-011"), ("", "COM-011"), ("   ", "TXT-001"), ("nan", "COM-013")],
)
def test_missing_or_blank_seller_code_becomes_null_with_aggregate_count(
    code: object, counter: str
) -> None:
    result = transform_comment_row(comment_row(seller_code=code), 42)
    assert result.candidate_row is not None
    assert result.candidate_row["seller_code"] is None
    assert result.audit_records == ()
    assert (counter, "seller_code") in result.aggregate_counter_keys


def test_exact_lowercase_seller_title_sentinel_is_column_specific() -> None:
    sentinel = transform_comment_row(comment_row(seller_title="nan"), 42)
    uppercase = transform_comment_row(comment_row(seller_title="NAN"), 43)
    assert sentinel.candidate_row is not None
    assert sentinel.candidate_row["seller_title"] is None
    assert ("COM-013", "seller_title") in sentinel.aggregate_counter_keys
    assert sentinel.audit_records == ()
    assert uppercase.candidate_row is not None
    assert uppercase.candidate_row["seller_title"] == "NAN"


def test_transform_result_enforces_accepted_or_quarantined_invariant() -> None:
    accepted = transform_product_row(product_row(), 2)
    rejected = transform_product_row(product_row(id="1.5"), 3)
    assert accepted.candidate_row is not None and accepted.quarantine_record is None
    assert rejected.candidate_row is None and rejected.quarantine_record is not None
    assert rejected.quarantine_record["source_row_number"] == 3
    assert '"id":"1.5"' in rejected.quarantine_record["raw_record_json"]
    with pytest.raises(ValueError, match="exactly one"):
        RowTransformResult({}, (), (), {})
    with pytest.raises(ValueError, match="exactly one"):
        RowTransformResult(None, (), (), None)


def test_candidate_rows_cannot_claim_dataset_level_results() -> None:
    product = transform_product_row(product_row(), 2)
    offer = transform_offer_row(product_row(), 2)
    comment = transform_comment_row(comment_row(), 2)
    assert not product.is_final_clean_row
    assert not offer.is_final_clean_row
    assert not comment.is_final_clean_row
    assert product.candidate_row is not None
    assert offer.candidate_row is not None
    assert comment.candidate_row is not None
    assert "core_attribute_conflict" not in product.candidate_row
    assert "canonical_source_row_number" not in product.candidate_row
    assert "high_price_review" not in offer.candidate_row
    assert "comment_id_conflict" not in comment.candidate_row
    assert "canonical_source_row_number" not in comment.candidate_row


@pytest.mark.parametrize(
    ("value", "valid", "expected"),
    [(Decimal(0), True, 0), (Decimal(str(INT64_MAX)), True, INT64_MAX),
     (Decimal("1E3"), True, 1000), ("1e3", True, 1000),
     (Decimal("1.01"), False, None), (float("nan"), False, None),
     (float("inf"), False, None), (float("-inf"), False, None),
     (Decimal("NaN"), False, None), (Decimal("Infinity"), False, None),
     (True, False, None), (False, False, None)],
)
def test_identifier_numeric_edge_cases(value: object, valid: bool, expected: int | None) -> None:
    result = parse_required_id(value)
    assert result.valid is valid
    assert result.value == expected


@pytest.mark.parametrize("value", ["-1", "0.1", "NaN", "Infinity", True])
def test_rate_count_rejects_numeric_edge_cases(value: object) -> None:
    assert transform_product_row(product_row(Rate_cnt=value), 2).quarantined


def test_price_int64_limit_and_decimal_conversion_have_no_float_intermediate() -> None:
    result = transform_offer_row(product_row(Price=Decimal(str(INT64_MAX))), 2)
    assert result.candidate_row is not None
    assert result.candidate_row["price_raw"] == INT64_MAX
    assert result.candidate_row["price_toman"] == Decimal("922337203685477580.7")
    assert isinstance(result.candidate_row["price_toman"], Decimal)
    assert transform_offer_row(product_row(Price=str(INT64_MAX + 1)), 2).quarantined
    assert transform_offer_row(product_row(Price="1e2"), 2).candidate_row is not None
    assert transform_offer_row(product_row(Price="1e-1"), 2).quarantined
    assert transform_offer_row(product_row(Price=True), 2).quarantined


@pytest.mark.parametrize(
    ("jalali", "gregorian"),
    [("29 اسفند 1399", date(2021, 3, 19)),
     ("30 اسفند 1399", date(2021, 3, 20)),
     ("1 فروردین 1400", date(2021, 3, 21)),
     ("29 اسفند 1402", date(2024, 3, 19)),
     ("1 فروردین 1403", date(2024, 3, 20))],
)
def test_jalali_year_boundary_sequence(jalali: str, gregorian: date) -> None:
    parsed = parse_jalali_date(jalali)
    assert parsed.valid and parsed.value is not None
    assert parsed.value.gregorian == gregorian


def test_jalali_supported_year_range_is_enforced() -> None:
    low = parse_jalali_date(f"1 فروردین {MIN_JALALI_YEAR:04d}")
    high = parse_jalali_date(f"29 اسفند {MAX_JALALI_YEAR:04d}")
    assert low.valid and high.valid
    assert not parse_jalali_date("1 فروردین 0000").valid
    assert not parse_jalali_date("1 فروردین 9378").valid
    assert not parse_jalali_date("1 فروردين 1402").valid


def _round_trip(table: pa.Table, path: Path) -> pa.Table:
    pq.write_table(table, path)
    restored = pq.read_table(path)
    assert restored.schema == table.schema
    return restored


def test_all_declared_schemas_round_trip_through_parquet(tmp_path: Path) -> None:
    product_row_clean = {
        "product_id": INT64_MAX, "title_fa": "کالای فارسی", "category1": None,
        "category2": "دسته", "brand": None, "rate": None, "rate_count": 0,
        "sub_category": "نمونه", "is_unrated": True,
        "inconsistent_zero_rate": False, "core_attribute_conflict": False,
        "canonical_source_row_number": 2,
    }
    offer_row_clean = {
        "offer_id": "a" * 64, "product_id": INT64_MAX, "seller": "فروشنده",
        "price_raw": INT64_MAX, "price_toman": Decimal("922337203685477580.7"),
        "is_fake": False, "min_price_last_month": None,
        "missing_price_history": True, "invalid_price": False,
        "high_price_review": False, "source_row_number": 2,
    }
    comment_row_clean = {
        "comment_id": INT64_MAX, "product_id": 12, "title": None,
        "body": "متن فارسی", "created_at_raw": "23 تیر 1395",
        "created_at_jalali": "1395-04-23",
        "created_at_gregorian": date(2016, 7, 13), "rate": None,
        "is_unrated": True, "invalid_rate": False,
        "recommendation_status": "پیشنهاد می‌کنم", "is_buyer": True,
        "advantages": None, "disadvantages": None, "likes": 0, "dislikes": 0,
        "seller_title": None, "seller_code": "کد۰۰۱", "true_to_size_rate": None,
        "comment_id_conflict": False, "canonical_source_row_number": 2,
    }
    quarantine = transform_offer_row(product_row(Price="نامعتبر"), 5)
    audit = transform_comment_row(comment_row(likes="invalid"), 6)
    assert quarantine.quarantine_record is not None and audit.audit_records
    cases = (
        ("products", PRODUCTS_CLEAN_SCHEMA, [product_row_clean]),
        ("offers", OFFERS_CLEAN_SCHEMA, [offer_row_clean]),
        ("comments", COMMENTS_CLEAN_SCHEMA, [comment_row_clean]),
        ("quarantine", QUARANTINE_SCHEMA, [quarantine.quarantine_record]),
        ("row_audit", ROW_AUDIT_SCHEMA, list(audit.audit_records)),
    )
    restored = {}
    for name, schema, rows in cases:
        table = pa.Table.from_pylist(rows, schema=schema)
        restored[name] = _round_trip(table, tmp_path / f"{name}.parquet")
    assert restored["offers"]["price_toman"][0].as_py() == Decimal(
        "922337203685477580.7"
    )
    assert restored["comments"]["created_at_gregorian"].type == pa.date32()
    assert restored["products"]["product_id"][0].as_py() == INT64_MAX
    assert restored["offers"]["is_fake"][0].as_py() is False
    assert restored["comments"]["body"][0].as_py() == "متن فارسی"
    assert restored["comments"]["seller_code"][0].as_py() == "کد۰۰۱"
    assert restored["products"]["category1"][0].as_py() is None
