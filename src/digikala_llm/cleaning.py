"""Typed schemas and deterministic row transformations for cleaning milestone 1.

This module intentionally contains no cross-row state, deduplication, canonical selection,
referential-integrity checks, file writing, or CLI behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Generic, TypeVar

import pyarrow as pa

INT64_MAX = 2**63 - 1
MONEY_TYPE = pa.decimal128(20, 1)
MIN_JALALI_YEAR = 1
MAX_JALALI_YEAR = 9377
PERSIAN_MONTHS = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)
_MONTH_NUMBER = {name: number for number, name in enumerate(PERSIAN_MONTHS, start=1)}
_JALALI_PATTERN = re.compile(
    rf"^([0-9]{{1,2}}) ({'|'.join(map(re.escape, PERSIAN_MONTHS))}) ([0-9]{{4}})$"
)

PRODUCTS_CLEAN_SCHEMA = pa.schema(
    [
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("title_fa", pa.string()),
        pa.field("category1", pa.string()),
        pa.field("category2", pa.string()),
        pa.field("brand", pa.string()),
        pa.field("rate", pa.float64()),
        pa.field("rate_count", pa.int64()),
        pa.field("sub_category", pa.string()),
        pa.field("is_unrated", pa.bool_(), nullable=False),
        pa.field("inconsistent_zero_rate", pa.bool_(), nullable=False),
        pa.field("core_attribute_conflict", pa.bool_(), nullable=False),
        pa.field("canonical_source_row_number", pa.int64(), nullable=False),
    ]
)

OFFERS_CLEAN_SCHEMA = pa.schema(
    [
        pa.field("offer_id", pa.string(), nullable=False),
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("seller", pa.string()),
        pa.field("price_raw", pa.int64()),
        pa.field("price_toman", MONEY_TYPE),
        pa.field("is_fake", pa.bool_()),
        pa.field("min_price_last_month", pa.int64()),
        pa.field("missing_price_history", pa.bool_(), nullable=False),
        pa.field("invalid_price", pa.bool_(), nullable=False),
        pa.field("high_price_review", pa.bool_(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
    ]
)

COMMENTS_CLEAN_SCHEMA = pa.schema(
    [
        pa.field("comment_id", pa.int64(), nullable=False),
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("body", pa.string()),
        pa.field("created_at_raw", pa.string(), nullable=False),
        pa.field("created_at_jalali", pa.string(), nullable=False),
        pa.field("created_at_gregorian", pa.date32(), nullable=False),
        pa.field("rate", pa.float64()),
        pa.field("is_unrated", pa.bool_(), nullable=False),
        pa.field("invalid_rate", pa.bool_(), nullable=False),
        pa.field("recommendation_status", pa.string()),
        pa.field("is_buyer", pa.bool_()),
        pa.field("advantages", pa.string()),
        pa.field("disadvantages", pa.string()),
        pa.field("likes", pa.int64()),
        pa.field("dislikes", pa.int64()),
        pa.field("seller_title", pa.string()),
        pa.field("seller_code", pa.string()),
        pa.field("true_to_size_rate", pa.string()),
        pa.field("comment_id_conflict", pa.bool_(), nullable=False),
        pa.field("canonical_source_row_number", pa.int64(), nullable=False),
    ]
)

# Milestone-1 transforms emit candidates, not final clean rows. Fields whose truth requires
# cross-row or dataset-level work are deliberately absent, and provenance remains explicit.
PRODUCT_ROW_CANDIDATE_SCHEMA = pa.schema(
    [field for field in PRODUCTS_CLEAN_SCHEMA if field.name not in {
        "core_attribute_conflict", "canonical_source_row_number"
    }]
    + [pa.field("source_row_number", pa.int64(), nullable=False)]
)
OFFER_ROW_CANDIDATE_SCHEMA = pa.schema(
    [field for field in OFFERS_CLEAN_SCHEMA if field.name != "high_price_review"]
)
COMMENT_ROW_CANDIDATE_SCHEMA = pa.schema(
    [field for field in COMMENTS_CLEAN_SCHEMA if field.name not in {
        "comment_id_conflict", "canonical_source_row_number"
    }]
    + [pa.field("source_row_number", pa.int64(), nullable=False)]
)

QUARANTINE_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("entity_id", pa.int64()),
        pa.field("rule_ids", pa.list_(pa.string()), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("raw_record_json", pa.string(), nullable=False),
    ]
)

ROW_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("entity_id", pa.int64()),
        pa.field("field", pa.string(), nullable=False),
        pa.field("raw_value", pa.string()),
        pa.field("rule_id", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("detail", pa.string()),
        pa.field("is_lossy", pa.bool_(), nullable=False),
    ]
)

PRODUCT_CONFLICT_SCHEMA = pa.schema(
    [
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("title_fa", pa.string()),
        pa.field("category1", pa.string()),
        pa.field("category2", pa.string()),
        pa.field("brand", pa.string()),
        pa.field("rate", pa.float64()),
        pa.field("rate_count", pa.int64()),
        pa.field("sub_category", pa.string()),
        pa.field("is_unrated", pa.bool_(), nullable=False),
        pa.field("inconsistent_zero_rate", pa.bool_(), nullable=False),
        pa.field("candidate_source_row_number", pa.int64(), nullable=False),
        pa.field("canonical_source_row_number", pa.int64(), nullable=False),
        pa.field("core_completeness", pa.int64(), nullable=False),
        pa.field("core_digest", pa.string(), nullable=False),
        pa.field("raw_core_json", pa.string(), nullable=False),
        pa.field("selected_as_canonical", pa.bool_(), nullable=False),
    ]
)

T = TypeVar("T")


@dataclass(frozen=True)
class ValueResult(Generic[T]):
    """Result of a strict scalar conversion."""

    value: T | None
    valid: bool
    raw_value: str | None
    reason: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    field: str
    raw_value: str | None
    rule_id: str
    detail: str | None = None


@dataclass(frozen=True)
class RuleDefinition:
    action: str
    severity: str
    is_lossy: bool
    row_level: bool


RULE_CATALOG = {
    "ID-002": RuleDefinition("quarantine row", "error", True, True),
    "ID-003": RuleDefinition("set invalid optional identifier to null", "warning", True, True),
    "BOOL-002": RuleDefinition("quarantine row", "error", True, True),
    "TXT-001": RuleDefinition("set blank text to null", "info", True, False),
    "PRD-004": RuleDefinition("quarantine product candidate", "error", True, True),
    "PRD-005": RuleDefinition("set product rate to null as unrated", "info", True, False),
    "PRD-006": RuleDefinition("preserve zero product rate and flag", "warning", False, True),
    "PRD-007": RuleDefinition("quarantine product candidate", "error", True, True),
    "OFF-003": RuleDefinition("set zero price to null and flag", "warning", True, True),
    "OFF-004": RuleDefinition("quarantine offer", "error", True, True),
    "OFF-007": RuleDefinition("set unavailable price history to null", "info", True, False),
    "OFF-008": RuleDefinition("quarantine offer with invalid price history", "error", True, True),
    "COM-005": RuleDefinition("set comment rate to null as unrated", "info", True, False),
    "COM-006": RuleDefinition("set invalid comment rate to null", "warning", True, True),
    "COM-009": RuleDefinition("retain comment with missing body", "info", False, False),
    "COM-010": RuleDefinition("preserve opaque seller code", "info", False, False),
    "COM-011": RuleDefinition("retain missing seller code as null", "info", False, False),
    "COM-012": RuleDefinition("set invalid optional count to null", "warning", True, True),
    "COM-013": RuleDefinition("set explicit seller sentinel to null", "info", True, False),
    "DATE-003": RuleDefinition("quarantine comment", "error", True, True),
}


@dataclass(frozen=True)
class RowTransformResult:
    """A transformed row candidate plus row-level audit/quarantine records."""

    candidate_row: dict[str, Any] | None
    audit_records: tuple[dict[str, Any], ...]
    aggregate_counter_keys: tuple[tuple[str, str], ...]
    quarantine_record: dict[str, Any] | None

    def __post_init__(self) -> None:
        accepted = self.candidate_row is not None and self.quarantine_record is None
        quarantined = self.candidate_row is None and self.quarantine_record is not None
        if not (accepted or quarantined):
            raise ValueError("result must be exactly one of accepted or quarantined")

    @property
    def quarantined(self) -> bool:
        return self.quarantine_record is not None

    @property
    def is_final_clean_row(self) -> bool:
        """Milestone-1 results are never final dataset-level clean rows."""
        return False


@dataclass(frozen=True)
class JalaliDateResult:
    raw: str
    canonical_jalali: str
    gregorian: date


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _raw_string(value: object) -> str | None:
    return None if _is_missing(value) else str(value)


def clean_text(value: object) -> str | None:
    """Convert blank text to null while preserving every nonblank code point exactly."""
    if _is_missing(value):
        return None
    text = str(value)
    return None if text.strip() == "" else text


def _clean_text_field(
    raw: dict[str, object],
    field: str,
    events: list[AuditEvent],
    counters: list[tuple[str, str]],
) -> str | None:
    value = raw.get(field)
    cleaned = clean_text(value)
    if isinstance(value, str) and value.strip() == "":
        _record_event(events, counters, field, value, "TXT-001")
    return cleaned


def _record_event(
    events: list[AuditEvent],
    counters: list[tuple[str, str]],
    field: str,
    raw_value: str | None,
    rule_id: str,
    detail: str | None = None,
) -> None:
    """Route a stable rule occurrence to traceability records or aggregate counters."""
    rule = RULE_CATALOG[rule_id]
    if rule.row_level:
        events.append(AuditEvent(field, raw_value, rule_id, detail))
    else:
        counters.append((rule_id, field))


def _decimal(value: object) -> Decimal | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def parse_required_id(value: object) -> ValueResult[int]:
    """Parse a non-negative integral signed-int64 source identifier."""
    raw = _raw_string(value)
    number = _decimal(value)
    if number is None:
        return ValueResult(None, False, raw, "missing, blank, or non-numeric identifier")
    if number != number.to_integral_value():
        return ValueResult(None, False, raw, "fractional identifier")
    integer = int(number)
    if integer < 0:
        return ValueResult(None, False, raw, "negative identifier")
    if integer > INT64_MAX:
        return ValueResult(None, False, raw, "identifier exceeds signed int64")
    return ValueResult(integer, True, raw)


def parse_strict_boolean(value: object) -> ValueResult[bool]:
    """Accept only the two observed semantic values, without truthiness coercion."""
    raw = _raw_string(value)
    if value is True or value == "True":
        return ValueResult(True, True, raw)
    if value is False or value == "False":
        return ValueResult(False, True, raw)
    return ValueResult(None, False, raw, "unrecognized boolean; expected True or False")


def _parse_nonnegative_int(value: object, *, allow_missing: bool) -> ValueResult[int]:
    raw = _raw_string(value)
    if (_is_missing(value) or (isinstance(value, str) and value.strip() == "")) and allow_missing:
        return ValueResult(None, True, raw)
    number = _decimal(value)
    if number is None:
        return ValueResult(None, False, raw, "missing or non-numeric integer")
    if number != number.to_integral_value():
        return ValueResult(None, False, raw, "fractional integer")
    integer = int(number)
    if integer < 0 or integer > INT64_MAX:
        return ValueResult(None, False, raw, "integer outside non-negative signed-int64 range")
    return ValueResult(integer, True, raw)


def _parse_product_rate(value: object) -> ValueResult[Decimal]:
    raw = _raw_string(value)
    number = _decimal(value)
    if number is None or number < 0 or number > 100:
        return ValueResult(None, False, raw, "product Rate must be numeric and within 0–100")
    return ValueResult(number, True, raw)


def _parse_comment_rate(value: object) -> tuple[float | None, bool, bool, str | None]:
    if _is_missing(value) or (isinstance(value, str) and value.strip() == ""):
        return None, False, False, None
    number = _decimal(value)
    if number is None or number < 1 or number > 5:
        if number == 0:
            return None, True, False, "COM-005"
        return None, False, True, "COM-006"
    return float(number), False, False, None


def _jalali_to_gregorian(year: int, month: int, day: int) -> tuple[int, int, int]:
    """Convert Solar Hijri to Gregorian with the established Jalaali integer algorithm."""
    adjusted_year = year + 1595
    days = (
        -355668
        + 365 * adjusted_year
        + (adjusted_year // 33) * 8
        + ((adjusted_year % 33 + 3) // 4)
        + day
    )
    days += (month - 1) * 31 if month < 7 else (month - 7) * 30 + 186
    gregorian_year = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gregorian_year += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gregorian_year += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gregorian_year += (days - 1) // 365
        days = (days - 1) % 365
    gregorian_day = days + 1
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    is_leap = gregorian_year % 4 == 0 and (
        gregorian_year % 100 != 0 or gregorian_year % 400 == 0
    )
    if is_leap:
        month_days[2] = 29
    gregorian_month = 1
    while gregorian_day > month_days[gregorian_month]:
        gregorian_day -= month_days[gregorian_month]
        gregorian_month += 1
    return gregorian_year, gregorian_month, gregorian_day


def _esfand_days(year: int) -> int:
    this_new_year = date(*_jalali_to_gregorian(year, 1, 1))
    next_new_year = date(*_jalali_to_gregorian(year + 1, 1, 1))
    return 30 if (next_new_year - this_new_year).days == 366 else 29


def parse_jalali_date(value: object) -> ValueResult[JalaliDateResult]:
    """Parse the exact source form for supported Jalali years 1 through 9377."""
    raw = _raw_string(value)
    if raw is None:
        return ValueResult(None, False, raw, "missing Jalali date")
    match = _JALALI_PATTERN.fullmatch(raw)
    if match is None:
        return ValueResult(None, False, raw, "expected D <Persian Jalali month name> YYYY")
    day = int(match.group(1))
    month = _MONTH_NUMBER[match.group(2)]
    year = int(match.group(3))
    if not MIN_JALALI_YEAR <= year <= MAX_JALALI_YEAR:
        return ValueResult(None, False, raw, "Jalali year is outside supported range 1–9377")
    try:
        maximum_day = 31 if month <= 6 else 30 if month <= 11 else _esfand_days(year)
    except (OverflowError, ValueError):
        return ValueResult(None, False, raw, "Jalali year is outside supported date range")
    if not 1 <= day <= maximum_day:
        return ValueResult(None, False, raw, "invalid day for Jalali month and year")
    try:
        gregorian = date(*_jalali_to_gregorian(year, month, day))
    except (OverflowError, ValueError):
        return ValueResult(None, False, raw, "Jalali year is outside supported date range")
    result = JalaliDateResult(raw, f"{year:04d}-{month:02d}-{day:02d}", gregorian)
    return ValueResult(result, True, raw)


def _audit_records(
    dataset: str,
    source_row_number: int,
    entity_id: int | None,
    events: list[AuditEvent],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "dataset": dataset,
            "source_file": _source_file(dataset),
            "source_row_number": source_row_number,
            "entity_id": entity_id,
            "field": event.field,
            "raw_value": event.raw_value,
            "rule_id": event.rule_id,
            "action": RULE_CATALOG[event.rule_id].action,
            "severity": RULE_CATALOG[event.rule_id].severity,
            "detail": event.detail,
            "is_lossy": RULE_CATALOG[event.rule_id].is_lossy,
        }
        for event in events
    )


def _safe_raw_record(raw: dict[str, object]) -> str:
    serializable = {key: _raw_string(value) for key, value in raw.items()}
    return json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_file(dataset: str) -> str:
    if dataset == "comments":
        return "data/raw/digikala-comments.csv"
    return "data/raw/digikala-products.csv"


def _quarantine(
    dataset: str,
    source_row_number: int,
    entity_id: int | None,
    raw: dict[str, object],
    events: list[AuditEvent],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_file": _source_file(dataset),
        "source_row_number": source_row_number,
        "entity_id": entity_id,
        "rule_ids": sorted({event.rule_id for event in events}),
        "reason": "; ".join(
            f"{event.rule_id}: {RULE_CATALOG[event.rule_id].action}"
            + (f" ({event.detail})" if event.detail else "")
            for event in events
        ),
        "raw_record_json": _safe_raw_record(raw),
    }


def _finish_row(
    dataset: str,
    source_row_number: int,
    entity_id: int | None,
    raw: dict[str, object],
    row: dict[str, Any],
    events: list[AuditEvent],
    counters: list[tuple[str, str]],
    fatal: bool,
) -> RowTransformResult:
    audit = _audit_records(dataset, source_row_number, entity_id, events)
    quarantine = _quarantine(dataset, source_row_number, entity_id, raw, events) if fatal else None
    return RowTransformResult(None if fatal else row, audit, tuple(counters), quarantine)


def transform_product_row(raw: dict[str, object], source_row_number: int) -> RowTransformResult:
    """Transform one raw product-fact candidate; no canonical selection is performed."""
    events: list[AuditEvent] = []
    counters: list[tuple[str, str]] = []
    product_id = parse_required_id(raw.get("id"))
    rate_count = _parse_nonnegative_int(raw.get("Rate_cnt"), allow_missing=False)
    rate = _parse_product_rate(raw.get("Rate"))
    for field, parsed, rule in (
        ("id", product_id, "ID-002"),
        ("Rate_cnt", rate_count, "PRD-007"),
        ("Rate", rate, "PRD-004"),
    ):
        if not parsed.valid:
            _record_event(events, counters, field, parsed.raw_value, rule, parsed.reason)
    fatal = any(not parsed.valid for parsed in (product_id, rate_count, rate))
    rate_value: float | None = None
    is_unrated = False
    inconsistent = False
    if rate.valid and rate_count.valid:
        if rate.value == 0 and rate_count.value == 0:
            is_unrated = True
            _record_event(events, counters, "Rate", rate.raw_value, "PRD-005")
        else:
            rate_value = float(rate.value) if rate.value is not None else None
            inconsistent = rate.value == 0 and (rate_count.value or 0) > 0
            if inconsistent:
                _record_event(events, counters, "Rate", rate.raw_value, "PRD-006")
    row = {
        "product_id": product_id.value,
        "title_fa": _clean_text_field(raw, "title_fa", events, counters),
        "category1": _clean_text_field(raw, "Category1", events, counters),
        "category2": _clean_text_field(raw, "Category2", events, counters),
        "brand": _clean_text_field(raw, "Brand", events, counters),
        "rate": rate_value,
        "rate_count": rate_count.value,
        "sub_category": _clean_text_field(raw, "sub_category", events, counters),
        "is_unrated": is_unrated,
        "inconsistent_zero_rate": inconsistent,
        "source_row_number": source_row_number,
    }
    return _finish_row(
        "products", source_row_number, product_id.value, raw, row, events, counters, fatal
    )


def _offer_id(raw: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for field in ("id", "Seller", "Price", "Is_Fake", "min_price_last_month"):
        value = _raw_string(raw.get(field))
        if value is None:
            digest.update(b"N")
        else:
            encoded = value.encode("utf-8")
            digest.update(b"V")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def transform_offer_row(raw: dict[str, object], source_row_number: int) -> RowTransformResult:
    """Transform one seller-offer candidate; p99.9 review flagging remains dataset-level."""
    events: list[AuditEvent] = []
    counters: list[tuple[str, str]] = []
    product_id = parse_required_id(raw.get("id"))
    is_fake = parse_strict_boolean(raw.get("Is_Fake"))
    price = _parse_nonnegative_int(raw.get("Price"), allow_missing=False)
    history = _parse_nonnegative_int(raw.get("min_price_last_month"), allow_missing=True)
    for field, parsed, rule in (
        ("id", product_id, "ID-002"),
        ("Is_Fake", is_fake, "BOOL-002"),
        ("Price", price, "OFF-004"),
        ("min_price_last_month", history, "OFF-008"),
    ):
        if not parsed.valid:
            _record_event(events, counters, field, parsed.raw_value, rule, parsed.reason)
    fatal = any(not parsed.valid for parsed in (product_id, is_fake, price, history))
    invalid_price = price.valid and price.value == 0
    price_raw = None if invalid_price else price.value
    if invalid_price:
        _record_event(events, counters, "Price", price.raw_value, "OFF-003")
    missing_history = history.valid and history.value == 0
    history_value = None if history.value is None or missing_history else history.value
    if missing_history:
        _record_event(
            events, counters, "min_price_last_month", history.raw_value, "OFF-007"
        )
    price_toman = Decimal(price_raw).scaleb(-1) if price_raw is not None else None
    row = {
        "offer_id": _offer_id(raw),
        "product_id": product_id.value,
        "seller": _clean_text_field(raw, "Seller", events, counters),
        "price_raw": price_raw,
        "price_toman": price_toman,
        "is_fake": is_fake.value,
        "min_price_last_month": history_value,
        "missing_price_history": missing_history,
        "invalid_price": invalid_price,
        "source_row_number": source_row_number,
    }
    return _finish_row(
        "offers", source_row_number, product_id.value, raw, row, events, counters, fatal
    )


def _optional_count(
    raw: dict[str, object],
    field: str,
    events: list[AuditEvent],
    counters: list[tuple[str, str]],
) -> int | None:
    result = _parse_nonnegative_int(raw.get(field), allow_missing=True)
    if not result.valid:
        _record_event(events, counters, field, result.raw_value, "COM-012", result.reason)
    return result.value


def _seller_text(
    raw: dict[str, object],
    field: str,
    events: list[AuditEvent],
    counters: list[tuple[str, str]],
) -> str | None:
    value = raw.get(field)
    if field == "seller_code" and (_is_missing(value) or value == ""):
        _record_event(events, counters, field, _raw_string(value), "COM-011")
        return None
    if value == "nan":
        _record_event(events, counters, field, "nan", "COM-013")
        return None
    text = _clean_text_field(raw, field, events, counters)
    if field == "seller_code" and text is not None:
        _record_event(events, counters, field, text, "COM-010")
    return text


def transform_comment_row(raw: dict[str, object], source_row_number: int) -> RowTransformResult:
    """Transform one comment candidate; ID conflict and orphan handling remain out of scope."""
    events: list[AuditEvent] = []
    counters: list[tuple[str, str]] = []
    comment_id = parse_required_id(raw.get("id"))
    product_id = parse_required_id(raw.get("product_id"))
    buyer = parse_strict_boolean(raw.get("is_buyer"))
    created = parse_jalali_date(raw.get("created_at"))
    for field, parsed, rule in (
        ("id", comment_id, "ID-002"),
        ("product_id", product_id, "ID-002"),
        ("is_buyer", buyer, "BOOL-002"),
        ("created_at", created, "DATE-003"),
    ):
        if not parsed.valid:
            _record_event(events, counters, field, parsed.raw_value, rule, parsed.reason)
    fatal = any(not parsed.valid for parsed in (comment_id, product_id, buyer, created))
    rate, is_unrated, invalid_rate, rate_rule = _parse_comment_rate(raw.get("rate"))
    if rate_rule is not None:
        _record_event(events, counters, "rate", _raw_string(raw.get("rate")), rate_rule)
    likes = _optional_count(raw, "likes", events, counters)
    dislikes = _optional_count(raw, "dislikes", events, counters)
    seller_code = _seller_text(raw, "seller_code", events, counters)
    seller_title = _seller_text(raw, "seller_title", events, counters)
    parsed_date = created.value
    body = _clean_text_field(raw, "body", events, counters)
    if body is None:
        _record_event(events, counters, "body", _raw_string(raw.get("body")), "COM-009")
    row = {
        "comment_id": comment_id.value,
        "product_id": product_id.value,
        "title": _clean_text_field(raw, "title", events, counters),
        "body": body,
        "created_at_raw": parsed_date.raw if parsed_date else None,
        "created_at_jalali": parsed_date.canonical_jalali if parsed_date else None,
        "created_at_gregorian": parsed_date.gregorian if parsed_date else None,
        "rate": rate,
        "is_unrated": is_unrated,
        "invalid_rate": invalid_rate,
        "recommendation_status": _clean_text_field(
            raw, "recommendation_status", events, counters
        ),
        "is_buyer": buyer.value,
        "advantages": _clean_text_field(raw, "advantages", events, counters),
        "disadvantages": _clean_text_field(raw, "disadvantages", events, counters),
        "likes": likes,
        "dislikes": dislikes,
        "seller_title": seller_title,
        "seller_code": seller_code,
        "true_to_size_rate": _clean_text_field(
            raw, "true_to_size_rate", events, counters
        ),
        "source_row_number": source_row_number,
    }
    return _finish_row(
        "comments", source_row_number, comment_id.value, raw, row, events, counters, fatal
    )
