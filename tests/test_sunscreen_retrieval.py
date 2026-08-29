from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.cleaning import COMMENTS_CLEAN_SCHEMA, PRODUCTS_CLEAN_SCHEMA
from digikala_llm.sunscreen_builder import HISTORICAL_PRICE_LABEL, PRICE_SCHEMA
from digikala_llm.sunscreen_retrieval import (
    MAX_EXCERPT_CHARS,
    SunscreenLexicalIndex,
    normalize_persian,
    tokenize_persian,
)


def _index(tmp_path: Path) -> SunscreenLexicalIndex:
    products = []
    for product_id, title, brand in [(1, "ضد آفتاب پوست چرب", "برند الف"), (2, "ضد آفتاب بدون رنگ", "برند ب"), (3, "ضد آفتاب محبوب", "برند الف")]:
        products.append({"product_id": product_id, "title_fa": title, "category1": "مراقبت پوست", "category2": "کرم ضد آفتاب", "brand": brand, "rate": None, "rate_count": 0, "sub_category": None, "is_unrated": True, "inconsistent_zero_rate": False, "core_attribute_conflict": False, "canonical_source_row_number": product_id})
    pq.write_table(pa.Table.from_pylist(products, schema=PRODUCTS_CLEAN_SCHEMA), tmp_path / "sunscreen_products.parquet")
    pq.write_table(pa.Table.from_pylist([{"product_id": 1, "historical_price_inferred_irr": 100, "valid_price_offer_count": 1, "historical_price_label": HISTORICAL_PRICE_LABEL}, {"product_id": 2, "historical_price_inferred_irr": 200, "valid_price_offer_count": 1, "historical_price_label": HISTORICAL_PRICE_LABEL}, {"product_id": 3, "historical_price_inferred_irr": None, "valid_price_offer_count": 0, "historical_price_label": HISTORICAL_PRICE_LABEL}], schema=PRICE_SCHEMA), tmp_path / "sunscreen_prices.parquet")
    comments = []
    for comment_id, product_id, body, buyer in [(1, 1, "برای پوست چرب عالی است " * 30, True), (2, 2, "بدون رنگ و سبک", True), *[(n, 3, "محبوب", False) for n in range(3, 23)]]:
        comments.append({"comment_id": comment_id, "product_id": product_id, "title": None, "body": body, "created_at_raw": "1 فروردین 1400", "created_at_jalali": "1400-01-01", "created_at_gregorian": date(2021, 3, 21), "rate": Decimal(5), "is_unrated": False, "invalid_rate": False, "recommendation_status": "recommended", "is_buyer": buyer, "advantages": None, "disadvantages": None, "likes": None, "dislikes": None, "seller_title": None, "seller_code": None, "true_to_size_rate": None, "comment_id_conflict": False, "canonical_source_row_number": comment_id + 1})
    pq.write_table(pa.Table.from_pylist(comments, schema=COMMENTS_CLEAN_SCHEMA), tmp_path / "sunscreen_comments_canonical.parquet")
    (tmp_path / "_SUCCESS").write_text("\n")
    return SunscreenLexicalIndex(tmp_path)


def test_persian_normalization_and_deterministic_relevance_over_popularity(tmp_path: Path) -> None:
    assert normalize_persian("  كِتاب\u200cها ي ") == "کتاب ها ی"
    assert tokenize_persian("الف الف ب") == ("الف", "ب")
    index = _index(tmp_path)
    first, second = index.search("ضد آفتاب برای پوست چرب"), index.search("ضد آفتاب برای پوست چرب")
    assert first == second
    assert first["results"][0]["product_id"] == 1
    assert first["results"][0]["score_components"]["lexical_relevance"] > first["results"][-1]["score_components"]["lexical_relevance"]


def test_filters_evidence_traceability_labels_and_empty_queries(tmp_path: Path) -> None:
    index = _index(tmp_path)
    result = index.search("بدون رنگ", brand="برند ب", min_historical_price=150, max_historical_price=250, min_review_evidence=1)
    item = result["results"][0]
    assert item["product_id"] == 2 and item["historical_price_label"] == HISTORICAL_PRICE_LABEL
    assert item["evidence"][0]["comment_id"] == 2
    assert item["evidence"][0]["canonical_source_row_number"] == 3
    assert len(item["evidence"][0]["excerpt"]) <= MAX_EXCERPT_CHARS
    assert index.search("", brand="برند الف")["reason"] == "empty_query"
    assert index.search("ناشناخته")["reason"] == "no_matching_products"
    assert index.search("ضد آفتاب", min_historical_price=150)["results"][0]["product_id"] == 2
    assert "spf" not in item and "skin_type" not in item and "medical_suitability" not in item
