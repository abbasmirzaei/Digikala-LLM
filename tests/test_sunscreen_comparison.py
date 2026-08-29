from types import SimpleNamespace

import pytest

from digikala_llm.sunscreen_builder import HISTORICAL_PRICE_LABEL
from digikala_llm.sunscreen_comparison import SunscreenComparisonService
from digikala_llm.sunscreen_retrieval import MAX_EXCERPT_CHARS, tokenize_persian


def _comment(comment_id: int, body: str, rate: float, buyer: bool, status: str, row: int) -> dict[str, object]:
    return {"comment_id": comment_id, "body": body, "title": None, "rate": rate, "is_buyer": buyer, "recommendation_status": status, "canonical_source_row_number": row, "tokens": frozenset(tokenize_persian(body))}


def _service() -> SunscreenComparisonService:
    products = {
        1: {"product_id": 1, "title": "ضد آفتاب الف", "brand": "الف", "price": {"historical_price_inferred_irr": 100}, "comments": (_comment(1, "سبک و عالی برای پوست چرب " * 30, 5, True, "recommended", 11), _comment(2, "گران و بد", 1, False, "not_recommended", 12))},
        2: {"product_id": 2, "title": "ضد آفتاب ب", "brand": "ب", "price": {"historical_price_inferred_irr": 200}, "comments": (_comment(3, "سبک و خوب", 4, True, "recommended", 13),)},
        3: {"product_id": 3, "title": "ضد آفتاب ج", "brand": "ج", "price": {"historical_price_inferred_irr": None}, "comments": ()},
        4: {"product_id": 4, "title": "ضد آفتاب د", "brand": "د", "price": {"historical_price_inferred_irr": 100}, "comments": (_comment(4, "ضعیف", 2, False, "not_recommended", 14),)},
    }
    return SunscreenComparisonService(SimpleNamespace(products=products))


def test_comparison_validation_and_determinism() -> None:
    service = _service()
    with pytest.raises(ValueError, match="two to four"):
        service.compare([1])
    with pytest.raises(ValueError, match="unique"):
        service.compare([1, 1])
    with pytest.raises(KeyError, match="unknown"):
        service.compare([1, 99])
    assert service.compare([1, 2], query="سبک") == service.compare([1, 2], query="سبک")
    assert len(service.compare([1, 2, 3, 4])["products"]) == 4


def test_aggregates_winners_and_traceable_bounded_evidence() -> None:
    result = _service().compare([1, 2], query="پوست چرب")
    first = result["products"][0]
    assert first["historical_price_label"] == HISTORICAL_PRICE_LABEL
    assert first["buyer_review_count"] == 1
    assert first["buyer_review_percentage"] == 50
    assert first["valid_average_rating"] == 3
    assert first["recommendation_status_distribution"] == {"not_recommended": 1, "recommended": 1}
    assert first["price_difference_vs_compared_inferred_irr"] == {"2": -100}
    assert result["winner_indicators"]["lower_historical_price"]["winner_product_ids"] == [1]
    assert result["overall_winner"] is None
    assert result["products"][0]["positive_evidence"][0]["comment_id"] == 1
    critical = first["critical_evidence"][0]
    assert critical["comment_id"] == 2 and critical["canonical_source_row_number"] == 12
    assert critical["evidence_type"] == "user_review_evidence"
    for evidence in first["positive_evidence"] + first["critical_evidence"]:
        assert len(evidence["excerpt"]) <= MAX_EXCERPT_CHARS
    assert "skin_type" not in first and "medical_suitability" not in first


def test_query_prioritizes_evidence_without_creating_overall_winner() -> None:
    result = _service().compare([1, 2], query="پوست چرب", priority="lower_historical_price")
    assert result["priority_winner_product_ids"] == [1]
    assert result["overall_winner"] is None
    assert result["products"][0]["positive_evidence"][0]["matched_query_tokens"] == ["پوست", "چرب"]
