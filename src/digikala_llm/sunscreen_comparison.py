"""Evidence-backed deterministic comparison of two to four scoped sunscreen products."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from digikala_llm.sunscreen_builder import HISTORICAL_PRICE_LABEL
from digikala_llm.sunscreen_retrieval import (
    DEFAULT_DATA_DIR,
    MAX_EXCERPTS,
    SunscreenLexicalIndex,
    _excerpt,
    normalize_persian,
    tokenize_persian,
)

_CRITICAL_TERMS = frozenset({"بد", "ضعیف", "ناراضی", "گران", "مشکل", "سوزش", "قرمزی", "سنگین"})
_PRIORITIES = {
    "lower_historical_price",
    "more_review_evidence",
    "higher_valid_average_rating",
    "higher_positive_recommendation_share",
}
COMPARISON_SCHEMA = {
    "products": "product metadata, aggregates, per-field price differences, and user-review evidence",
    "winner_indicators": "field-level tied winners only; no overall winner without priority",
    "evidence": "bounded excerpts with comment_id and canonical_source_row_number",
}


def _status_key(value: str | None) -> str:
    return value if value is not None else "unknown"


def _is_positive(comment: dict[str, Any]) -> bool:
    rate = comment["rate"]
    return (rate is not None and 4 <= float(rate) <= 5) or normalize_persian(comment["recommendation_status"]) == "recommended"


def _is_critical(comment: dict[str, Any]) -> bool:
    rate = comment["rate"]
    status = normalize_persian(comment["recommendation_status"])
    text_tokens = comment["tokens"]
    return (rate is not None and 0 < float(rate) <= 2) or status in {"not recommended", "not_recommended"} or bool(text_tokens & _CRITICAL_TERMS)


def _evidence(comment: dict[str, Any], query_terms: set[str], kind: str) -> dict[str, Any]:
    return {
        "evidence_type": "user_review_evidence",
        "polarity": kind,
        "comment_id": comment["comment_id"],
        "canonical_source_row_number": comment["canonical_source_row_number"],
        "is_buyer": comment["is_buyer"],
        "matched_query_tokens": sorted(comment["tokens"] & query_terms),
        "excerpt": _excerpt(comment["body"] or comment["title"] or "", query_terms),
    }


class SunscreenComparisonService:
    """Comparison layer over ``SunscreenLexicalIndex``; it reads no raw data itself."""

    def __init__(self, index: SunscreenLexicalIndex) -> None:
        self.index = index

    def compare(
        self, product_ids: list[int], *, query: str | None = None, priority: str | None = None
    ) -> dict[str, Any]:
        if not 2 <= len(product_ids) <= 4:
            raise ValueError("comparison requires two to four product IDs")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("comparison product IDs must be unique")
        unknown = sorted(set(product_ids) - self.index.products.keys())
        if unknown:
            raise KeyError(f"unknown scoped product IDs: {unknown}")
        if priority is not None and priority not in _PRIORITIES:
            raise ValueError(f"unsupported comparison priority: {priority}")
        query_terms = set(tokenize_persian(query))
        products = [self._product(self.index.products[product_id], query_terms) for product_id in product_ids]
        prices = {item["product_id"]: item["historical_price_inferred_irr"] for item in products}
        for item in products:
            item["price_difference_vs_compared_inferred_irr"] = {
                str(other): None if item["historical_price_inferred_irr"] is None or price is None else item["historical_price_inferred_irr"] - price
                for other, price in prices.items()
                if other != item["product_id"]
            }
        winners = self._winners(products)
        response: dict[str, Any] = {
            "comparison_schema": COMPARISON_SCHEMA,
            "comparison_query": query,
            "normalized_query": normalize_persian(query),
            "product_ids": product_ids,
            "products": products,
            "winner_indicators": winners,
            "overall_winner": None,
        }
        if priority is not None:
            response["priority"] = priority
            response["priority_winner_product_ids"] = winners[priority]["winner_product_ids"]
        return response

    def _product(self, product: dict[str, Any], query_terms: set[str]) -> dict[str, Any]:
        comments = product["comments"]
        valid_ratings = [float(comment["rate"]) for comment in comments if comment["rate"] is not None and 0 < float(comment["rate"]) <= 5]
        statuses = Counter(_status_key(comment["recommendation_status"]) for comment in comments)
        known_statuses = sum(count for status, count in statuses.items() if status != "unknown")
        positive_share = statuses["recommended"] / known_statuses if known_statuses else None
        positives = [comment for comment in comments if _is_positive(comment)]
        critical = [comment for comment in comments if _is_critical(comment)]
        rank = lambda comment: (-len(comment["tokens"] & query_terms), -(comment["is_buyer"] is True), comment["comment_id"])
        price_row = product["price"]
        return {
            "product_id": product["product_id"],
            "title": product["title"],
            "brand": product["brand"],
            "historical_price_inferred_irr": None if price_row is None else price_row["historical_price_inferred_irr"],
            "historical_price_label": HISTORICAL_PRICE_LABEL,
            "canonical_review_count": len(comments),
            "buyer_review_count": sum(comment["is_buyer"] is True for comment in comments),
            "buyer_review_percentage": 100 * sum(comment["is_buyer"] is True for comment in comments) / len(comments) if comments else 0.0,
            "valid_average_rating": sum(valid_ratings) / len(valid_ratings) if valid_ratings else None,
            "valid_rating_count": len(valid_ratings),
            "recommendation_status_distribution": dict(sorted(statuses.items())),
            "positive_recommendation_share": positive_share,
            "positive_evidence": [_evidence(comment, query_terms, "positive") for comment in sorted(positives, key=rank)[:MAX_EXCERPTS]],
            "critical_evidence": [_evidence(comment, query_terms, "critical") for comment in sorted(critical, key=rank)[:MAX_EXCERPTS]],
        }

    @staticmethod
    def _winners(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        def winners(field: str, reverse: bool = True) -> dict[str, Any]:
            values = [item for item in products if item[field] is not None]
            if not values:
                return {"winner_product_ids": [], "value": None}
            value = (max if reverse else min)(item[field] for item in values)
            return {"winner_product_ids": [item["product_id"] for item in values if item[field] == value], "value": value}
        return {
            "lower_historical_price": winners("historical_price_inferred_irr", reverse=False),
            "more_review_evidence": winners("canonical_review_count"),
            "higher_valid_average_rating": winners("valid_average_rating"),
            "higher_positive_recommendation_share": winners("positive_recommendation_share"),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two to four sunscreen products with user-review evidence.")
    parser.add_argument("product_ids", nargs="+", type=int)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--query")
    parser.add_argument("--priority", choices=sorted(_PRIORITIES))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    result = SunscreenComparisonService(SunscreenLexicalIndex(args.data_dir)).compare(args.product_ids, query=args.query, priority=args.priority)
    print(json.dumps({"latency_ms": round((time.perf_counter() - started) * 1000, 3), **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
