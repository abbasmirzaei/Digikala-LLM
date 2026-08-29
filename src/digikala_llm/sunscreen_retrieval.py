"""Deterministic CPU-only Persian lexical product retrieval over published sunscreen data."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from digikala_llm.sunscreen_builder import HISTORICAL_PRICE_LABEL

DEFAULT_DATA_DIR = Path("data/processed/sunscreen_mvp/v1")
MAX_EXCERPTS, MAX_EXCERPT_CHARS = 3, 240
_ARABIC = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})
_DIACRITICS, _NON_WORD, _SPACE = re.compile(r"[\u064b-\u065f\u0670\u0640]"), re.compile(r"[^\w\u0600-\u06ff]+"), re.compile(r"\s+")
SCORING_FORMULA = "100*lexical_relevance + min(8,evidence)*1.5 + min(6,buyer_evidence) + rating_strength + coverage; non-lexical components are capped."


def normalize_persian(value: str | None) -> str:
    if value is None:
        return ""
    value = _DIACRITICS.sub("", value.translate(_ARABIC)).replace("\u200c", " ")
    return _SPACE.sub(" ", _NON_WORD.sub(" ", value.casefold())).strip()


def tokenize_persian(value: str | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_persian(value).split()))


def _excerpt(text: str, terms: set[str]) -> str:
    words = text.split()
    start = next((max(0, i - 8) for i, word in enumerate(words) if set(tokenize_persian(word)) & terms), 0)
    excerpt = " ".join(words[start:])
    return excerpt if len(excerpt) <= MAX_EXCERPT_CHARS else excerpt[: MAX_EXCERPT_CHARS - 1].rstrip() + "…"


class SunscreenLexicalIndex:
    """In-memory index of only published scoped products, prices, and canonical comments."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        started = time.perf_counter()
        self.data_dir = Path(data_dir)
        required = ("sunscreen_products.parquet", "sunscreen_prices.parquet", "sunscreen_comments_canonical.parquet", "_SUCCESS")
        missing = [name for name in required if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"published sunscreen artifacts missing: {missing}")
        prices = {row["product_id"]: row for row in pq.read_table(self.data_dir / "sunscreen_prices.parquet").to_pylist()}
        comments: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in pq.read_table(self.data_dir / "sunscreen_comments_canonical.parquet").to_pylist():
            row["tokens"] = frozenset(tokenize_persian(f"{row['title'] or ''} {row['body'] or ''}"))
            comments[row["product_id"]].append(row)
        self.products = {}
        for row in pq.read_table(self.data_dir / "sunscreen_products.parquet").to_pylist():
            product_id = row["product_id"]
            self.products[product_id] = {"product_id": product_id, "title": row["title_fa"], "brand": row["brand"], "title_tokens": frozenset(tokenize_persian(row["title_fa"])), "price": prices.get(product_id), "comments": tuple(sorted(comments[product_id], key=lambda item: item["comment_id"]))}
        self.build_runtime_seconds = time.perf_counter() - started
        self.artifact_bytes = sum((self.data_dir / name).stat().st_size for name in required[:3])

    def stats(self) -> dict[str, int | float]:
        return {"products": len(self.products), "canonical_comments": sum(len(p["comments"]) for p in self.products.values()), "artifact_bytes": self.artifact_bytes, "build_runtime_seconds": self.build_runtime_seconds}

    def search(self, query: str, *, min_historical_price: int | None = None, max_historical_price: int | None = None, brand: str | None = None, min_review_evidence: int = 0, limit: int = 10) -> dict[str, Any]:
        if limit <= 0 or min_review_evidence < 0:
            raise ValueError("limit must be positive and min_review_evidence must be non-negative")
        if min_historical_price is not None and max_historical_price is not None and min_historical_price > max_historical_price:
            raise ValueError("minimum historical price cannot exceed maximum historical price")
        terms = frozenset(tokenize_persian(query))
        response: dict[str, Any] = {"query": query, "normalized_query": normalize_persian(query), "query_tokens": sorted(terms), "filters": {"min_historical_price": min_historical_price, "max_historical_price": max_historical_price, "brand": brand, "min_review_evidence": min_review_evidence}, "scoring_formula": SCORING_FORMULA, "results": []}
        if not terms:
            response["reason"] = "empty_query"
            return response
        normalized_brand, price_filtered, results = normalize_persian(brand), min_historical_price is not None or max_historical_price is not None, []
        for product in self.products.values():
            price_row = product["price"]
            price = None if price_row is None else price_row["historical_price_inferred_irr"]
            if (price_filtered and price is None) or (min_historical_price is not None and price < min_historical_price) or (max_historical_price is not None and price > max_historical_price):
                continue
            if brand is not None and normalize_persian(product["brand"]) != normalized_brand:
                continue
            comments = product["comments"]
            if len(comments) < min_review_evidence:
                continue
            matched = [comment for comment in comments if comment["tokens"] & terms]
            matched_terms = (product["title_tokens"] & terms) | set().union(*(comment["tokens"] & terms for comment in matched)) if matched else product["title_tokens"] & terms
            if not matched_terms:
                continue
            buyers = sum(comment["is_buyer"] is True for comment in matched)
            valid = sum(comment["rate"] is not None and 0 < float(comment["rate"]) <= 5 for comment in matched)
            recommended = sum(normalize_persian(comment["recommendation_status"]) == "recommended" for comment in matched)
            lexical, coverage = len(matched_terms) / len(terms), min(2.0, 2 * len(matched) / max(1, len(comments)))
            components = {"lexical_relevance": lexical, "supporting_evidence_count": len(matched), "buyer_supporting_evidence_count": buyers, "rating_recommendation_strength": min(4.0, 0.5 * (valid + recommended)), "review_coverage_adjustment": coverage}
            components["score"] = 100 * lexical + min(8, len(matched)) * 1.5 + min(6, buyers) + components["rating_recommendation_strength"] + coverage
            evidence = []
            for comment in sorted(matched, key=lambda item: (-len(item["tokens"] & terms), -(item["is_buyer"] is True), item["comment_id"]))[:MAX_EXCERPTS]:
                evidence.append({"comment_id": comment["comment_id"], "canonical_source_row_number": comment["canonical_source_row_number"], "is_buyer": comment["is_buyer"], "matched_query_tokens": sorted(comment["tokens"] & terms), "excerpt": _excerpt(comment["body"] or comment["title"] or "", set(terms))})
            results.append({"product_id": product["product_id"], "title": product["title"], "brand": product["brand"], "historical_price_inferred_irr": price, "historical_price_label": HISTORICAL_PRICE_LABEL, "total_canonical_review_count": len(comments), "score_components": components, "evidence": evidence})
        response["results"] = sorted(results, key=lambda row: (-row["score_components"]["score"], -row["score_components"]["lexical_relevance"], row["product_id"]))[:limit]
        if not response["results"]:
            response["reason"] = "no_matching_products"
        return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect deterministic sunscreen lexical retrieval.")
    parser.add_argument("query")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-historical-price", type=int)
    parser.add_argument("--max-historical-price", type=int)
    parser.add_argument("--brand")
    parser.add_argument("--min-review-evidence", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    index = SunscreenLexicalIndex(args.data_dir)
    print(json.dumps({"index": index.stats(), **index.search(args.query, min_historical_price=args.min_historical_price, max_historical_price=args.max_historical_price, brand=args.brand, min_review_evidence=args.min_review_evidence, limit=args.limit)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
