"""Focused contracts for validated, deterministic hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.cleaning import COMMENTS_CLEAN_SCHEMA, PRODUCTS_CLEAN_SCHEMA
from digikala_llm.sunscreen_builder import HISTORICAL_PRICE_LABEL, PRICE_SCHEMA
from digikala_llm.sunscreen_hybrid import (
    EMBEDDINGS_FILENAME,
    METADATA_FILENAME,
    HybridSunscreenRetriever,
    SemanticArtifactError,
    SemanticCommentIndex,
    SemanticRetrievalUnavailable,
)
from digikala_llm.sunscreen_llm import retrieval_context
from digikala_llm.sunscreen_retrieval import SunscreenLexicalIndex


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[SunscreenLexicalIndex, Path]:
    products = [
        {"product_id": n, "title_fa": f"ضد آفتاب {name}", "category1": "مراقبت پوست", "category2": "کرم ضد آفتاب", "brand": "الف" if n == 1 else "ب", "rate": None, "rate_count": 0, "sub_category": None, "is_unrated": True, "inconsistent_zero_rate": False, "core_attribute_conflict": False, "canonical_source_row_number": n}
        for n, name in ((1, "سبک"), (2, "بی رنگ"))
    ]
    comments = []
    for comment_id, product_id, body, row in ((10, 1, "حس سبکی دارد", 101), (20, 2, "رد سفیدی ندارد", 202)):
        comments.append({"comment_id": comment_id, "product_id": product_id, "title": None, "body": body, "created_at_raw": "1 فروردین 1400", "created_at_jalali": "1400-01-01", "created_at_gregorian": date(2021, 3, 21), "rate": Decimal(5), "is_unrated": False, "invalid_rate": False, "recommendation_status": "recommended", "is_buyer": True, "advantages": None, "disadvantages": None, "likes": None, "dislikes": None, "seller_title": None, "seller_code": None, "true_to_size_rate": None, "comment_id_conflict": False, "canonical_source_row_number": row})
    pq.write_table(pa.Table.from_pylist(products, schema=PRODUCTS_CLEAN_SCHEMA), tmp_path / "sunscreen_products.parquet")
    pq.write_table(pa.Table.from_pylist(comments, schema=COMMENTS_CLEAN_SCHEMA), tmp_path / "sunscreen_comments_canonical.parquet")
    pq.write_table(pa.Table.from_pylist([{"product_id": 1, "historical_price_inferred_irr": 100, "valid_price_offer_count": 1, "historical_price_label": HISTORICAL_PRICE_LABEL}, {"product_id": 2, "historical_price_inferred_irr": 200, "valid_price_offer_count": 1, "historical_price_label": HISTORICAL_PRICE_LABEL}], schema=PRICE_SCHEMA), tmp_path / "sunscreen_prices.parquet")
    (tmp_path / "_SUCCESS").write_text("\n")
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[0, 0], vectors[1, 1] = 1, 1
    np.save(semantic / EMBEDDINGS_FILENAME, vectors)
    pq.write_table(pa.Table.from_pylist([{"vector_row": 0, "product_id": 1, "comment_id": 10, "canonical_source_row_number": 101}, {"vector_row": 1, "product_id": 2, "comment_id": 20, "canonical_source_row_number": 202}]), semantic / METADATA_FILENAME)
    outputs = {name: {"bytes": (semantic / name).stat().st_size, "sha256": _sha(semantic / name)} for name in (EMBEDDINGS_FILENAME, METADATA_FILENAME)}
    (semantic / "manifest.json").write_text(json.dumps({"status": "success", "embedding_model": "intfloat/multilingual-e5-small", "dimension": 384, "dtype": "float32", "normalization": "l2", "record_count": 2, "outputs": outputs}))
    (semantic / "_SUCCESS").write_text("\n")
    return SunscreenLexicalIndex(tmp_path), semantic


def test_manifest_success_checksum_mmap_and_exact_dot_product(tmp_path: Path) -> None:
    _, artifact = _fixture(tmp_path)
    index = SemanticCommentIndex(artifact)
    assert isinstance(index.matrix, np.memmap)
    assert index.search(np.eye(1, 384, 0, dtype=np.float32)[0], 2)[0]["product_id"] == 1
    (artifact / "_SUCCESS").unlink()
    try:
        SemanticCommentIndex(artifact)
    except SemanticArtifactError:
        pass
    else:
        raise AssertionError("_SUCCESS must be required")


def test_hybrid_prefix_filters_citations_rrf_ties_and_fallback(tmp_path: Path, monkeypatch) -> None:
    lexical, artifact = _fixture(tmp_path)
    calls: list[str] = []

    class Encoder:
        def encode(self, values, **_kwargs):
            calls.extend(values)
            vector = np.zeros((1, 384), dtype=np.float32)
            vector[0, 0] = 1
            return vector

    monkeypatch.setattr("digikala_llm.sunscreen_hybrid._query_embedder", lambda: Encoder())
    retriever = HybridSunscreenRetriever(lexical, artifact)
    result = retriever.search("توصیف متفاوت", brand="الف")
    assert result["retrieval_mode"] == "hybrid" and result["results"][0]["product_id"] == 1
    assert result["results"][0]["evidence"][0]["canonical_source_row_number"] == 101
    assert calls == ["query: توصیف متفاوت"]
    assert retriever.search("توصیف متفاوت", brand="الف") == result
    context = json.dumps(retrieval_context("توصیف متفاوت", result), ensure_ascii=False)
    assert "score" not in context and "embedding" not in context and "_semantic" not in context
    fallback = HybridSunscreenRetriever(lexical, tmp_path / "missing").search("ضد آفتاب")
    assert fallback["retrieval_mode"] == "lexical_fallback"


def test_only_expected_semantic_unavailability_falls_back(tmp_path: Path, monkeypatch) -> None:
    lexical, artifact = _fixture(tmp_path)
    retriever = HybridSunscreenRetriever(lexical, artifact)
    monkeypatch.setattr(
        retriever,
        "_semantic_results",
        lambda *_args: (_ for _ in ()).throw(SemanticRetrievalUnavailable("offline")),
    )
    assert retriever.search("ضد آفتاب")["retrieval_mode"] == "lexical_fallback"


def test_unexpected_semantic_programming_error_is_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    lexical, artifact = _fixture(tmp_path)
    retriever = HybridSunscreenRetriever(lexical, artifact)
    monkeypatch.setattr(
        retriever,
        "_semantic_results",
        lambda *_args: (_ for _ in ()).throw(AssertionError("programming error")),
    )
    try:
        retriever.search("ضد آفتاب")
    except AssertionError as error:
        assert str(error) == "programming error"
    else:
        raise AssertionError("unexpected errors must not become lexical fallbacks")
