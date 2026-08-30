"""Offline tests for deterministic published-artifact semantic construction."""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from digikala_llm import sunscreen_semantic_builder as semantic
from digikala_llm.cleaning import COMMENTS_CLEAN_SCHEMA, PRODUCTS_CLEAN_SCHEMA


class _Embedder:
    def encode(self, sentences: list[str], **_kwargs: object) -> np.ndarray:
        vectors = np.zeros((len(sentences), semantic.VECTOR_DIMENSION), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            digest = hashlib.sha256(sentence.encode()).digest()
            vectors[row, int.from_bytes(digest[:2], "big") % semantic.VECTOR_DIMENSION] = 1
        return vectors


def _comment(comment_id: int, product_id: int, source_row: int, body: str) -> dict[str, object]:
    return {
        "comment_id": comment_id,
        "product_id": product_id,
        "title": None,
        "body": body,
        "created_at_raw": "1 فروردین 1400",
        "created_at_jalali": "1400-01-01",
        "created_at_gregorian": date(2021, 3, 21),
        "rate": Decimal(5),
        "is_unrated": False,
        "invalid_rate": False,
        "recommendation_status": "recommended",
        "is_buyer": True,
        "advantages": None,
        "disadvantages": None,
        "likes": None,
        "dislikes": None,
        "seller_title": None,
        "seller_code": None,
        "true_to_size_rate": None,
        "comment_id_conflict": False,
        "canonical_source_row_number": source_row,
    }


def _published(tmp_path: Path) -> Path:
    source = tmp_path / "published"
    source.mkdir(parents=True)
    products = [
        {
            "product_id": 1,
            "title_fa": "ضد آفتاب الف",
            "category1": "مراقبت پوست",
            "category2": "کرم ضد آفتاب",
            "brand": "برند الف",
            "rate": None,
            "rate_count": 0,
            "sub_category": None,
            "is_unrated": True,
            "inconsistent_zero_rate": False,
            "core_attribute_conflict": False,
            "canonical_source_row_number": 1,
        },
        {
            "product_id": 2,
            "title_fa": "ضد آفتاب ب",
            "category1": "مراقبت پوست",
            "category2": "کرم ضد آفتاب",
            "brand": "برند ب",
            "rate": None,
            "rate_count": 0,
            "sub_category": None,
            "is_unrated": True,
            "inconsistent_zero_rate": False,
            "core_attribute_conflict": False,
            "canonical_source_row_number": 2,
        },
    ]
    comments = [
        _comment(10, 1, 101, "متن اول"),
        _comment(11, 1, 102, "متن دوم"),
        _comment(20, 2, 201, "متن سوم"),
    ]
    pq.write_table(pa.Table.from_pylist(products, schema=PRODUCTS_CLEAN_SCHEMA), source / "sunscreen_products.parquet")
    pq.write_table(
        pa.Table.from_pylist(comments, schema=COMMENTS_CLEAN_SCHEMA),
        source / "sunscreen_comments_canonical.parquet",
    )
    (source / "_SUCCESS").write_text("\n", encoding="utf-8")
    return source


def _build(tmp_path: Path, **kwargs: object) -> tuple[dict[str, object], Path]:
    source = _published(tmp_path)
    output = tmp_path / "semantic"
    manifest = semantic.build_semantic_artifact(
        source,
        output,
        expected_count=3,
        batch_size=2,
        embedder_factory=_Embedder,
        **kwargs,
    )
    return manifest, output


def test_prefix_contract_contains_only_permitted_passage_fields() -> None:
    passage = semantic.passage_text("عنوان", "برند", "متن نظر")
    assert passage == "passage: عنوان\nبرند\nمتن نظر"
    assert semantic.query_text("ضد آفتاب سبک") == "query: ضد آفتاب سبک"


def test_builder_validates_dimension_dtype_norms_and_metadata_alignment(tmp_path: Path) -> None:
    manifest, output = _build(tmp_path)
    vectors = np.load(output / semantic.EMBEDDINGS_FILENAME)
    metadata = pq.read_table(output / semantic.METADATA_FILENAME).to_pylist()
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert metadata == [
        {"vector_row": 0, "product_id": 1, "comment_id": 10, "canonical_source_row_number": 101},
        {"vector_row": 1, "product_id": 1, "comment_id": 11, "canonical_source_row_number": 102},
        {"vector_row": 2, "product_id": 2, "comment_id": 20, "canonical_source_row_number": 201},
    ]
    assert manifest["record_count"] == 3
    assert (output / "_SUCCESS").is_file()


def test_artifact_output_and_manifest_checksums_are_deterministic(tmp_path: Path) -> None:
    manifest_a, output_a = _build(tmp_path / "a")
    manifest_b, output_b = _build(tmp_path / "b")
    for filename in (semantic.EMBEDDINGS_FILENAME, semantic.METADATA_FILENAME):
        assert (output_a / filename).read_bytes() == (output_b / filename).read_bytes()
        assert manifest_a["outputs"][filename]["sha256"] == semantic._sha256(output_a / filename)
    assert manifest_a["source_checksums"] == manifest_b["source_checksums"]


def test_atomic_failure_cleans_staging_and_never_writes_success(tmp_path: Path) -> None:
    source = _published(tmp_path)
    output = tmp_path / "semantic"
    with pytest.raises(RuntimeError, match="injected"):
        semantic.build_semantic_artifact(
            source,
            output,
            expected_count=3,
            embedder_factory=_Embedder,
            failure_injector=lambda _phase: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".semantic.staging-*"))


def test_missing_published_artifacts_are_rejected_without_raw_data_access(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        semantic.build_semantic_artifact(tmp_path, tmp_path / "semantic", expected_count=3)
