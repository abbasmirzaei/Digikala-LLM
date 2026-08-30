"""Validated exact-vector semantic retrieval and deterministic hybrid fusion.

This module consumes only the published semantic artifact; it never reads raw CSV files and does
not rebuild embeddings.  Semantic failures are intentionally represented as a safe lexical mode.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.sunscreen_retrieval import (
    DEFAULT_DATA_DIR,
    HISTORICAL_PRICE_LABEL,
    MAX_EXCERPTS,
    SunscreenLexicalIndex,
    _excerpt,
    normalize_persian,
    tokenize_persian,
)
from digikala_llm.sunscreen_semantic_builder import (
    EMBEDDINGS_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_ID,
    VECTOR_DIMENSION,
    query_text,
)

DEFAULT_SEMANTIC_DIR = Path("data/processed/sunscreen_mvp/semantic_v1")
RRF_K = 60
SEMANTIC_COMMENT_LIMIT = 80


class SemanticRetrievalUnavailable(RuntimeError):
    """An expected local semantic dependency, model, or artifact is unavailable."""


class SemanticArtifactError(SemanticRetrievalUnavailable):
    """An artifact is unavailable or fails its published integrity contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SemanticCommentIndex:
    """Memory-mapped, validated normalized comment vectors with provenance metadata."""

    def __init__(self, artifact_dir: Path | str = DEFAULT_SEMANTIC_DIR) -> None:
        started = time.perf_counter()
        self.artifact_dir = Path(artifact_dir)
        required = (MANIFEST_FILENAME, "_SUCCESS", EMBEDDINGS_FILENAME, METADATA_FILENAME)
        missing = [name for name in required if not (self.artifact_dir / name).is_file()]
        if missing:
            raise SemanticArtifactError("semantic artifact is unavailable")
        try:
            manifest = json.loads((self.artifact_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            if (
                manifest.get("status") != "success"
                or manifest.get("embedding_model") != MODEL_ID
                or manifest.get("dimension") != VECTOR_DIMENSION
                or manifest.get("dtype") != "float32"
                or manifest.get("normalization") != "l2"
                or not isinstance(manifest.get("record_count"), int)
            ):
                raise ValueError("manifest contract")
            outputs = manifest["outputs"]
            for name in (EMBEDDINGS_FILENAME, METADATA_FILENAME):
                expected = outputs[name]
                path = self.artifact_dir / name
                if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
                    raise ValueError("checksum")
            matrix = np.load(self.artifact_dir / EMBEDDINGS_FILENAME, mmap_mode="r")
            count = manifest["record_count"]
            if matrix.dtype != np.dtype("float32") or matrix.shape != (count, VECTOR_DIMENSION):
                raise ValueError("matrix shape")
            metadata = pq.read_table(self.artifact_dir / METADATA_FILENAME).to_pylist()
            if len(metadata) != count or any(row["vector_row"] != number for number, row in enumerate(metadata)):
                raise ValueError("metadata alignment")
            if any(np.dtype(type(row[field])).kind not in "iu" for row in metadata for field in ("product_id", "comment_id", "canonical_source_row_number")):
                raise ValueError("metadata schema")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
                raise ValueError("normalization")
        except (
            EOFError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            pa.ArrowException,
        ) as error:
            raise SemanticArtifactError("semantic artifact validation failed") from error
        self.manifest, self.matrix, self.metadata = manifest, matrix, tuple(metadata)
        self.load_seconds = time.perf_counter() - started

    def search(self, vector: np.ndarray, limit: int = SEMANTIC_COMMENT_LIMIT) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (VECTOR_DIMENSION,) or not np.allclose(np.linalg.norm(vector), 1.0, atol=1e-5):
            raise ValueError("query must be normalized float32 with expected dimension")
        scores = self.matrix @ vector
        count = min(limit, len(scores))
        product_ids = np.fromiter((row["product_id"] for row in self.metadata), dtype=np.int64)
        source_rows = np.fromiter(
            (row["canonical_source_row_number"] for row in self.metadata), dtype=np.int64
        )
        comment_ids = np.fromiter((row["comment_id"] for row in self.metadata), dtype=np.int64)
        # Full ordering makes tied cutoff rows deterministic as well as bounded in returned size.
        chosen = np.lexsort((comment_ids, source_rows, product_ids, -scores))[:count]
        return [
            {**self.metadata[int(row)], "_semantic_score": float(scores[int(row)])}
            for row in chosen
        ]


@lru_cache(maxsize=1)
def _query_embedder() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SemanticRetrievalUnavailable("semantic dependency is unavailable") from error

    try:
        return SentenceTransformer(MODEL_ID, device="cpu", local_files_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise SemanticRetrievalUnavailable("cached semantic model is unavailable") from error


def embed_query(query: str) -> np.ndarray:
    """Encode one model-card-compliant CPU query and validate its normalization."""
    try:
        encoded = _query_embedder().encode(
            [query_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SemanticRetrievalUnavailable("semantic query embedding is unavailable") from error
    vector = np.asarray(encoded[0], dtype=np.float32)
    if vector.shape != (VECTOR_DIMENSION,) or not np.isfinite(vector).all():
        raise SemanticArtifactError("semantic query embedding is unavailable")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise SemanticArtifactError("semantic query embedding is unavailable")
    vector /= norm
    return vector


class HybridSunscreenRetriever:
    """Unchanged lexical channel plus exact semantic channel fused by RRF(k=60)."""

    def __init__(
        self,
        lexical_index: SunscreenLexicalIndex | None = None,
        semantic_dir: Path | str = DEFAULT_SEMANTIC_DIR,
        *,
        semantic_index: SemanticCommentIndex | None = None,
    ) -> None:
        self.lexical = lexical_index or SunscreenLexicalIndex(DEFAULT_DATA_DIR)
        self.semantic_error: str | None = None
        try:
            self.semantic = semantic_index or SemanticCommentIndex(semantic_dir)
        except SemanticRetrievalUnavailable:
            self.semantic = None
            self.semantic_error = "unavailable"

    def _eligible(self, product: dict[str, Any], filters: dict[str, Any]) -> bool:
        price = None if product["price"] is None else product["price"]["historical_price_inferred_irr"]
        low, high, brand, minimum = (
            filters["min_historical_price"], filters["max_historical_price"], filters["brand"], filters["min_review_evidence"]
        )
        if (low is not None or high is not None) and price is None:
            return False
        if (low is not None and price < low) or (high is not None and price > high):
            return False
        return not (
            (brand is not None and normalize_persian(product["brand"]) != normalize_persian(brand))
            or len(product["comments"]) < minimum
        )

    def _semantic_results(self, query: str, filters: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        if self.semantic is None:
            raise SemanticArtifactError("semantic retriever unavailable")
        hits = self.semantic.search(embed_query(query))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for hit in hits:
            product = self.lexical.products.get(hit["product_id"])
            if product is not None and self._eligible(product, filters):
                grouped[hit["product_id"]].append(hit)
        rows = []
        for product_id, product_hits in grouped.items():
            product_hits.sort(key=lambda hit: (-hit["_semantic_score"], hit["canonical_source_row_number"], hit["comment_id"]))
            product = self.lexical.products[product_id]
            evidence = []
            for hit in product_hits[:MAX_EXCERPTS]:
                comment = next(item for item in product["comments"] if item["comment_id"] == hit["comment_id"] and item["canonical_source_row_number"] == hit["canonical_source_row_number"])
                evidence.append({"comment_id": hit["comment_id"], "canonical_source_row_number": hit["canonical_source_row_number"], "is_buyer": comment["is_buyer"], "matched_query_tokens": [], "excerpt": _excerpt(comment["body"] or comment["title"] or "", set(tokenize_persian(query)))})
            price = None if product["price"] is None else product["price"]["historical_price_inferred_irr"]
            rows.append({"product_id": product_id, "title": product["title"], "brand": product["brand"], "historical_price_inferred_irr": price, "historical_price_label": HISTORICAL_PRICE_LABEL, "total_canonical_review_count": len(product["comments"]), "evidence": evidence, "_rank_score": product_hits[0]["_semantic_score"], "_source": product_hits[0]["canonical_source_row_number"]})
        return sorted(rows, key=lambda row: (-row["_rank_score"], row["product_id"], row["_source"]))[:limit]

    @staticmethod
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if not key.startswith("_")}

    def search(self, query: str, *, mode: str = "hybrid", min_historical_price: int | None = None, max_historical_price: int | None = None, brand: str | None = None, min_review_evidence: int = 0, limit: int = 10) -> dict[str, Any]:
        if mode not in {"hybrid", "lexical", "semantic"}:
            raise ValueError("mode must be hybrid, lexical, or semantic")
        filters = {"min_historical_price": min_historical_price, "max_historical_price": max_historical_price, "brand": brand, "min_review_evidence": min_review_evidence}
        lexical = self.lexical.search(query, limit=limit, **filters)
        base = {key: value for key, value in lexical.items() if key not in {"results", "scoring_formula"}}
        if mode == "lexical":
            return {**base, "results": lexical["results"], "retrieval_mode": "lexical"}
        try:
            semantic = self._semantic_results(query, filters, limit)
        except SemanticRetrievalUnavailable:
            return {**base, "results": lexical["results"], "retrieval_mode": "lexical_fallback"}
        if mode == "semantic":
            return {**base, "results": [self._public(row) for row in semantic], "retrieval_mode": "semantic"}
        by_id = {row["product_id"]: row for row in lexical["results"]}
        semantic_by_id = {row["product_id"]: row for row in semantic}
        fused = []
        for product_id in set(by_id) | set(semantic_by_id):
            left, right = by_id.get(product_id), semantic_by_id.get(product_id)
            lrank = next((rank for rank, row in enumerate(lexical["results"], 1) if row["product_id"] == product_id), None)
            srank = next((rank for rank, row in enumerate(semantic, 1) if row["product_id"] == product_id), None)
            score = (0 if lrank is None else 1 / (RRF_K + lrank)) + (0 if srank is None else 1 / (RRF_K + srank))
            selected = dict(left or right)
            if right is not None:
                selected["evidence"] = right["evidence"]
            selected["_rrf"] = score
            selected["_source"] = min((row["canonical_source_row_number"] for row in selected["evidence"]), default=0)
            fused.append(selected)
        fused.sort(key=lambda row: (-row["_rrf"], row["product_id"], row["_source"]))
        return {**base, "results": [self._public(row) for row in fused[:limit]], "retrieval_mode": "hybrid"}
