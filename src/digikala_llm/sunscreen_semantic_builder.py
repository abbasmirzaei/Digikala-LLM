"""CPU-only semantic embedding artifacts from published sunscreen MVP data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE_DIR = Path("data/processed/sunscreen_mvp/v1")
DEFAULT_OUTPUT_DIR = Path("data/processed/sunscreen_mvp/semantic_v1")
MODEL_ID = "intfloat/multilingual-e5-small"
VECTOR_DIMENSION = 384
CANONICAL_COMMENT_COUNT = 53_365
ARTIFACT_VERSION = "sunscreen-semantic-v1"
EMBEDDINGS_FILENAME = "comment_embeddings.npy"
METADATA_FILENAME = "comment_embedding_metadata.parquet"
MANIFEST_FILENAME = "manifest.json"
SOURCE_FILENAMES = ("sunscreen_products.parquet", "sunscreen_comments_canonical.parquet")
METADATA_SCHEMA = pa.schema(
    [
        pa.field("vector_row", pa.int64(), nullable=False),
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("comment_id", pa.int64(), nullable=False),
        pa.field("canonical_source_row_number", pa.int64(), nullable=False),
    ]
)


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passage_text(title: str | None, brand: str | None, body: str | None) -> str:
    """Return the model-card-required passage prefix and only permitted source fields."""
    return "passage: " + "\n".join(value or "" for value in (title, brand, body))


def query_text(query: str) -> str:
    """Expose the paired model-card query prefix for the later retrieval milestone."""
    return f"query: {query}"


def _source_paths(source_dir: Path) -> dict[str, Path]:
    paths = {name: source_dir / name for name in SOURCE_FILENAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing or not (source_dir / "_SUCCESS").is_file():
        raise FileNotFoundError(f"published sunscreen artifacts missing: {missing or ['_SUCCESS']}")
    return paths


def _published_rows(source_dir: Path) -> Iterator[dict[str, Any]]:
    paths = _source_paths(source_dir)
    products = {
        row["product_id"]: row
        for row in pq.read_table(paths["sunscreen_products.parquet"], columns=["product_id", "title_fa", "brand"])
        .to_pylist()
    }
    previous_key: tuple[int, int, int] | None = None
    for batch in pq.ParquetFile(paths["sunscreen_comments_canonical.parquet"]).iter_batches(
        batch_size=4_096,
        columns=["product_id", "comment_id", "canonical_source_row_number", "body"],
    ):
        for comment in pa.Table.from_batches([batch]).to_pylist():
            product = products.get(comment["product_id"])
            if product is None:
                raise RuntimeError(f"canonical comment references unknown product {comment['product_id']}")
            key = (
                comment["comment_id"],
                comment["product_id"],
                comment["canonical_source_row_number"],
            )
            if previous_key is not None and key <= previous_key:
                raise RuntimeError("canonical comments must be strictly ordered by provenance key")
            previous_key = key
            yield {
                "product_id": comment["product_id"],
                "comment_id": comment["comment_id"],
                "canonical_source_row_number": comment["canonical_source_row_number"],
                "passage": passage_text(product["title_fa"], product["brand"], comment["body"]),
            }


def published_sample(source_dir: Path | str, limit: int = 2) -> list[dict[str, Any]]:
    """Read a bounded published-artifact sample for a model smoke test."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows: list[dict[str, Any]] = []
    for row in _published_rows(Path(source_dir)):
        rows.append(row)
        if len(rows) == limit:
            break
    return rows


def _load_embedder(model_id: str = MODEL_ID) -> Embedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device="cpu")


def _encode(embedder: Embedder, passages: list[str], batch_size: int) -> np.ndarray:
    vectors = np.asarray(
        embedder.encode(
            passages,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape != (len(passages), VECTOR_DIMENSION):
        raise RuntimeError(f"unexpected embedding shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("embedding vector has zero norm")
    vectors /= norms
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise RuntimeError("embedding normalization validation failed")
    return vectors


def model_smoke(source_dir: Path | str, batch_size: int = 2) -> dict[str, Any]:
    """Run the approved tiny model-backed smoke without publishing an artifact."""
    rows = published_sample(source_dir, limit=2)
    if not all(row["passage"].startswith("passage: ") for row in rows):
        raise RuntimeError("passage prefix validation failed")
    vectors = _encode(_load_embedder(), [row["passage"] for row in rows], batch_size)
    return {
        "model": MODEL_ID,
        "device": "cpu",
        "records": len(rows),
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "norms": np.linalg.norm(vectors, axis=1).tolist(),
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "metadata": [{key: row[key] for key in METADATA_SCHEMA.names if key != "vector_row"} for row in rows],
    }


def _validate_artifact(embeddings_path: Path, metadata_path: Path, count: int) -> None:
    vectors = np.load(embeddings_path, mmap_mode="r")
    metadata = pq.read_table(metadata_path).to_pylist()
    if vectors.dtype != np.dtype("float32") or vectors.shape != (count, VECTOR_DIMENSION):
        raise RuntimeError("embedding matrix shape or dtype validation failed")
    if len(metadata) != count or [row["vector_row"] for row in metadata] != list(range(count)):
        raise RuntimeError("vector metadata row alignment validation failed")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise RuntimeError("published embedding matrix normalization validation failed")


def build_semantic_artifact(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    batch_size: int = 64,
    expected_count: int | None = CANONICAL_COMMENT_COUNT,
    embedder_factory: Callable[[], Embedder] | None = None,
    failure_injector: Callable[[str], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Atomically publish vectors and explicit provenance from published artifacts only."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = _source_paths(source_dir)
    source_checksums = {name: _sha256(path) for name, path in sorted(paths.items())}
    rows = _published_rows(source_dir)
    first_batch: list[dict[str, Any]] = []
    for _ in range(batch_size):
        try:
            first_batch.append(next(rows))
        except StopIteration:
            break
    if not first_batch:
        raise RuntimeError("published canonical comments are empty")
    embedder = (embedder_factory or _load_embedder)()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    started = time.monotonic()
    metadata: list[dict[str, int]] = []
    try:
        batches = iter([first_batch])
        embeddings_path = staging / EMBEDDINGS_FILENAME
        matrix: np.memmap | None = None
        count = 0
        pending = list(first_batch)
        while pending:
            vectors = _encode(embedder, [row["passage"] for row in pending], batch_size)
            if matrix is None:
                capacity = expected_count if expected_count is not None else len(pending)
                matrix = np.lib.format.open_memmap(
                    embeddings_path, mode="w+", dtype=np.float32, shape=(capacity, VECTOR_DIMENSION)
                )
            if count + len(pending) > matrix.shape[0]:
                raise RuntimeError("published record count exceeds expected_count")
            matrix[count : count + len(pending)] = vectors
            metadata.extend(
                {
                    "vector_row": count + offset,
                    "product_id": row["product_id"],
                    "comment_id": row["comment_id"],
                    "canonical_source_row_number": row["canonical_source_row_number"],
                }
                for offset, row in enumerate(pending)
            )
            count += len(pending)
            if progress is not None:
                progress(f"embedded {count:,} published canonical comments")
            pending = []
            for _ in range(batch_size):
                try:
                    pending.append(next(rows))
                except StopIteration:
                    break
        del batches
        if expected_count is not None and count != expected_count:
            raise RuntimeError(f"canonical comment count mismatch: expected {expected_count}, got {count}")
        assert matrix is not None
        matrix.flush()
        del matrix
        if expected_count is None:
            raise RuntimeError("expected_count is required for publishable deterministic artifacts")
        pq.write_table(
            pa.Table.from_pylist(metadata, schema=METADATA_SCHEMA),
            staging / METADATA_FILENAME,
            compression="snappy",
        )
        _validate_artifact(embeddings_path, staging / METADATA_FILENAME, count)
        if failure_injector is not None:
            failure_injector("after_validation")
        outputs = {
            EMBEDDINGS_FILENAME: {"bytes": embeddings_path.stat().st_size, "sha256": _sha256(embeddings_path)},
            METADATA_FILENAME: {
                "bytes": (staging / METADATA_FILENAME).stat().st_size,
                "sha256": _sha256(staging / METADATA_FILENAME),
            },
        }
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "embedding_model": MODEL_ID,
            "dimension": VECTOR_DIMENSION,
            "dtype": "float32",
            "normalization": "l2",
            "record_count": count,
            "source_checksums": source_checksums,
            "outputs": outputs,
            "configuration": {"device": "cpu", "batch_size": batch_size, "passage_prefix": "passage: "},
            "runtime_seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "status": "success",
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "_SUCCESS").write_text("\n", encoding="utf-8")
        staging.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CPU semantic artifacts from published sunscreen data.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="embed only two published records; do not publish")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(model_smoke(args.source_dir, args.batch_size), ensure_ascii=False, sort_keys=True))
        return 0
    manifest = build_semantic_artifact(
        args.source_dir,
        args.output_dir,
        batch_size=args.batch_size,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
