"""Focused leakage and metric contracts for recommendation-status evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm import recommendation_status_evaluation as evaluation

LABELS = ["recommended", "no_idea", "not_recommended"]


def _published(path: Path) -> Path:
    rows = []
    for label_index, label in enumerate(LABELS):
        for group_offset in range(5):
            product_id = label_index * 10 + group_offset
            for row_offset in range(2):
                rows.append(
                    {
                        "product_id": product_id,
                        "recommendation_status": label,
                        "title": f"عنوان {label}",
                        "body": f"متن {label} شماره {row_offset}",
                    }
                )
    rows.extend(
        [
            {"product_id": 99, "recommendation_status": None, "title": "x", "body": "x"},
            {"product_id": 100, "recommendation_status": "   ", "title": "x", "body": "x"},
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_target_validation_and_text_features_preserve_valid_labels(tmp_path: Path) -> None:
    examples, total, excluded = evaluation.load_examples(_published(tmp_path / "comments.parquet"))
    assert (total, len(examples), excluded) == (32, 30, 2)
    assert evaluation.valid_target(" recommended ") == " recommended "
    assert evaluation.valid_target("  ") is None
    assert evaluation.comment_text("عنوان", "متن") == "عنوان\nمتن"
    assert all(set(example) == {"product_id", "label", "text"} for example in examples)
    assert all(field not in str(examples) for field in ("rate", "likes", "is_buyer"))


def test_group_holdout_is_deterministic_disjoint_and_class_complete(tmp_path: Path) -> None:
    examples, _, _ = evaluation.load_examples(_published(tmp_path / "comments.parquet"))
    labels = evaluation.label_order(examples)
    first = evaluation.group_holdout(examples, labels)
    assert first == evaluation.group_holdout(examples, labels)
    train, test = ([examples[index] for index in indexes] for indexes in first)
    assert {row["product_id"] for row in train}.isdisjoint({row["product_id"] for row in test})
    assert {row["label"] for row in train} == set(labels) == {row["label"] for row in test}


def test_metrics_have_deterministic_label_order_and_true_predicted_matrix_orientation() -> None:
    metrics = evaluation._metrics(
        ["no_idea", "recommended", "recommended"],
        ["no_idea", "no_idea", "recommended"],
        ["no_idea", "recommended"],
    )
    assert metrics["confusion_matrix"] == {
        "label_order": ["no_idea", "recommended"],
        "orientation": "rows=true, columns=predicted",
        "values": [[1, 0], [1, 1]],
    }
    assert metrics["macro_f1"] == 2 / 3


def test_dummy_and_text_baselines_execute_and_report_atomically(tmp_path: Path) -> None:
    source = _published(tmp_path / "comments.parquet")
    result = evaluation.evaluate_recommendation_status(source)
    assert result["counts"]["excluded_rows"] == 2
    assert result["split"]["product_id_overlap_count"] == 0
    assert result["labels"] == sorted(LABELS)
    assert set(result["models"]) == {"dummy_most_frequent", "text_tfidf_logistic_regression"}
    assert result["models"]["text_tfidf_logistic_regression"]["configuration"]["feature_fields"] == ["title", "body"]
    evaluation.write_reports(result, tmp_path / "reports")
    first = (tmp_path / "reports" / "recommendation_status_evaluation.json").read_text()
    evaluation.write_reports(result, tmp_path / "reports")
    assert (tmp_path / "reports" / "recommendation_status_evaluation.json").read_text() == first
    assert json.loads(first)["labels"] == sorted(LABELS)
