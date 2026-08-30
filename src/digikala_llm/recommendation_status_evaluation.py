"""Leakage-safe Macro-F1 baseline for canonical sunscreen recommendation statuses."""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_DATA_DIR = Path("data/processed/sunscreen_mvp/v1")
DEFAULT_INPUT = DEFAULT_DATA_DIR / "sunscreen_comments_canonical.parquet"
DEFAULT_OUTPUT_DIR = Path("reports/evaluation")
SEED = 20_260_830
SPLIT_ALGORITHM = "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260830); first valid fold"
FEATURE_FIELDS = ("title", "body")
TARGET_FIELD = "recommendation_status"
GROUP_FIELD = "product_id"


class EvaluationDataError(ValueError):
    """The published input cannot support the documented evaluation contract."""


def valid_target(value: object) -> str | None:
    """Keep authoritative nonblank strings verbatim; all other target values are excluded."""
    return value if isinstance(value, str) and value.strip() else None


def comment_text(title: object, body: object) -> str:
    """Return the only permitted feature fields, retaining their cleaned published text."""
    return "\n".join(value for value in (title, body) if isinstance(value, str))


def load_examples(path: Path | str = DEFAULT_INPUT) -> tuple[list[dict[str, Any]], int, int]:
    """Load exactly the target, group, and allowed feature columns from published Parquet."""
    rows = pq.read_table(path, columns=[GROUP_FIELD, TARGET_FIELD, *FEATURE_FIELDS]).to_pylist()
    examples = []
    for row in rows:
        label = valid_target(row[TARGET_FIELD])
        if label is not None:
            examples.append(
                {"product_id": row[GROUP_FIELD], "label": label, "text": comment_text(row["title"], row["body"])}
            )
    return examples, len(rows), len(rows) - len(examples)


def label_order(examples: list[dict[str, Any]]) -> list[str]:
    return sorted({str(example["label"]) for example in examples})


def group_holdout(examples: list[dict[str, Any]], labels: list[str]) -> tuple[list[int], list[int]]:
    """Choose a deterministic, class-complete group-disjoint 80/20 fold."""
    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as error:
        raise EvaluationDataError("the optional ml dependency is required") from error
    y = np.asarray([example["label"] for example in examples])
    groups = np.asarray([example["product_id"] for example in examples])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    expected = set(labels)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if set(y[train]) == expected and set(y[test]) == expected and not set(groups[train]) & set(groups[test]):
            return train.tolist(), test.tolist()
    raise EvaluationDataError("no deterministic class-complete product-group holdout is feasible")


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "macro_f1": float(sum(f1) / len(labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": {"label_order": labels, "orientation": "rows=true, columns=predicted", "values": matrix},
    }


def _model_results(train: list[dict[str, Any]], test: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_text, train_labels = [row["text"] for row in train], [row["label"] for row in train]
    test_text, test_labels = [row["text"] for row in test], [row["label"] for row in test]
    dummy_started = time.perf_counter()
    dummy = DummyClassifier(strategy="most_frequent").fit([[0]] * len(train), train_labels)
    dummy_fit_seconds = time.perf_counter() - dummy_started
    dummy_predict_started = time.perf_counter()
    dummy_predictions = dummy.predict([[0]] * len(test)).tolist()
    dummy_prediction_seconds = time.perf_counter() - dummy_predict_started

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    text_started = time.perf_counter()
    train_features = vectorizer.fit_transform(train_text)
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED, solver="lbfgs"
    ).fit(train_features, train_labels)
    text_fit_seconds = time.perf_counter() - text_started
    text_predict_started = time.perf_counter()
    text_predictions = model.predict(vectorizer.transform(test_text)).tolist()
    text_prediction_seconds = time.perf_counter() - text_predict_started
    return {
        "dummy_most_frequent": {
            "configuration": {"strategy": "most_frequent"},
            **_metrics(test_labels, dummy_predictions, labels),
            "fit_seconds": dummy_fit_seconds,
            "prediction_seconds": dummy_prediction_seconds,
        },
        "text_tfidf_logistic_regression": {
            "configuration": {
                "feature_fields": list(FEATURE_FIELDS),
                "vectorizer": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2, "sublinear_tf": True},
                "logistic_regression": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": SEED, "solver": "lbfgs"},
            },
            **_metrics(test_labels, text_predictions, labels),
            "fit_seconds": text_fit_seconds,
            "prediction_seconds": text_prediction_seconds,
        },
    }


def evaluate_recommendation_status(path: Path | str = DEFAULT_INPUT) -> dict[str, Any]:
    """Run the fixed holdout once; output contains no retrieval, provider, or source-data mutation."""
    examples, total, excluded = load_examples(path)
    labels = label_order(examples)
    if len(labels) < 2:
        raise EvaluationDataError("at least two valid target labels are required")
    train_indices, test_indices = group_holdout(examples, labels)
    train, test = [examples[index] for index in train_indices], [examples[index] for index in test_indices]
    train_groups, test_groups = {row["product_id"] for row in train}, {row["product_id"] for row in test}
    if train_groups & test_groups:
        raise AssertionError("product-group leakage in evaluation split")
    models = _model_results(train, test, labels)
    text = models["text_tfidf_logistic_regression"]
    dummy = models["dummy_most_frequent"]
    weakest = min(text["per_class"].items(), key=lambda item: (item[1]["f1"], item[0]))[0]
    return {
        "input": {"path": str(path), "published_artifact_only": True},
        "target": {"field": TARGET_FIELD, "rule": "exclude null, non-string, and blank/whitespace values; preserve valid strings verbatim"},
        "features": {"fields": list(FEATURE_FIELDS), "forbidden": [TARGET_FIELD, "rate", "likes", "dislikes", "is_buyer", "product aggregates"]},
        "counts": {"total_canonical_rows": total, "usable_labelled_rows": len(examples), "excluded_rows": excluded, "class_distribution": dict(sorted(Counter(row["label"] for row in examples).items()))},
        "split": {"algorithm": SPLIT_ALGORITHM, "seed": SEED, "train_rows": len(train), "test_rows": len(test), "train_class_distribution": dict(sorted(Counter(row["label"] for row in train).items())), "test_class_distribution": dict(sorted(Counter(row["label"] for row in test).items())), "train_product_count": len(train_groups), "test_product_count": len(test_groups), "product_id_overlap_count": len(train_groups & test_groups)},
        "labels": labels,
        "models": models,
        "improvement_over_dummy_macro_f1": text["macro_f1"] - dummy["macro_f1"],
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": ["Single sunscreen-only holdout; no claim beyond this subset.", "No hyperparameter tuning was performed on the holdout.", "Text-only prediction does not establish product quality or medical suitability."],
        "failure_analysis": {"lowest_text_model_f1_label": weakest, "note": "Inspect the labelled confusion matrix; it is true rows by predicted columns."},
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# Recommendation-status baseline evaluation", "", "Published sunscreen canonical comments only.", "", "## Counts", ""]
    for key, value in result["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Split", "", f"- algorithm: `{result['split']['algorithm']}`", f"- seed: `{result['split']['seed']}`", f"- product overlap: `{result['split']['product_id_overlap_count']}`", "", "## Metrics", ""]
    for name, metrics in result["models"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Macro F1: `{metrics['macro_f1']:.6f}`",
                f"- Accuracy: `{metrics['accuracy']:.6f}`",
                f"- Configuration: `{metrics['configuration']}`",
                "",
                "| Label | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {label} | {values['precision']:.6f} | {values['recall']:.6f} | "
            f"{values['f1']:.6f} | {values['support']} |"
            for label, values in metrics["per_class"].items()
        )
        matrix = metrics["confusion_matrix"]
        lines.extend(
            [
                "",
                f"Confusion matrix ({matrix['orientation']}; {matrix['label_order']}):",
                "",
                "```text",
                *[" ".join(str(value) for value in row) for row in matrix["values"]],
                "```",
                "",
            ]
        )
    lines += ["", f"Improvement over dummy Macro F1: `{result['improvement_over_dummy_macro_f1']:.6f}`", "", "## Failure analysis", "", f"- Lowest text-model F1 label: `{result['failure_analysis']['lowest_text_model_f1_label']}`", f"- {result['failure_analysis']['note']}", "", "## Limitations", ""]
    lines.extend(f"- {item}" for item in result["limitations"])
    return "\n".join(lines) + "\n"


def write_reports(result: dict[str, Any], output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> None:
    """Atomically publish deterministic report content to the requested evaluation directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"recommendation_status_evaluation.json": json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "recommendation_status_evaluation.md": _markdown(result)}
    for name, content in outputs.items():
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                destination.write(content)
            os.replace(temporary, output_dir / name)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate recommendation_status from published sunscreen comments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = evaluate_recommendation_status(args.input)
    write_reports(result, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
