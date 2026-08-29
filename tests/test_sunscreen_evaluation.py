"""Focused tests for fixed, deterministic sunscreen evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from digikala_llm import sunscreen_evaluation as evaluation


class _Index:
    def __init__(self, _data_dir: Path) -> None:
        self.products = {1: {"product_id": 1}, 2: {"product_id": 2}}

    def search(self, query: str, **_filters: object) -> dict[str, object]:
        if query == "none":
            return {"results": []}
        return {"results": [{"product_id": 1, "historical_price_label": "historical inferred IRR", "evidence": [{"comment_id": 10, "canonical_source_row_number": 20, "excerpt": "نظر کوتاه"}]}]}

    def stats(self) -> dict[str, int]: return {"products": 2}


class _Comparison:
    def __init__(self, _index: _Index) -> None: pass

    def compare(self, product_ids: list[int], **_kwargs: object) -> dict[str, object]:
        return {"overall_winner": None, "products": [{"product_id": product_id, "canonical_review_count": 2, "buyer_review_count": 1} for product_id in product_ids]}


def _write_cases(path: Path) -> None:
    cases = [
        {"id": "search", "kind": "search", "query": "سبک", "expect_results": True},
        {"id": "empty", "kind": "search", "query": "none", "expect_results": False},
        {"id": "comparison", "kind": "comparison", "product_ids": [1, 2]},
    ]
    path.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")


def test_fixed_evaluation_set_has_ten_unique_valid_cases() -> None:
    cases = [json.loads(line) for line in evaluation.DEFAULT_CASES.read_text(encoding="utf-8").splitlines() if line]
    assert len(cases) == 10
    assert len({case["id"] for case in cases}) == 10
    assert all(case["kind"] in {"search", "comparison"} for case in cases)


def test_evaluation_is_deterministic_and_reports_required_schema(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cases_path = tmp_path / "cases.jsonl"; _write_cases(cases_path)
    monkeypatch.setattr(evaluation, "SunscreenLexicalIndex", _Index)
    monkeypatch.setattr(evaluation, "SunscreenComparisonService", _Comparison)
    first = evaluation.run_evaluation(tmp_path, cases_path)
    second = evaluation.run_evaluation(tmp_path, cases_path)
    assert first["summary"] == {"passed": 3, "failed": 0}
    assert [{key: row[key] for key in ("id", "status", "passed", "checks", "failures")} for row in first["cases"]] == [{key: row[key] for key in ("id", "status", "passed", "checks", "failures")} for row in second["cases"]]
    for row in first["cases"]:
        assert row["status"] == "pass"
        assert isinstance(row["latency_ms"], float)
        assert row["checks"] == {"failures": []}


def test_evaluation_detects_evidence_and_price_label_failures(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _BadIndex(_Index):
        def search(self, _query: str, **_filters: object) -> dict[str, object]:
            return {"results": [{"product_id": 99, "historical_price_label": "current price", "evidence": [{"excerpt": "x" * (evaluation.MAX_EXCERPT_CHARS + 1)}]}]}

    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps({"id": "bad", "kind": "search", "query": "bad", "expect_results": True}) + "\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "SunscreenLexicalIndex", _BadIndex)
    monkeypatch.setattr(evaluation, "SunscreenComparisonService", _Comparison)
    outcome = evaluation.run_evaluation(tmp_path, cases_path)["cases"][0]
    assert outcome["status"] == "fail"
    assert set(outcome["failures"]) == {"catalog_or_price_label", "bad_evidence"}
