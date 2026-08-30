"""Offline evaluation of the deterministic candidate and comparison layers.

Groq is deliberately outside this runner: retrieval and measurable comparison remain local,
deterministic, CPU-only sources of truth. Groq only summarizes their bounded output in the UI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from digikala_llm.sunscreen_comparison import SunscreenComparisonService
from digikala_llm.sunscreen_hybrid import DEFAULT_SEMANTIC_DIR, HybridSunscreenRetriever
from digikala_llm.sunscreen_retrieval import (
    DEFAULT_DATA_DIR,
    MAX_EXCERPT_CHARS,
    SunscreenLexicalIndex,
)

DEFAULT_CASES = Path("specs/002-sunscreen-mvp/evaluation.jsonl")


def run_evaluation(
    data_dir: Path | str = DEFAULT_DATA_DIR, cases_path: Path | str = DEFAULT_CASES
) -> dict[str, Any]:
    index = SunscreenLexicalIndex(data_dir)
    service = SunscreenComparisonService(index)
    cases = [
        json.loads(line)
        for line in Path(cases_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    outcomes = []
    for case in cases:
        if case.get("semantic_only"):
            continue
        started = time.perf_counter()
        failures = []
        if case["kind"] == "search":
            result = index.search(case["query"], **case.get("filters", {}))
            again = index.search(case["query"], **case.get("filters", {}))
            if bool(result["results"]) != case["expect_results"]:
                failures.append("unexpected_result_presence")
            if [x["product_id"] for x in result["results"]] != [
                x["product_id"] for x in again["results"]
            ]:
                failures.append("nondeterministic_order")
            for item in result["results"]:
                if (
                    item["product_id"] not in index.products
                    or item["historical_price_label"] != "historical inferred IRR"
                ):
                    failures.append("catalog_or_price_label")
                for evidence in item["evidence"]:
                    if len(evidence["excerpt"]) > MAX_EXCERPT_CHARS or not all(
                        k in evidence for k in ("comment_id", "canonical_source_row_number")
                    ):
                        failures.append("bad_evidence")
                if any(k in item for k in ("skin_type", "medical_suitability", "spf")):
                    failures.append("unsupported_claim")
        else:
            result = service.compare(case["product_ids"], query=case.get("query"))
            if result["overall_winner"] is not None or len(result["products"]) != len(
                case["product_ids"]
            ):
                failures.append("comparison_shape")
            for item in result["products"]:
                if item["buyer_review_count"] > item["canonical_review_count"]:
                    failures.append("aggregate_reconciliation")
        outcomes.append(
            {
                "id": case["id"],
                "status": "pass" if not failures else "fail",
                "passed": not failures,
                "checks": {"failures": failures},
                "failures": failures,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    return {
        "generation_note": "Deterministic retrieval and comparison are evaluated locally; Groq synthesis is not required.",
        "summary": {
            "passed": sum(x["passed"] for x in outcomes),
            "failed": sum(not x["passed"] for x in outcomes),
        },
        "passed": sum(x["passed"] for x in outcomes),
        "failed": sum(not x["passed"] for x in outcomes),
        "cases": outcomes,
        "index": index.stats(),
    }


def run_retrieval_comparison(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    cases_path: Path | str = DEFAULT_CASES,
    semantic_dir: Path | str = DEFAULT_SEMANTIC_DIR,
) -> dict[str, Any]:
    """Record deterministic lexical/semantic/hybrid comparison without claiming uplift."""
    retriever = HybridSunscreenRetriever(SunscreenLexicalIndex(data_dir), semantic_dir)
    cases = [json.loads(line) for line in Path(cases_path).read_text(encoding="utf-8").splitlines() if line]
    outcomes = []
    for case in cases:
        if case["kind"] != "search":
            continue
        channels, repeatable = {}, True
        for mode in ("lexical", "semantic", "hybrid"):
            started = time.perf_counter()
            result = retriever.search(case["query"], mode=mode, **case.get("filters", {}))
            again = retriever.search(case["query"], mode=mode, **case.get("filters", {}))
            channels[mode] = {
                "retrieval_mode": result["retrieval_mode"],
                "product_ids": [row["product_id"] for row in result["results"]],
                "evidence": [[{"comment_id": e["comment_id"], "canonical_source_row_number": e["canonical_source_row_number"]} for e in row["evidence"]] for row in result["results"]],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            repeatable = repeatable and result == again
        outcomes.append({"id": case["id"], "deterministic": repeatable, "channels": channels})
    return {"note": "Comparison records observed fixed-case outcomes; it makes no improvement claim.", "semantic_load_seconds": None if retriever.semantic is None else retriever.semantic.load_seconds, "cases": outcomes}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    p.add_argument("--comparison", action="store_true", help="record lexical, semantic, and hybrid retrieval")
    p.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    a = p.parse_args(argv)
    result = run_retrieval_comparison(a.data_dir, a.cases, a.semantic_dir) if a.comparison else run_evaluation(a.data_dir, a.cases)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "sunscreen_mvp_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (a.output_dir / "sunscreen_mvp_evaluation.md").write_text(
        "# Sunscreen MVP evaluation\n\n"
        + (f"Passed: {result['passed']}\n\nFailed: {result['failed']}\n\n" if not a.comparison else result["note"] + "\n\n")
        + "\n".join(
            f"- {x['id']}: {'PASS' if (x['deterministic'] if a.comparison else x['passed']) else 'FAIL'}"
            for x in result["cases"]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if a.comparison or not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
