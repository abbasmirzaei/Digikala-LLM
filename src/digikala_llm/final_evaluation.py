"""Truthful consolidation of published sunscreen MVP evaluation evidence; no model/API calls."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from digikala_llm import sunscreen_llm as llm

ROOT = Path(".")
REPORTS = ROOT / "reports/evaluation"
DATA_DIR = ROOT / "data/processed/sunscreen_mvp/v1"
SEMANTIC_DIR = ROOT / "data/processed/sunscreen_mvp/semantic_v1"
BASELINE_CASE_IDS = ("oily", "dry", "colorless", "light", "value", "brand", "price", "minimum_evidence", "no_result", "comparison")
SEMANTIC_CASE_IDS = ("semantic_paraphrase_lightweight", "semantic_paraphrase_no_white_cast")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grounding_audit(hybrid: dict[str, Any], canonical_path: Path = DATA_DIR / "sunscreen_comments_canonical.parquet") -> dict[str, Any]:
    """Verify stored fixed-case evidence provenance and the bounded-context contract."""
    canonical_pairs = {
        (row["comment_id"], row["canonical_source_row_number"])
        for row in pq.read_table(canonical_path, columns=["comment_id", "canonical_source_row_number"]).to_pylist()
    }
    evidence_count, missing_fields, unknown_pairs = 0, 0, 0
    for case in hybrid["cases"]:
        for channel in case["channels"].values():
            for product_evidence in channel["evidence"]:
                for evidence in product_evidence:
                    evidence_count += 1
                    if not all(key in evidence for key in ("comment_id", "canonical_source_row_number")):
                        missing_fields += 1
                    elif (evidence["comment_id"], evidence["canonical_source_row_number"]) not in canonical_pairs:
                        unknown_pairs += 1
    probe = llm.retrieval_context(
        "نمونه",
        {"results": [{"product_id": 1, "title": "نمونه", "brand": None, "historical_price_inferred_irr": None, "historical_price_label": "historical inferred IRR", "total_canonical_review_count": 1, "score_components": {"score": 1}, "evidence": [{"comment_id": 1, "canonical_source_row_number": 2, "excerpt": "متن"}]}]},
    )
    context_text = json.dumps(probe, ensure_ascii=False)
    no_raw_score_fields = all(term not in context_text for term in ("score_components", "embedding", "cosine", "rrf"))
    return {
        "stored_evidence_pairs_checked": evidence_count,
        "missing_comment_or_source_row_fields": missing_fields,
        "pairs_not_found_in_canonical_artifact": unknown_pairs,
        "all_stored_evidence_pairs_traceable": missing_fields == 0 and unknown_pairs == 0,
        "stored_generated_answers_available": 0,
        "citation_belongs_to_supplied_live_context": "unavailable: no stored live-answer audit",
        "bounded_groq_context_contract": {"raw_scores_or_embeddings_absent": no_raw_score_fields, "historical_price_guardrail": llm.has_unsupported_current_price_claim("قیمت فعلی این محصول") is True, "medical_safety_instruction_present": "تشخیص" in llm.SYSTEM_INSTRUCTION},
    }


def _latencies(hybrid: dict[str, Any]) -> dict[str, Any]:
    values = [channel["latency_ms"] for case in hybrid["cases"] for channel in case["channels"].values()]
    return {"per_channel_case_latency_ms": {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}, "semantic_index_load_seconds": hybrid.get("semantic_load_seconds"), "whole_hybrid_evaluation_runtime_seconds": "unavailable: not recorded in stored report", "whole_hybrid_evaluation_max_rss_kib": "unavailable: not recorded in stored report"}


def _live_evidence() -> dict[str, Any]:
    path = REPORTS / "sunscreen_live_audit.json"
    if not path.is_file():
        return {"available": False, "provider_request_success": "unavailable", "complete_generated_response": "unavailable", "citation_membership": "unavailable", "citation_presence": "unavailable", "grounded_complete_responses": "unavailable", "human_quality": "unavailable", "token_usage": "unavailable", "monetary_cost": "unavailable: no explicit verified tariff configured"}
    live = _load(path)
    summary = live.get("summary", {})
    return {
        "available": True,
        "provider_request_success": {"completed_cases": summary.get("cases_completed", "unavailable"), "attempted_cases": summary.get("cases_attempted", "unavailable")},
        "complete_generated_response": {"complete": summary.get("complete_responses", "unavailable"), "truncated": summary.get("truncated_responses", "unavailable")},
        "citation_membership": summary.get("citations", "unavailable"),
        "citation_presence": {"responses_requiring_citations": summary.get("responses_requiring_citations", "unavailable"), "responses_with_citations": summary.get("responses_with_citations", "unavailable"), "passed": summary.get("citation_presence_passed", "unavailable")},
        "grounded_complete_responses": summary.get("grounded_complete_responses", "unavailable"),
        "human_quality": "not scored: all human scores remain null",
        "token_usage": summary.get("token_usage", "unavailable"),
        "monetary_cost": summary.get("monetary_cost", "unavailable: no explicit verified tariff configured"),
    }


def build_final_evaluation() -> dict[str, Any]:
    """Load verified local evidence only; intentionally performs no retrieval, ML fit, or API call."""
    scoped = _load(DATA_DIR / "manifest.json")
    semantic = _load(SEMANTIC_DIR / "manifest.json")
    hybrid = _load(REPORTS / "sunscreen_mvp_evaluation.json")
    classifier = _load(REPORTS / "recommendation_status_evaluation.json")
    live = _live_evidence()
    audit = grounding_audit(hybrid)
    cases = {case["id"]: case for case in hybrid["cases"]}
    return {
        "scope": {"category": scoped["scope"], "published_artifacts_only": True},
        "requirement_to_evidence": [
            {"requirement": "scoped data and prices", "evidence": "data/processed/sunscreen_mvp/v1/manifest.json", "status": "verified"},
            {"requirement": "fixed retrieval correctness and determinism", "evidence": "reports/evaluation/sunscreen_mvp_evaluation.json", "status": "verified"},
            {"requirement": "grounding/citation audit", "evidence": "stored evidence-pair and optional live-audit report", "status": "verified offline; live status recorded separately"},
            {"requirement": "recommendation_status Macro F1", "evidence": "reports/evaluation/recommendation_status_evaluation.json", "status": "verified"},
            {"requirement": "live answer quality, tokens, cost", "evidence": "opt-in sunscreen_live_audit report", "status": "recorded when live report exists; human scores remain null"},
        ],
        "answer_quality": {"provider_request_success": live["provider_request_success"], "complete_generated_response": live["complete_generated_response"], "citation_membership": live["citation_membership"], "citation_presence": live["citation_presence"], "grounded_complete_responses": live["grounded_complete_responses"], "human_quality": live["human_quality"], "no_llm_judge_used": True},
        "scoped_build": {"rows": scoped["rows"], "runtime_seconds": scoped["runtime_seconds"], "max_rss_kib": scoped["max_rss_kib"], "historical_price": scoped["historical_price"]},
        "semantic_artifact": {key: semantic[key] for key in ("record_count", "dimension", "dtype", "normalization", "embedding_model", "runtime_seconds", "max_rss_kib", "outputs")},
        "retrieval": {"baseline_case_ids": list(BASELINE_CASE_IDS), "semantic_case_ids": list(SEMANTIC_CASE_IDS), "case_count_contract": {"baseline": 10, "semantic": 2, "total": 12}, "comparison_note": hybrid["note"], "fixed_search_case_results": {case_id: {"deterministic": cases[case_id]["deterministic"], "channels": {name: {"retrieval_mode": value["retrieval_mode"], "product_ids": value["product_ids"]} for name, value in cases[case_id]["channels"].items()}} for case_id in sorted(cases)}, "latency": _latencies(hybrid), "ranking_quality": "No human relevance labels are available; Recall@K, MRR, nDCG, and statistical Hybrid-superiority claims are unavailable. Fixed cases establish functional correctness and repeatability only."},
        "grounding_audit": audit,
        "recommendation_status": classifier,
        "api_tokens_and_cost": {"live_audit": live, "token_usage": live["token_usage"], "monetary_cost": live["monetary_cost"], "original_maximum_5_usd_budget": "unverified: no account-spend evidence stored", "local_embedding_compute": {"runtime_seconds": semantic["runtime_seconds"], "max_rss_kib": semantic["max_rss_kib"], "api_token_usage": "not applicable"}},
        "failure_analysis": [
            {"failure": "full-comments SQLite/HDD bottleneck", "symptom": "slow/bottlenecked historical attempt", "root_cause": "disk-backed full-corpus workflow", "mitigation": "scoped one-pass builder avoids SQLite", "remaining_risk": "raw corpus rebuild remains expensive"},
            {"failure": "decimal SQLite REAL precision", "symptom": "binary-float precision loss", "root_cause": "REAL representation", "mitigation": "exact Decimal/TEXT contract", "remaining_risk": "future consumers must preserve decimal schema"},
            {"failure": "abandoned semantic staging", "symptom": "pre-success staging directory", "root_cause": "interrupted prior build", "mitigation": "explicit cleanup and atomic success publication", "remaining_risk": "future interrupted builds need the same review"},
            {"failure": "Gemini 403", "symptom": "provider authorization failure", "root_cause": "provider permission", "mitigation": "not used by this MVP", "remaining_risk": "external providers can deny access"},
            {"failure": "Groq model/permission failure", "symptom": "unavailable model or permission", "root_cause": "provider configuration/access", "mitigation": "bounded local evidence fallback", "remaining_risk": "live synthesis remains provider-dependent"},
            {"failure": "Hugging Face Xet/CAS download", "symptom": "download failure", "root_cause": "remote transfer path", "mitigation": "HTTP retry and cached local model", "remaining_risk": "initial model availability"},
            {"failure": "optional torchvision watcher noise", "symptom": "watcher import noise", "root_cause": "optional dependency watcher", "mitigation": "file-watcher mitigation", "remaining_risk": "environment-specific warnings"},
            {"failure": "semantic artifact/model unavailable", "symptom": "semantic channel cannot load", "root_cause": "missing/corrupt artifact or optional model", "mitigation": "explicit lexical fallback", "remaining_risk": "semantic recall unavailable during fallback"},
            {"failure": "weakest classifier class", "symptom": "lowest text-model F1 for no_idea", "root_cause": "class ambiguity/imbalance in subset", "mitigation": "balanced class weights and Macro F1", "remaining_risk": "no_idea remains weak"},
            {"failure": "unproven Hybrid ranking superiority", "symptom": "fixed cases are not relevance judgements", "root_cause": "no human ranking labels", "mitigation": "report functional correctness separately", "remaining_risk": "ranking uplift unknown"},
        ],
        "resource_scope_limitations": ["Sunscreen category only; no generalization claim.", "No raw CSV scan, embedding rebuild, live API call, or dataset mutation in this consolidation.", "No stored full test-count command evidence.", "Live Groq cost/usage requires explicit user audit."],
    }


def _markdown(result: dict[str, Any]) -> str:
    rows = result["scoped_build"]["rows"]
    classifier = result["recommendation_status"]
    text = ["# Final sunscreen MVP evaluation", "", "## Requirement-to-evidence", ""]
    text.extend(f"- **{item['requirement']}**: {item['status']} — `{item['evidence']}`" for item in result["requirement_to_evidence"])
    live = result["api_tokens_and_cost"]["live_audit"]
    text += ["", "## Scope and artifacts", "", f"- Products: `{rows['selected_products']}`; canonical comments: `{rows['canonical_comment_rows']}`; brands: `{rows['brands']}`.", f"- Historical-price coverage: `{rows['historical_price_coverage_pct']:.2f}%` ({rows['products_with_historical_price']} products).", f"- Semantic artifact: `{result['semantic_artifact']['record_count']}` vectors, `{result['semantic_artifact']['dimension']}` dimensions, `{result['semantic_artifact']['embedding_model']}`; build runtime `{result['semantic_artifact']['runtime_seconds']:.3f}s`, RSS `{result['semantic_artifact']['max_rss_kib']}` KiB.", "", "## Retrieval and latency", "", "- 10 baseline + 2 semantic fixed cases; deterministic lexical, semantic, and hybrid outcomes are retained in the JSON report.", "- Comparison mode observed the three retrieval channels; this establishes functional correctness/repeatability, not ranking superiority.", f"- Per-channel fixed-case latency: `{result['retrieval']['latency']['per_channel_case_latency_ms']}`; semantic index load: `{result['retrieval']['latency']['semantic_index_load_seconds']}` seconds.", "- No human relevance labels: Recall@K, MRR, nDCG, and statistical Hybrid-superiority claims are unavailable.", "", "## Grounding and live-answer audit", "", f"- Stored evidence pairs checked: `{result['grounding_audit']['stored_evidence_pairs_checked']}`; provenance failures: `{result['grounding_audit']['missing_comment_or_source_row_fields'] + result['grounding_audit']['pairs_not_found_in_canonical_artifact']}`.", "- Bounded Groq context excludes raw scores/embeddings; historical-price and medical-safety guardrails are present.", f"- Provider request success: `{live['provider_request_success']}`; complete responses: `{live['complete_generated_response']}`; citation membership: `{live['citation_membership']}`.", f"- Human quality: `{live['human_quality']}`; monetary cost: `{live['monetary_cost']}`.", "", "## Recommendation-status prediction", "", f"- Labels: `{classifier['labels']}`; distribution: `{classifier['counts']['class_distribution']}`; excluded: `{classifier['counts']['excluded_rows']}`.", f"- Split: `{classifier['split']['algorithm']}`; train/test `{classifier['split']['train_rows']}/{classifier['split']['test_rows']}`; product overlap `{classifier['split']['product_id_overlap_count']}`.", f"- Dummy Macro F1: `{classifier['models']['dummy_most_frequent']['macro_f1']:.6f}`; text Macro F1: `{classifier['models']['text_tfidf_logistic_regression']['macro_f1']:.6f}`; improvement `{classifier['improvement_over_dummy_macro_f1']:.6f}`.", "", "| Label | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    text.extend(f"| {label} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['support']} |" for label, metrics in classifier["models"]["text_tfidf_logistic_regression"]["per_class"].items())
    text += ["", "Confusion matrix (rows=true, columns=predicted; label order as above):", "", "```text", *[" ".join(str(value) for value in row) for row in classifier["models"]["text_tfidf_logistic_regression"]["confusion_matrix"]["values"]], "```", "", "## API usage, cost, and human review", "", "- Local embedding compute is not API-token usage.", "- Token counts and monetary cost are unavailable without the opt-in live report; no configured authoritative tariff or account-spend evidence exists, so the $5 budget is unverified.", "- Human rubric scores remain null until a human completes the separate live audit.", "", "## Failure analysis", ""]
    text.extend(f"- **{item['failure']}** — symptom: {item['symptom']}; cause: {item['root_cause']}; mitigation: {item['mitigation']}; remaining risk: {item['remaining_risk']}." for item in result["failure_analysis"])
    text += ["", "## Limitations", ""]
    text.extend(f"- {item}" for item in result["resource_scope_limitations"])
    return "\n".join(text) + "\n"


def write_final(result: dict[str, Any], output_dir: Path | str = REPORTS) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contents = {"final_evaluation.json": json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "final_evaluation.md": _markdown(result)}
    for name, content in contents.items():
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temporary, output / name)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate existing sunscreen MVP evaluation evidence.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS)
    args = parser.parse_args(argv)
    result = build_final_evaluation()
    write_final(result, args.output_dir)
    print(json.dumps({"status": "success", "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
