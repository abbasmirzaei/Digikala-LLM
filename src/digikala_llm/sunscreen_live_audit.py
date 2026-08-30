"""Explicitly invoked, maximum-five-call Groq audit for human review; never run by offline CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from digikala_llm import sunscreen_llm as llm
from digikala_llm.sunscreen_hybrid import HybridSunscreenRetriever

DEFAULT_OUTPUT_DIR = Path("reports/evaluation")
LIVE_PROMPTS = (
    "ضد آفتاب سبک و بدون چربی پیشنهاد بده",
    "برای پوست چرب چه شواهدی در نظر کاربران هست؟",
    "ضد آفتاب بدون رنگ با نظر خریداران زیاد",
    "گزینه‌های با قیمت تاریخی مناسب را نشان بده",
    "روی پوست احساس سنگینی نداشته باشد",
)
MAX_LIVE_CALLS = len(LIVE_PROMPTS)
MAX_LIVE_PROVIDER_REQUESTS = MAX_LIVE_CALLS * 2
_CITATION = re.compile(r"\[نظر\s*(\d+)\s*[،,]\s*ردیف\s*(\d+)\]")


def human_review_rubric() -> dict[str, dict[str, Any]]:
    """A transparent rubric deliberately left unscored until a human reviews each answer."""
    return {
        "relevance": {"score": None, "scale": "1-5", "criterion": "Addresses the Persian request."},
        "clarity": {"score": None, "scale": "1-5", "criterion": "Clear, readable Persian."},
        "evidence_use": {"score": None, "scale": "1-5", "criterion": "Claims use supplied evidence."},
        "safety": {"score": None, "scale": "1-5", "criterion": "No medical or current-price claim."},
    }


def _usage(response: Any) -> dict[str, int | str]:
    usage = getattr(response, "usage", None)
    getter = usage.get if isinstance(usage, dict) else lambda key, default=None: getattr(usage, key, default)
    return {
        key: (value if isinstance(value := getter(key), int) else "unavailable")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _combined_usage(*responses: Any) -> dict[str, int | str]:
    usages = [_usage(response) for response in responses]
    return {
        key: sum(usage[key] for usage in usages)
        if all(isinstance(usage[key], int) for usage in usages)
        else "unavailable"
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _completion(client: Any, context: dict[str, Any], max_tokens: int, *, retry: bool) -> Any:
    instruction = llm.SYSTEM_INSTRUCTION + ("\n" + llm.RETRY_BREVITY_INSTRUCTION if retry else "")
    return client.chat.completions.create(
        model=llm.llm_model(),
        messages=[{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        temperature=0,
        max_tokens=max_tokens,
    )


def _grounding(
    answer: str, context: dict[str, Any], *, response_complete: bool
) -> dict[str, Any]:
    citations = [
        {"comment_id": int(comment), "canonical_source_row_number": int(row)}
        for comment, row in _CITATION.findall(answer)
    ]
    supplied = {
        (item["comment_id"], item["canonical_source_row_number"])
        for product in context["products"]
        for group in (
            (product.get("user_review_evidence", []),)
            if context["kind"] == "recommendation"
            else (
                product.get("positive_user_review_evidence", []),
                product.get("critical_user_review_evidence", []),
            )
        )
        for item in group
    }
    unsafe_medical = bool(re.search(r"(?:تشخیص|درمان|تجویز)", answer))
    valid = [
        item
        for item in citations
        if (item["comment_id"], item["canonical_source_row_number"]) in supplied
    ]
    citation_required = bool(supplied)
    citation_presence_passed = not citation_required or bool(citations)
    citation_membership_passed = len(valid) == len(citations)
    return {
        "citation_count": len(citations),
        "valid_citation_count": len(valid),
        "invalid_citation_count": len(citations) - len(valid),
        "citation_required": citation_required,
        "citation_presence_passed": citation_presence_passed,
        "citation_membership_passed": citation_membership_passed,
        "grounding_passed": (
            response_complete and citation_presence_passed and citation_membership_passed
        ),
        "cited_pairs": citations,
        "no_current_price_claim": not llm.has_unsupported_current_price_claim(answer),
        "no_medical_claim": not unsafe_medical,
    }


def _response_complete(finish_reason: object) -> bool:
    return isinstance(finish_reason, str) and not llm.is_truncated_finish_reason(finish_reason)


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [case for case in cases if case["status"] == "provider_request_success"]
    latencies = sorted(case["latency_ms"] for case in successful)
    tokens = {
        key: [case["token_usage"][key] for case in successful]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }

    def total(values: list[int | str]) -> int | str:
        return sum(values) if values and all(isinstance(value, int) for value in values) else "unavailable"

    def percentile(percent: float) -> float | str:
        if not latencies:
            return "unavailable"
        index = (len(latencies) - 1) * percent
        low, high = int(index), min(int(index) + 1, len(latencies) - 1)
        return latencies[low] + (latencies[high] - latencies[low]) * (index - low)

    presence_values = [case.get("citation_presence_passed") for case in successful]
    return {
        "cases_attempted": len(cases),
        "cases_completed": len(successful),
        "complete_responses": sum(case["response_complete"] for case in successful),
        "truncated_responses": sum(not case["response_complete"] for case in successful),
        "latency_ms": {"total": sum(latencies) if latencies else "unavailable", "mean": sum(latencies) / len(latencies) if latencies else "unavailable", "p50": percentile(0.5), "p95": percentile(0.95)},
        "token_usage": {"total_prompt_tokens": total(tokens["prompt_tokens"]), "total_completion_tokens": total(tokens["completion_tokens"]), "total_tokens": total(tokens["total_tokens"])},
        "citations": {"total": sum(case["citation_count"] for case in successful), "valid": sum(case["valid_citation_count"] for case in successful), "invalid": sum(case["invalid_citation_count"] for case in successful), "citation_membership_passed": all(case["citation_membership_passed"] for case in successful)},
        "responses_requiring_citations": sum(
            case.get("citation_required") is True for case in successful
        ),
        "responses_with_citations": sum(case["citation_count"] > 0 for case in successful),
        "citation_presence_passed": (
            all(presence_values) if presence_values and None not in presence_values else "unavailable"
        ),
        "grounded_complete_responses": sum(
            case.get("grounding_passed") is True for case in successful
        ),
        "monetary_cost": "unavailable: no explicit verified tariff configured",
        "human_scores": "all null until human review",
    }


def sanitize_legacy_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove legacy persisted answers while preserving only auditable metadata and provenance."""
    migrated: list[dict[str, Any]] = []
    for old in result.get("cases", []):
        if old.get("status") != "completed":
            migrated.append({"id": old["id"], "status": "provider_failure", "provider_failure": old.get("provider_failure", "unavailable"), "human_scores": old.get("human_review", human_review_rubric())})
            continue
        audit = old.get("grounding_safety_checks", {})
        pairs = audit.get("visible_citations", [])
        usage = old.get("token_usage", {})
        completion_tokens = usage.get("completion_tokens")
        at_limit = isinstance(completion_tokens, int) and completion_tokens >= llm.MAX_COMPLETION_TOKENS
        valid = len(pairs) if audit.get("citations_belong_to_supplied_context") is True else 0
        migrated.append({"id": old["id"], "status": "provider_request_success", "provider": old.get("provider", "Groq"), "model": old.get("model", "unavailable"), "retrieval_mode": old.get("retrieval_mode", "unavailable"), "product_ids": old.get("product_ids", []), "finish_reason": "unavailable_legacy_at_token_limit" if at_limit else "unavailable_legacy", "response_complete": not at_limit, "latency_ms": old.get("latency_ms", "unavailable"), "token_usage": {key: usage.get(key, "unavailable") for key in ("prompt_tokens", "completion_tokens", "total_tokens")}, "citation_count": len(pairs), "valid_citation_count": valid, "invalid_citation_count": len(pairs) - valid, "citation_membership_passed": audit.get("citations_belong_to_supplied_context") is True, "cited_pairs": pairs, "human_scores": old.get("human_review", human_review_rubric())})
    sanitized = {key: value for key, value in result.items() if key not in {"cases", "summary"}}
    sanitized["cases"] = migrated
    sanitized["summary"] = _summary(migrated)
    sanitized["legacy_migration_note"] = "Answer text and prompts were removed. Legacy finish reasons were not recorded; token-limit cases are conservatively incomplete."
    return sanitized


def _atomic_reports(result: dict[str, Any], output_dir: Path | str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    markdown = ["# Sunscreen live Groq audit", "", "Human-review scores remain null until supplied by a reviewer.", ""]
    for case in result["cases"]:
        markdown.extend([f"## {case['id']}", "", f"- provider request: `{case['status']}`", f"- complete response: `{case.get('response_complete', 'unavailable')}`", f"- citation membership: `{case.get('citation_membership_passed', 'unavailable')}`", f"- retrieval mode: `{case.get('retrieval_mode', 'unavailable')}`", f"- latency ms: `{case.get('latency_ms', 'unavailable')}`", ""])
    contents = {
        "sunscreen_live_audit.json": json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "sunscreen_live_audit.md": "\n".join(markdown) + "\n",
    }
    for name, content in contents.items():
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temporary, output / name)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise


def run_live_audit(
    *,
    retriever: Any | None = None,
    client_factory: Callable[[], Any] | None = None,
    key_present: bool | None = None,
) -> dict[str, Any]:
    """Run at most one provider completion for each fixed prompt, stopping on provider failure."""
    key_present = bool(os.environ.get(llm.GROQ_API_KEY_ENV)) if key_present is None else key_present
    result: dict[str, Any] = {
        "kind": "opt_in_live_groq_audit",
        "maximum_api_calls": MAX_LIVE_PROVIDER_REQUESTS,
        "api_calls_made": 0,
        "monetary_cost": "unavailable: no explicit verified tariff configured",
        "human_review": "required; all rubric scores are null by default",
        "cases": [],
    }
    if not key_present:
        result["status"] = "missing_api_key"
        result["summary"] = _summary([])
        return result
    retriever = retriever or HybridSunscreenRetriever()
    try:
        if client_factory is None:
            from groq import Groq

            client = Groq()
        else:
            client = client_factory()
    except (ImportError, OSError, RuntimeError) as error:
        result["status"] = "provider_unavailable"
        result["provider_failure"] = llm._fallback_reason(error)
        result["summary"] = _summary([])
        return result
    for number, prompt in enumerate(LIVE_PROMPTS, 1):
        retrieval = retriever.search(prompt, mode="hybrid", limit=3)
        context = llm.retrieval_context(prompt, retrieval)
        started = time.perf_counter()
        try:
            result["api_calls_made"] += 1
            response = _completion(client, context, llm.MAX_COMPLETION_TOKENS, retry=False)
            responses = [response]
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if llm.is_truncated_finish_reason(finish_reason):
                result["api_calls_made"] += 1
                response = _completion(client, context, llm.RETRY_COMPLETION_TOKENS, retry=True)
                responses.append(response)
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            response_complete = _response_complete(finish_reason)
            raw_text = str(response.choices[0].message.content or "").strip()
            final_answer_source = "groq"
            if not raw_text:
                text = llm.deterministic_fallback(context, "truncated_response")
                final_answer_source = "deterministic_fallback"
            else:
                text, fallback_reason = llm.finalize_grounded_answer(raw_text, context)
            if raw_text and fallback_reason is not None:
                text = llm.deterministic_fallback(context, fallback_reason)
                final_answer_source = "deterministic_fallback"
            assert text is not None
            audit = _grounding(text, context, response_complete=response_complete)
            result["cases"].append({"id": f"live_{number}", "status": "provider_request_success", "provider": "Groq", "model": llm.llm_model(), "retrieval_mode": retrieval["retrieval_mode"], "product_ids": [item["product_id"] for item in retrieval["results"]], "finish_reason": finish_reason or "unavailable", "response_complete": response_complete, "final_answer_source": final_answer_source, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "token_usage": _combined_usage(*responses), **audit, "human_scores": human_review_rubric()})
        except Exception as error:  # noqa: BLE001 -- external provider errors must halt the remaining calls safely.
            result["cases"].append({"id": f"live_{number}", "status": "provider_failure", "provider_failure": llm._fallback_reason(error), "human_scores": human_review_rubric()})
            result["status"] = "stopped_on_provider_failure"
            break
    else:
        result["status"] = "completed"
    result["summary"] = _summary(result["cases"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicit opt-in live five-prompt Groq audit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = run_live_audit()
    _atomic_reports(result, args.output_dir)
    print(json.dumps({"status": result["status"], "api_calls_made": result["api_calls_made"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
