"""Focused contracts for offline final consolidation and opt-in live audit boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from digikala_llm import final_evaluation as final
from digikala_llm import sunscreen_live_audit as live
from digikala_llm import sunscreen_llm as llm


def _retrieval() -> dict[str, object]:
    return {
        "retrieval_mode": "hybrid",
        "results": [
            {
                "product_id": 1,
                "title": "ضد آفتاب",
                "brand": "الف",
                "historical_price_inferred_irr": 10,
                "historical_price_label": "historical inferred IRR",
                "total_canonical_review_count": 1,
                "evidence": [{"comment_id": 7, "canonical_source_row_number": 8, "excerpt": "نظر"}],
            }
        ],
    }


class _Retriever:
    def search(self, *_args, **_kwargs): return _retrieval()


def test_final_report_schema_is_truthful_and_atomic(tmp_path: Path) -> None:
    result = final.build_final_evaluation()
    assert result["retrieval"]["case_count_contract"] == {"baseline": 10, "semantic": 2, "total": 12}
    assert result["grounding_audit"]["all_stored_evidence_pairs_traceable"] is True
    assert result["api_tokens_and_cost"]["monetary_cost"].startswith("unavailable")
    assert "unavailable" in result["retrieval"]["latency"]["whole_hybrid_evaluation_runtime_seconds"]
    final.write_final(result, tmp_path)
    assert json.loads((tmp_path / "final_evaluation.json").read_text())["scope"]["published_artifacts_only"]
    assert (tmp_path / "final_evaluation.md").is_file()


def test_live_audit_missing_key_makes_no_calls_and_human_scores_are_null() -> None:
    result = live.run_live_audit(key_present=False)
    assert result["status"] == "missing_api_key" and result["api_calls_made"] == 0
    assert all(item["score"] is None for item in live.human_review_rubric().values())


def test_live_audit_hard_bounds_valid_citations_usage_and_no_answer_persistence(tmp_path: Path) -> None:
    calls = []

    class _Completions:
        def create(self, **_kwargs):
            calls.append(1)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="پاسخ [نظر 7، ردیف 8]"), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=12, total_tokens=23),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    result = live.run_live_audit(retriever=_Retriever(), client_factory=lambda: client, key_present=True)
    assert result["status"] == "completed"
    assert result["api_calls_made"] == live.MAX_LIVE_CALLS == len(calls) == 5
    assert result["cases"][0]["token_usage"] == {"prompt_tokens": 11, "completion_tokens": 12, "total_tokens": 23}
    assert result["cases"][0]["citation_membership_passed"] is True
    assert result["cases"][0]["citation_required"] is True
    assert result["cases"][0]["citation_presence_passed"] is True
    assert result["cases"][0]["grounding_passed"] is True
    assert result["cases"][0]["valid_citation_count"] == 1
    assert result["summary"]["token_usage"]["total_tokens"] == 115
    assert result["summary"]["responses_requiring_citations"] == 5
    assert result["summary"]["grounded_complete_responses"] == 5
    assert "answer_text" not in str(result)
    live._atomic_reports(result, tmp_path)
    assert "پاسخ [نظر" not in (tmp_path / "sunscreen_live_audit.json").read_text()


def test_live_audit_marks_invented_citations_without_persisting_key() -> None:
    class _Completions:
        def __init__(self, answer: str): self.answer = answer
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    invented = live.run_live_audit(retriever=_Retriever(), client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions("[نظر 9، ردیف 10]"))), key_present=True)
    assert invented["cases"][0]["invalid_citation_count"] == 1
    assert invented["cases"][0]["citation_membership_passed"] is False
    assert invented["cases"][0]["grounding_passed"] is False
    assert "GROQ_API_KEY" not in json.dumps(invented)


def test_grounding_requires_visible_citation_when_evidence_was_supplied() -> None:
    context = llm.retrieval_context("سبک", _retrieval())
    audit = live._grounding("بدون ارجاع", context, response_complete=True)
    assert audit["citation_required"] is True
    assert audit["citation_count"] == 0
    assert audit["citation_presence_passed"] is False
    assert audit["citation_membership_passed"] is True
    assert audit["grounding_passed"] is False


def test_grounding_does_not_require_citation_without_evidence() -> None:
    context = {"kind": "recommendation", "products": []}
    audit = live._grounding("بدون ارجاع", context, response_complete=True)
    assert audit["citation_required"] is False
    assert audit["citation_presence_passed"] is True
    assert audit["grounding_passed"] is True


def test_live_audit_uses_displayed_citation_completion_for_grounding() -> None:
    context = llm.retrieval_context("سبک", _retrieval())
    final_text, reason = llm.finalize_grounded_answer("بدون قالب ارجاع", context)
    assert reason is None and final_text is not None
    audit = live._grounding(final_text, context, response_complete=True)
    assert audit["citation_count"] == audit["valid_citation_count"] == 1
    assert audit["grounding_passed"] is True


def test_live_audit_never_persists_or_prints_api_key(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    class _Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    monkeypatch.setenv("GROQ_API_KEY", "audit-secret-must-not-appear")
    result = live.run_live_audit(
        retriever=_Retriever(),
        client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())),
    )
    assert "audit-secret-must-not-appear" not in json.dumps(result)
    assert "audit-secret-must-not-appear" not in capsys.readouterr().out


def test_live_audit_marks_truncated_finish_reason_incomplete() -> None:
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="پاسخ"), finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1000, total_tokens=1001),
            )

    result = live.run_live_audit(retriever=_Retriever(), client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), key_present=True)
    assert result["cases"][0]["status"] == "provider_request_success"
    assert result["cases"][0]["response_complete"] is False
    assert result["summary"]["truncated_responses"] == 5
    assert result["api_calls_made"] == len(calls) == live.MAX_LIVE_PROVIDER_REQUESTS
    assert [call["max_tokens"] for call in calls] == [
        llm.MAX_COMPLETION_TOKENS,
        llm.RETRY_COMPLETION_TOKENS,
    ] * live.MAX_LIVE_CALLS
    assert llm.RETRY_BREVITY_INSTRUCTION in calls[1]["messages"][0]["content"]


def test_legacy_sanitization_removes_answer_text_and_marks_token_limit_incomplete() -> None:
    legacy = {"cases": [{"id": "live_5", "status": "completed", "answer_text": "secret", "prompt": "secret", "provider": "Groq", "model": "model", "retrieval_mode": "hybrid", "product_ids": [1], "latency_ms": 1.0, "token_usage": {"prompt_tokens": 1, "completion_tokens": llm.MAX_COMPLETION_TOKENS, "total_tokens": llm.MAX_COMPLETION_TOKENS + 1}, "grounding_safety_checks": {"visible_citations": [{"comment_id": 7, "canonical_source_row_number": 8}], "citations_belong_to_supplied_context": True}, "human_review": live.human_review_rubric()}]}
    sanitized = live.sanitize_legacy_result(legacy)
    case = sanitized["cases"][0]
    assert "secret" not in json.dumps(sanitized)
    assert case["response_complete"] is False
    assert case["finish_reason"] == "unavailable_legacy_at_token_limit"


def test_live_audit_stops_on_provider_failure_and_marks_missing_usage_unavailable() -> None:
    class _Completions:
        def create(self, **_kwargs): raise RuntimeError("quota")

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    result = live.run_live_audit(retriever=_Retriever(), client_factory=lambda: client, key_present=True)
    assert result["status"] == "stopped_on_provider_failure"
    assert result["api_calls_made"] == 1 and len(result["cases"]) == 1
    assert live._usage(SimpleNamespace(usage=None))["total_tokens"] == "unavailable"
