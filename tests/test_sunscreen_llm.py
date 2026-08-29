"""Offline tests for bounded, grounded Groq synthesis."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from digikala_llm import sunscreen_llm as llm


def _retrieval(products: int = 1, evidence: int = 1) -> dict[str, object]:
    rows = []
    for product_id in range(1, products + 1):
        rows.append(
            {
                "product_id": product_id,
                "title": f"ضد آفتاب {product_id}",
                "brand": "الف",
                "historical_price_inferred_irr": 1000 * product_id,
                "historical_price_label": "historical inferred IRR",
                "total_canonical_review_count": 10,
                "score_components": {"score": 999},
                "evidence": [
                    {
                        "comment_id": product_id * 100 + index,
                        "canonical_source_row_number": product_id * 1000 + index,
                        "excerpt": "متن " * 100,
                    }
                    for index in range(evidence)
                ],
            }
        )
    return {"results": rows}


def _comparison() -> dict[str, object]:
    evidence = {"comment_id": 88, "canonical_source_row_number": 99, "excerpt": "نظر خوب"}
    product = {
        "product_id": 1,
        "title": "الف",
        "brand": "الف",
        "historical_price_inferred_irr": 1000,
        "historical_price_label": "historical inferred IRR",
        "canonical_review_count": 10,
        "buyer_review_count": 8,
        "buyer_review_percentage": 80.0,
        "valid_average_rating": 4.0,
        "valid_rating_count": 9,
        "positive_recommendation_share": 0.75,
        "price_difference_vs_compared_inferred_irr": {"2": -10},
        "positive_evidence": [evidence],
        "critical_evidence": [],
    }
    return {"products": [product, {**product, "product_id": 2, "title": "ب"}]}


def test_model_configuration_reads_only_expected_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(llm.GROQ_MODEL_ENV, raising=False)
    assert llm.llm_model() == llm.DEFAULT_GROQ_MODEL
    monkeypatch.setenv(llm.GROQ_MODEL_ENV, "test-model")
    assert llm.llm_model() == "test-model"
    monkeypatch.delenv(llm.GROQ_API_KEY_ENV, raising=False)
    answer = llm.GroundedAssistant().answer_recommendation("سبک", _retrieval())
    assert "never-show-this" not in str(answer)


def test_retrieval_context_is_bounded_and_score_independent() -> None:
    context = llm.retrieval_context("سبک", _retrieval(products=5, evidence=5))
    assert len(context["products"]) == llm.MAX_PRODUCTS_IN_PROMPT
    assert all(
        len(product["user_review_evidence"]) == llm.MAX_EXCERPTS for product in context["products"]
    )
    assert all(
        len(item["excerpt"]) <= llm.MAX_EXCERPT_CHARS
        for product in context["products"]
        for item in product["user_review_evidence"]
    )
    assert "score_components" not in str(context)
    assert "skin_type" not in str(context)


def test_grounded_instruction_citation_and_mocked_groq_call(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    class _Completions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="نظر کاربران مثبت است [نظر 100، ردیف 1000]")
                    )
                ]
            )

    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    ).answer_recommendation("سبک", _retrieval())
    assert answer["source"] == "groq"
    assert answer["reason"] == "api_success"
    assert "[نظر 100، ردیف 1000]" in answer["text"]
    assert (
        "فقط از کاتالوگ" in llm.SYSTEM_INSTRUCTION
        and "تشخیص" in llm.SYSTEM_INSTRUCTION
        and "SPF" in llm.SYSTEM_INSTRUCTION
    )
    assert "score_components" not in str(captured["messages"])
    assert captured["messages"][0]["content"] == llm.SYSTEM_INSTRUCTION
    assert "models" not in str(captured)
    assert "شواهد قابل بررسی" not in answer["text"]


def test_groq_adapter_never_discovers_models() -> None:
    source = __import__(llm.__name__, fromlist=["unused"]).__loader__.get_source(llm.__name__)
    assert ".models.list" not in source
    assert ".models.retrieve" not in source


def test_missing_key_and_api_error_have_explicit_local_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(llm.GROQ_API_KEY_ENV, raising=False)
    missing = llm.GroundedAssistant().answer_recommendation("سبک", _retrieval())
    assert missing["source"] == "deterministic_fallback" and missing["reason"] == "missing_api_key"
    assert "بدون مدل زبانی" in missing["text"]
    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "never-show-this")
    failed = llm.GroundedAssistant(
        lambda: (_ for _ in ()).throw(RuntimeError("quota"))
    ).answer_recommendation("سبک", _retrieval())
    assert failed["source"] == "deterministic_fallback"
    assert failed["reason"] == "quota_or_rate_limit"
    assert "never-show-this" not in str(failed)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (TimeoutError("timeout"), "timeout"),
        (PermissionError("permission"), "permission_or_api_error"),
        (RuntimeError("quota"), "quota_or_rate_limit"),
        (ImportError("groq"), "missing_sdk"),
    ],
)
def test_timeout_api_quota_model_and_sdk_errors_fall_back(  # type: ignore[no-untyped-def]
    monkeypatch, failure: Exception, reason: str
) -> None:
    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: (_ for _ in ()).throw(failure)
    ).answer_recommendation("سبک", _retrieval())
    assert answer["source"] == "deterministic_fallback"
    assert answer["reason"] == reason


def test_unavailable_model_error_falls_back_without_provider_details(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class ModelNotFound(Exception):
        status_code = 404

    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: (_ for _ in ()).throw(ModelNotFound("model unavailable"))
    ).answer_recommendation("سبک", _retrieval())
    assert answer["reason"] == "unavailable_model"
    assert "model unavailable" not in answer["text"]


def test_comparison_synthesis_adapter_is_bounded_and_traceable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(llm.GROQ_API_KEY_ENV, raising=False)
    answer = llm.GroundedAssistant().answer_comparison("ارزش خرید", _comparison())
    assert answer["context"]["kind"] == "comparison"
    assert "[نظر 88، ردیف 99]" in answer["text"]
    assert "برندهٔ کلی" not in answer["text"]


def test_length_finish_reason_retries_once_and_never_accepts_truncated_text(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, object]] = []

    class _Completions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="پاسخ ناتمام (رنگ و لایه‌"),
                            finish_reason="length",
                        )
                    ]
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="پاسخ کامل است [نظر 100، ردیف 1000]"),
                        finish_reason="stop",
                    )
                ]
            )

    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    ).answer_recommendation("سبک", _retrieval())
    assert answer["source"] == "groq"
    assert answer["text"] == "پاسخ کامل است [نظر 100، ردیف 1000]"
    assert len(calls) == 2
    assert [call["max_tokens"] for call in calls] == [
        llm.MAX_COMPLETION_TOKENS,
        llm.RETRY_COMPLETION_TOKENS,
    ]


def test_second_length_response_uses_complete_non_llm_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    class _Completions:
        def create(self, **_kwargs: object) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="پاسخ ناتمام (رنگ و لایه‌"),
                        finish_reason="length",
                    )
                ]
            )

    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    ).answer_recommendation("سبک", _retrieval())
    assert answer["source"] == "deterministic_fallback"
    assert answer["reason"] == "truncated_response"
    assert "رنگ و لایه‌" not in answer["text"]
    assert calls == 2


def test_citation_normalization_and_bounded_appendix() -> None:
    context = llm.retrieval_context("سبک", _retrieval(products=3, evidence=3))
    normalized = llm.normalize_citations("نظر [COMMENT_ID 100، ردیف 1000]", context)
    assert normalized == "نظر [نظر 100، ردیف 1000]"
    assert "COMMENT_ID" not in normalized
    referenced = "؛ ".join(
        f"نظر شماره {comment_id}" for comment_id in llm._citations_by_comment_id(context)
    )
    appendix = llm.citation_appendix(referenced, context)
    assert appendix.count("[نظر ") == llm.MAX_APPENDED_CITATIONS
    assert appendix.count("[نظر 100، ردیف 1000]") == 1
    assert llm.citation_appendix(normalized, context) == ""


def test_instruction_retains_grounding_and_concise_complete_safety_contract() -> None:
    for term in (
        "فقط از کاتالوگ",
        "فقط نامزدهای",
        "به کاربران نسبت بده",
        "سازگاری پزشکی",
        "۲۵۰ تا ۴۵۰",
        "جمله‌های کامل",
        "COMMENT_ID",
    ):
        assert term in llm.SYSTEM_INSTRUCTION


@pytest.mark.parametrize(
    "claim",
    [
        "قیمت فعلی این محصول بیشتر است.",
        "این محصول اکنون ارزان‌تر است.",
        "قیمت امروز پایین‌تر شده است.",
        "قیمت امروز آن افزایش یافته است.",
        "افزایش قیمت در بازار فعلی دیده می‌شود.",
        "Its current market price is higher.",
        "Today's price decreased.",
        "It is cheaper today.",
    ],
)
def test_unsupported_current_price_claim_uses_complete_local_fallback(  # type: ignore[no-untyped-def]
    monkeypatch, claim: str
) -> None:
    class _Completions:
        def create(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=claim), finish_reason="stop")
                ]
            )

    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    ).answer_recommendation("سبک", _retrieval())
    assert answer["source"] == "deterministic_fallback"
    assert answer["reason"] == "unsupported_current_price_claim"
    assert "بدون مدل زبانی" in answer["text"]
    assert claim not in answer["text"]


def test_current_price_guard_and_instruction_cover_persian_and_english_variants() -> None:
    assert llm.has_unsupported_current_price_claim("بازار امروز تغییر دارد")
    assert llm.has_unsupported_current_price_claim("The current price increased")
    assert not llm.has_unsupported_current_price_claim("قیمت تاریخی استنباط‌شده به ریال")
    for term in ("قیمت تاریخی", "زمان حال", "تغییرات قیمت"):
        assert term in llm.SYSTEM_INSTRUCTION


@pytest.mark.parametrize(
    "statement",
    [
        "قیمت تاریخی استنباط‌شده است، نه قیمت فعلی.",
        "این مبلغ قیمت فعلی بازار نیست.",
        "داده‌ای درباره قیمت امروز در دسترس نیست.",
        "نمی‌توان قیمت فعلی را از این داده تعیین کرد.",
    ],
)
def test_safe_current_price_disclaimers_are_accepted_as_groq_output(  # type: ignore[no-untyped-def]
    monkeypatch, statement: str
) -> None:
    class _Completions:
        def create(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=statement), finish_reason="stop")
                ]
            )

    assert not llm.has_unsupported_current_price_claim(statement)
    monkeypatch.setenv(llm.GROQ_API_KEY_ENV, "test-key")
    answer = llm.GroundedAssistant(
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    ).answer_recommendation("سبک", _retrieval())
    assert answer["source"] == "groq"
    assert answer["reason"] == "api_success"
    assert answer["text"] == statement
