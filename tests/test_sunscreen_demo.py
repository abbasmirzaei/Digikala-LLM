"""Focused tests for the conversational Streamlit presentation adapters."""

from __future__ import annotations

from digikala_llm import sunscreen_demo as demo


def _result() -> dict[str, object]:
    evidence = {"comment_id": 11, "canonical_source_row_number": 31, "excerpt": "سبک و خوشایند"}
    return {
        "results": [
            {
                "product_id": 1,
                "title": "ضد آفتاب الف",
                "brand": "الف",
                "historical_price_inferred_irr": 125_000,
                "historical_price_label": "historical inferred IRR",
                "total_canonical_review_count": 8,
                "score_components": {"score": 100},
                "evidence": [evidence],
            }
        ]
    }


def test_conversational_presenter_keeps_price_safety_and_citations() -> None:
    row = _result()["results"][0]
    assert "ریال" in demo.format_search_result(row)
    assert "نه قیمت فعلی" in demo.format_search_result(row)
    assert "[نظر 11، ردیف 31]" in demo.card_summary(row).replace("\u2066", "").replace("\u2069", "")
    for text in ("تاریخی", "نه قیمت فعلی", "دیدگاه کاربران", "تشخیص پزشکی", "تضمین سازگاری"):
        assert text in demo.SAFETY_NOTICE
    assert "دادهٔ منتشرشده یافت نشد" in demo.missing_artifact_message(FileNotFoundError("missing"))


def test_normal_presentation_is_compact_and_has_no_raw_or_debug_output() -> None:
    source = __import__(demo.__name__, fromlist=["unused"]).__loader__.get_source(demo.__name__)
    assert "دستیار انتخاب ضدآفتاب" in source
    assert "پیشنهاد بده" in source
    assert "تنظیمات پیشرفته" in source
    assert 'st.expander("تنظیمات پیشرفته", expanded=False)' in source
    assert "st.json" not in source
    assert "st.dataframe" not in source
    assert "score_components" not in source
    assert "جزئیات فنی" not in source
    assert "<br" not in source.casefold()
    assert 'result["results"][:MAX_PRIMARY_CARDS]' in source
    assert "MAX_PRIMARY_CARDS = 3" in source
    assert "st.columns(MAX_PRIMARY_CARDS)" in source
    assert "provider-badge" in source
    assert "hero-mark" in source
    assert "جمع‌بندی دستیار" in source
    assert "پیشنهاد اول" in source
    assert "گزینهٔ جایگزین" in source
    assert "st.columns([4, 1])" in source
    assert "انتخاب‌شده برای مقایسه" in source
    assert "@media (max-width: 640px)" in demo.APP_CSS
    assert "--navy" in demo.APP_CSS and "--teal" in demo.APP_CSS
    assert "--canvas" in demo.APP_CSS and "max-width:1240px" in demo.APP_CSS
    assert "Streamlit test IDs below are presentation hooks" in source
    assert len(demo.EXAMPLE_QUESTIONS) == 4


def test_comparison_adapter_preserves_measurable_fields_without_overall_winner() -> None:
    rows = demo._comparison_rows(
        {
            "products": [
                {
                    "title": "الف",
                    "brand": "ب",
                    "historical_price_inferred_irr": 3_000,
                    "canonical_review_count": 7,
                    "valid_average_rating": 4.5,
                    "positive_recommendation_share": 0.8,
                }
            ]
        }
    )
    assert rows[0]["قیمت تاریخی"].endswith("نه قیمت فعلی)")
    assert rows[0]["تعداد نظر"] == 7


def test_card_summary_is_compact_and_keeps_a_bidi_safe_citation() -> None:
    row = _result()["results"][0]
    row["evidence"][0]["excerpt"] = "متن " * 100
    summary = demo.card_summary(row)
    visible = summary.replace("\u2066", "").replace("\u2069", "")
    assert len(summary) < 230
    assert "[نظر 11، ردیف 31]" in visible


def test_retrieval_mode_badges_are_deterministic_and_never_mislabel_fallback() -> None:
    assert demo.retrieval_mode_label("hybrid") == "Hybrid RAG: واژگانی + معنایی"
    assert demo.retrieval_mode_label("semantic") == "بازیابی معنایی"
    assert demo.retrieval_mode_label("lexical_fallback") == "بازیابی واژگانی (fallback)"
    assert demo.retrieval_mode_label("lexical") is None
    assert demo.retrieval_mode_label(None) is None


def test_answer_header_renders_provider_and_actual_retrieval_badges() -> None:
    class _Container:
        def __enter__(self): return self
        def __exit__(self, *_args): return False

    class _Streamlit:
        def __init__(self): self.markdowns: list[str] = []
        def container(self, **_kwargs): return _Container()
        def markdown(self, value, **_kwargs): self.markdowns.append(value)
        def caption(self, _value): pass

    st = _Streamlit()
    demo._display_answer(st, {"source": "groq", "text": "پاسخ"}, "semantic")
    header = next(value for value in st.markdowns if "answer-badges" in value)
    assert "Groq" in header and "بازیابی معنایی" in header

    fallback = _Streamlit()
    demo._display_answer(
        fallback,
        {"source": "deterministic_fallback", "text": "پاسخ"},
        "lexical_fallback",
    )
    fallback_header = next(value for value in fallback.markdowns if "answer-badges" in value)
    assert "پاسخ محلی" in fallback_header
    assert "بازیابی واژگانی (fallback)" in fallback_header
    assert "Hybrid RAG" not in fallback_header
