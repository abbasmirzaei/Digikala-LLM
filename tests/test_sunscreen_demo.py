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
