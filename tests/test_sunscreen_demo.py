"""Focused tests for the conversational Streamlit presentation adapters."""

from __future__ import annotations

from pathlib import Path

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
    assert "قیمت:" in demo.format_search_result(row)
    assert "نه قیمت فعلی" not in demo.format_search_result(row)
    assert "[نظر 11، ردیف 31]" in demo.card_summary(row).replace("\u2066", "").replace("\u2069", "")
    assert demo.format_price(1_730_000) == "قیمت: \u2066۱٬۷۳۰٬۰۰۰\u2069 ریال"
    assert demo.format_price(6_998_000) == "قیمت: \u2066۶٬۹۹۸٬۰۰۰\u2069 ریال"
    assert "دادهٔ منتشرشده یافت نشد" in demo.missing_artifact_message(FileNotFoundError("missing"))


def test_normal_presentation_is_compact_and_has_no_raw_or_debug_output() -> None:
    source = __import__(demo.__name__, fromlist=["unused"]).__loader__.get_source(demo.__name__)
    assert "دستیار انتخاب ضدآفتاب" in source
    assert "پیشنهاد بده" in source
    assert "تنظیمات پیشرفته" not in source
    assert "st.sidebar" not in source
    assert "st.json" not in source
    assert "st.dataframe" not in source
    assert "score_components" not in source
    assert "جزئیات فنی یا تأثیرات پزشکی در دسترس نیست" not in source
    assert "<br" not in source.casefold()
    assert 'result["results"][:MAX_PRIMARY_CARDS]' in source
    assert "MAX_PRIMARY_CARDS = 3" in source
    assert "st.columns(MAX_PRIMARY_CARDS)" in source
    assert "provider-badge" in source
    assert "پاسخ دستیار" in source
    assert "جمع‌بندی دستیار" not in source
    assert "پیشنهاد اول" in source
    assert "گزینهٔ جایگزین" in source
    assert "چه ضدآفتابی می‌خواهید؟" in source
    assert "st.columns([2, 6, 2])" in source
    assert "st.columns([5, 2], vertical_alignment=\"bottom\")" in source
    assert 'st.popover(' in source
    assert '"نمونه"' in source
    assert 'type="tertiary"' in source
    assert "width=110" in source
    assert 'placeholder="چه ضدآفتابی می‌خواهید؟"' in source
    assert 'label_visibility="collapsed"' in source
    assert "search-label" not in source
    assert "محصولات پیشنهادی" in source
    assert "پیشنهادها" not in source
    assert "افزودن به مقایسه" not in source
    assert "انتخاب‌شده برای مقایسه" not in source
    assert "مقایسهٔ انتخاب‌ها" not in source
    assert "sunscreen_selected" not in source
    assert "SunscreenComparisonService" not in source
    assert "_render_comparison" not in source
    assert "محدودیت شواهد: اطلاعات فقط بر اساس نظرات کاربران" not in source
    assert "footer-limitation" not in source
    assert "footer-price-notice" in source
    assert source.count("PRICE_NOTICE") == 2
    assert demo.PRICE_NOTICE == "قیمت‌های نمایش‌داده‌شده تاریخی‌اند و قیمت فعلی بازار نیستند."
    assert ".st-key-initial-composition" in demo.APP_CSS
    assert "justify-content:center" in demo.APP_CSS
    assert "transform:translateY(-8.5rem)" in demo.APP_CSS
    assert ".st-key-compact-composition" in demo.APP_CSS
    assert '.st-key-hero' in demo.APP_CSS
    assert source.index('with st.container(key="hero"') < source.index("composition_key =")
    assert "sample_column, _sample_spacer = st.columns([5, 2])" in source
    assert 'key="sample-menu", horizontal_alignment="right"' in source
    assert ".st-key-sample-menu" in demo.APP_CSS
    assert "align-items:flex-start !important" in demo.APP_CSS
    assert "[data-testid='stPopoverBody'] [data-testid='stButton'] > button" in demo.APP_CSS
    assert ".st-key-answer-panel" in demo.APP_CSS
    assert "<strong>\\u2067{html.escape(row[\"title\"])}\\u2069</strong>" in source
    assert "st.rerun()" in source
    assert "st.info" not in source
    assert "راهنمای انتخاب مبتنی بر شواهد" not in source
    assert "برای شروع" not in source
    assert "text_alignment=\"right\"" in source
    assert "شواهد هر پیشنهاد از نظرهای کاربران همین محصول نمایش داده شده است." in source
    assert "@media (max-width: 640px)" in demo.APP_CSS
    assert "--navy" in demo.APP_CSS and "--teal" in demo.APP_CSS
    assert "--canvas" in demo.APP_CSS and "max-width:1240px" in demo.APP_CSS
    assert "Streamlit test IDs below are presentation hooks" in source
    assert len(demo.EXAMPLE_QUESTIONS) == 4
    for argument in (
        "min_historical_price=minimum_price or None",
        "max_historical_price=maximum_price or None",
        'brand=None if brand == "همه" else brand',
        "min_review_evidence=minimum_reviews",
    ):
        assert argument in source


def test_comparison_backend_remains_present_but_is_not_a_demo_feature() -> None:
    assert Path("src/digikala_llm/sunscreen_comparison.py").is_file()


def test_card_summary_is_compact_and_keeps_a_bidi_safe_citation() -> None:
    row = _result()["results"][0]
    row["evidence"][0]["excerpt"] = "متن " * 100
    summary = demo.card_summary(row)
    visible = summary.replace("\u2066", "").replace("\u2069", "")
    assert len(summary) < 230
    assert "[نظر 11، ردیف 31]" in visible


def test_visible_answer_formatter_hides_only_supplied_citations_and_prices() -> None:
    title = "کرم ضد آفتاب رنگی فاقد چربی شون با رنگ طبیعی + SPF50"
    context = {
        "products": [
            {
                "user_review_evidence": [
                    {"comment_id": 52694324, "canonical_source_row_number": 1876647}
                ]
            }
        ]
    }
    grounded = (
        f"{title} (قیمت تاریخی استنباط‌شده 852 000 ریال) با نظرات مثبت "
        "[52694324، 1876647] [نظر 52694324، ردیف 1876647] "
        "[COMMENT_ID 52694324، ردیف 1876647] (قیمت 6 998 000 ریال) "
        "(قیمت: 785 100 ریال)، گزینهٔ دیگر [999، 888] است..."
    )
    visible = demo.format_visible_answer(grounded, rows=[{"title": title}], context=context)

    assert grounded.endswith("[999، 888] است...")
    assert f"**\u2067{title}\u2069**" in visible
    assert "[52694324، 1876647]" not in visible
    assert "[نظر 52694324، ردیف 1876647]" not in visible
    assert "[COMMENT_ID 52694324، ردیف 1876647]" not in visible
    assert "[999، 888]" in visible  # Unknown pairs are not removed by a broad regex.
    assert "قیمت تاریخی" not in visible
    assert "قیمت استنباط" not in visible
    assert "852 000 ریال" not in visible
    assert "6 998 000 ریال" not in visible
    assert "785 100 ریال" not in visible
    assert "  " not in visible
    assert "،." not in visible and "؛." not in visible


def test_retrieval_mode_badges_are_deterministic_and_never_mislabel_fallback() -> None:
    assert demo.retrieval_mode_label("hybrid") == "Hybrid RAG: واژگانی + معنایی"
    assert demo.retrieval_mode_label("semantic") == "بازیابی معنایی"
    assert demo.retrieval_mode_label("lexical_fallback") == "بازیابی واژگانی (fallback)"
    assert demo.retrieval_mode_label("lexical") is None
    assert demo.retrieval_mode_label(None) is None


def test_empty_initial_query_and_sample_copy_do_not_submit() -> None:
    state = {"sunscreen_query": ""}
    assert state["sunscreen_query"] == ""
    result = demo.select_example(state, demo.EXAMPLE_QUESTIONS[1])
    assert result is None
    assert state["sunscreen_query"] == demo.EXAMPLE_QUESTIONS[1]


def test_visible_answer_hides_citations_without_changing_the_grounded_text() -> None:
    grounded = "این گزینه نزدیک‌تر است [نظر 11، ردیف 31]؛ ادامهٔ پاسخ."
    context = {
        "products": [
            {"user_review_evidence": [{"comment_id": 11, "canonical_source_row_number": 31}]}
        ]
    }
    visible = demo.format_visible_answer(grounded, rows=[], context=context)
    assert grounded == "این گزینه نزدیک‌تر است [نظر 11، ردیف 31]؛ ادامهٔ پاسخ."
    assert "[نظر" not in visible and "11" not in visible and "31" not in visible
    assert visible == "این گزینه نزدیک‌تر است؛ ادامهٔ پاسخ."


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
    assert "gap:7px" in demo.APP_CSS


def test_cards_keep_the_only_visible_price_and_traceable_citation() -> None:
    row = _result()["results"][0]
    assert demo.format_price(row["historical_price_inferred_irr"]) == "قیمت: \u2066۱۲۵٬۰۰۰\u2069 ریال"
    assert "[نظر 11، ردیف 31]" in demo.card_summary(row).replace("\u2066", "").replace("\u2069", "")
    assert ".product-title, .product-title strong" in demo.APP_CSS
    assert "font-weight:800 !important" in demo.APP_CSS
