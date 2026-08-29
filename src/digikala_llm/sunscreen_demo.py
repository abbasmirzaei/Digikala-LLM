"""Conversational Streamlit presenter for the sunscreen-only MVP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from digikala_llm.sunscreen_comparison import SunscreenComparisonService
from digikala_llm.sunscreen_llm import GroundedAssistant
from digikala_llm.sunscreen_retrieval import DEFAULT_DATA_DIR, SunscreenLexicalIndex

SAFETY_NOTICE = (
    "قیمت‌ها تاریخی و استنباط‌شده به ریال‌اند، نه قیمت فعلی. نظرها دیدگاه کاربران‌اند، نه واقعیت "
    "تأییدشده. این ابزار تشخیص پزشکی یا تضمین سازگاری پوست ارائه نمی‌کند."
)
EXAMPLE_QUESTIONS = (
    "ضد آفتاب سبک و بدون چربی پیشنهاد بده",
    "ضد آفتاب بدون رنگ با نظر خریداران زیاد",
    "گزینه‌های با قیمت تاریخی مناسب را نشان بده",
    "برای پوست چرب چه شواهدی در نظر کاربران هست؟",
)
MAX_PRIMARY_CARDS = 3
_CITATION = re.compile(r"\[نظر (\d+)، ردیف (\d+)\]")


def missing_artifact_message(error: FileNotFoundError) -> str:
    return f"دادهٔ منتشرشده یافت نشد: {error}. ابتدا digikala-build-sunscreen را اجرا کنید."


def format_price(value: object) -> str:
    if value is None:
        return "قیمت تاریخی استنباط‌شده: نامشخص"
    return f"قیمت تاریخی استنباط‌شده: \u2066{int(value):,}\u2069 ریال (نه قیمت فعلی)"


def format_search_result(row: dict[str, Any]) -> str:
    return f"{format_price(row['historical_price_inferred_irr'])} | {row['total_canonical_review_count']:,} نظر"


def evidence_caption(evidence: dict[str, Any]) -> str:
    citation = f"[نظر \u2066{evidence['comment_id']}\u2069، ردیف \u2066{evidence['canonical_source_row_number']}\u2069]"
    return (
        f"«{evidence['excerpt']}» — نظر کاربر "
        f"{citation}"
    )


def card_summary(row: dict[str, Any]) -> str:
    evidence = row.get("evidence", [])
    if not evidence:
        return "برای این محصول، شواهد کوتاهِ منطبق با پرسش بازیابی نشد."
    return evidence_caption(evidence[0])


def _render_evidence(st: Any, evidence: list[dict[str, Any]], title: str = "مشاهده شواهد") -> None:
    with st.expander(title, expanded=False):
        if evidence:
            for item in evidence:
                st.caption(evidence_caption(item))
        else:
            st.caption("شواهد کوتاهِ بازیابی‌شده‌ای در دسترس نیست.")


def _display_answer(st: Any, answer: dict[str, Any]) -> None:
    if answer["source"] == "deterministic_fallback":
        st.info("پاسخ زیر محلی و مبتنی بر شواهد است؛ توسط مدل زبانی تولید نشده است.")
    else:
        st.caption(f"پاسخ زبانی مبتنی بر Groq ({answer['model']}) و فقط بر پایهٔ شواهد نمایش‌داده‌شده")
    st.markdown(
        _CITATION.sub(
            lambda match: f"[نظر \u2066{match.group(1)}\u2069، ردیف \u2066{match.group(2)}\u2069]",
            answer["text"],
        )
    )


def _render_cards(st: Any, result: dict[str, Any], selected: list[int]) -> list[int]:
    for row in result["results"][:MAX_PRIMARY_CARDS]:
        with st.container(border=True):
            st.markdown(f"**{row['title']}**")
            st.caption(f"برند: {row['brand'] or 'برند نامشخص'}")
            st.caption(format_search_result(row))
            st.caption(card_summary(row))
            _render_evidence(st, row["evidence"])
            if st.button("افزودن به مقایسه", key=f"add-{row['product_id']}") and (
                row["product_id"] not in selected and len(selected) < 4
            ):
                selected.append(row["product_id"])
    return selected


def _comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "محصول": item["title"],
            "برند": item["brand"] or "نامشخص",
            "قیمت تاریخی": format_price(item["historical_price_inferred_irr"]),
            "تعداد نظر": item["canonical_review_count"],
            "میانگین امتیاز معتبر": item["valid_average_rating"],
            "سهم توصیهٔ مثبت": item["positive_recommendation_share"],
        }
        for item in comparison["products"]
    ]


def _render_comparison(st: Any, comparison: dict[str, Any], answer: dict[str, Any]) -> None:
    st.divider()
    st.header("مقایسهٔ انتخاب‌شده")
    _display_answer(st, answer)
    columns = st.columns(len(comparison["products"]))
    for column, item in zip(columns, comparison["products"]):
        with column:
            st.markdown(f"**{item['title']}**")
            st.caption(format_price(item["historical_price_inferred_irr"]))
            st.caption(f"تعداد نظر: \u2066{item['canonical_review_count']:,}\u2069")
            st.caption(f"میانگین امتیاز معتبر: \u2066{item['valid_average_rating']}\u2069")
    for item in comparison["products"]:
        with st.expander(f"شواهد {item['title']}", expanded=False):
            st.caption("شواهد زیر نظر کاربران‌اند، نه واقعیت تأییدشدهٔ محصول.")
            for evidence in item["positive_evidence"] + item["critical_evidence"]:
                st.caption(evidence_caption(evidence))
    st.caption("برندهٔ کلی اعلام نمی‌شود؛ نشانگرها فقط برای معیارهای اندازه‌پذیر هستند.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="دستیار انتخاب ضدآفتاب", layout="wide")
    st.markdown(
        """<style>
        html,body,[data-testid='stAppViewContainer']{direction:rtl;text-align:right}
        .stButton>button[kind='primary']{background:#386a78;border-color:#386a78}
        [data-testid='stVerticalBlockBorderWrapper']{padding:.55rem .75rem}
        </style>""",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load(path: str) -> SunscreenLexicalIndex:
        return SunscreenLexicalIndex(Path(path))

    st.title("دستیار انتخاب ضدآفتاب")
    st.caption("با شواهد بازیابی‌شده از نظر کاربران، برای محصولات مراقبت پوست > کرم ضد آفتاب")
    st.info(SAFETY_NOTICE)
    try:
        index = load(str(DEFAULT_DATA_DIR))
    except FileNotFoundError as error:
        st.error(missing_artifact_message(error))
        return

    if "sunscreen_query" not in st.session_state:
        st.session_state.sunscreen_query = EXAMPLE_QUESTIONS[0]
    st.write("نمونه پرسش‌ها")
    example_columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(example_columns, EXAMPLE_QUESTIONS):
        if column.button(example, key=f"example-{example}"):
            st.session_state.sunscreen_query = example
    query = st.text_input(
        "چه ضدآفتابی می‌خواهید؟",
        key="sunscreen_query",
        placeholder="مثلاً: ضد آفتاب سبک و بدون رنگ",
    )

    brands = sorted({product["brand"] for product in index.products.values() if product["brand"]})
    with st.expander("تنظیمات پیشرفته", expanded=False):
        first, second, third, fourth = st.columns(4)
        minimum_price = first.number_input("حداقل قیمت تاریخی (ریال)", min_value=0, value=0)
        maximum_price = second.number_input(
            "حداکثر قیمت تاریخی (ریال؛ صفر یعنی بدون سقف)", min_value=0, value=0
        )
        brand = third.selectbox("برند", ["همه"] + brands)
        minimum_reviews = fourth.slider("حداقل تعداد نظر", 0, 50, 1)

    if st.button("پیشنهاد بده", type="primary", use_container_width=True):
        result = index.search(
            query,
            min_historical_price=minimum_price or None,
            max_historical_price=maximum_price or None,
            brand=None if brand == "همه" else brand,
            min_review_evidence=minimum_reviews,
            limit=MAX_PRIMARY_CARDS,
        )
        st.session_state.sunscreen_result = result
        st.session_state.sunscreen_answer = GroundedAssistant().answer_recommendation(query, result)
        st.session_state.sunscreen_selected = []

    result = st.session_state.get("sunscreen_result")
    if not result:
        return
    answer = st.session_state.sunscreen_answer
    _display_answer(st, answer)
    if not result["results"]:
        st.warning("برای این پرسش و تنظیمات، محصول منطبقی پیدا نشد.")
        return
    st.header("پیشنهادها")
    selected = _render_cards(st, result, st.session_state.sunscreen_selected)
    st.session_state.sunscreen_selected = selected
    st.caption("برای مقایسه، دو تا چهار محصول را با دکمهٔ هر کارت انتخاب کنید.")
    if len(selected) >= 2 and st.button("مقایسهٔ انتخاب‌ها", use_container_width=True):
        comparison = SunscreenComparisonService(index).compare(selected, query=query)
        answer = GroundedAssistant().answer_comparison(query, comparison)
        _render_comparison(st, comparison, answer)


if __name__ == "__main__":
    main()
