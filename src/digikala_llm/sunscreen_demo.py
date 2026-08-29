"""Conversational Streamlit presenter for the sunscreen-only MVP."""

from __future__ import annotations

import html
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
# Streamlit test IDs below are presentation hooks; recheck them after a major Streamlit upgrade.
APP_CSS = """
<style>
:root { --navy:#17324d; --teal:#276b73; --muted:#667085; --line:#d9e0e2; --paper:#fffdf9; --canvas:#f7f4ee; }
.stApp { background: var(--canvas); }
html, body, [data-testid='stAppViewContainer'] { direction:rtl; text-align:right; color:var(--navy); }
header[data-testid='stHeader'] { background:transparent; }
[data-testid='stToolbar'] { visibility:hidden; }
.block-container { max-width:1240px; padding:1.15rem 1.5rem 3rem; }
h1 { font-size:clamp(1.8rem, 3vw, 2.25rem); margin:.05rem 0 .25rem; letter-spacing:-.025em; }
h2 { font-size:1.3rem; margin:1.65rem 0 .7rem; }
[data-testid='stCaptionContainer'] { color:var(--muted); }
[data-testid='stAlert'] { background:#f2f3f1; border:1px solid #e0e2de; border-radius:.7rem; color:#485562; padding:.45rem .7rem; }
.stButton > button { border-radius:.6rem; min-height:2.45rem; font-weight:650; }
.stButton > button[kind='primary'] { background:var(--teal); border-color:var(--teal); color:white; padding-inline:1.5rem; }
.stButton > button[kind='primary']:hover { background:#1e5960; border-color:#1e5960; }
[data-testid='stVerticalBlockBorderWrapper'] { background:var(--paper); border-color:var(--line); border-radius:1rem; box-shadow:0 5px 18px rgb(23 50 77 / 6%); padding:.85rem .95rem; }
[data-testid='stVerticalBlockBorderWrapper'] .stMarkdown { line-height:1.9; max-width:82ch; }
[data-testid='stExpander'] { border-color:var(--line); border-radius:.65rem; }
.provider-badge,.rank-badge { display:inline-block; border:1px solid #b9d6d7; border-radius:999px; color:#19565c; background:#edf7f6; font-size:.75rem; font-weight:750; padding:.16rem .52rem; }
.provider-badge--fallback { border-color:#d8dce2; color:#5d6775; background:#f5f6f8; }
.rank-badge--alternative { border-color:#dde1e6; color:#596575; background:#f8f9fa; }
.hero-kicker { color:var(--teal); font-size:.78rem; font-weight:750; letter-spacing:.035em; margin:0; }
.hero-mark { align-items:center; background:#e8f2f1; border-radius:50%; color:var(--teal); display:inline-flex; font-size:1.05rem; height:2.15rem; justify-content:center; margin-left:.55rem; width:2.15rem; }
.hero-subtitle { color:var(--muted); margin:0; }
.answer-heading { color:var(--navy); font-size:1.05rem; font-weight:750; margin:.1rem 0 .55rem; }
.answer-copy { line-height:1.95; max-width:82ch; }
.product-title { color:var(--navy); display:-webkit-box; font-size:1rem; font-weight:750; line-height:1.65; margin:.5rem 0 .35rem; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden; }
.price-metric { color:var(--teal); font-size:.9rem; font-weight:750; line-height:1.65; margin:.55rem 0 .2rem; }
[data-testid='stHorizontalBlock'] > [data-testid='column'] > div > [data-testid='stVerticalBlockBorderWrapper'] { height:100%; }
@media (max-width: 640px) {
  .block-container { padding:1rem .85rem 2rem; }
  h1 { font-size:1.7rem; }
  [data-testid='stHorizontalBlock'] { flex-wrap:wrap; gap:.55rem; }
  [data-testid='stHorizontalBlock'] > [data-testid='column'] { flex:1 1 100%; min-width:100%; }
  [data-testid='stVerticalBlockBorderWrapper'] { padding:.7rem; }
}
</style>
"""


def missing_artifact_message(error: FileNotFoundError) -> str:
    return f"دادهٔ منتشرشده یافت نشد: {error}. ابتدا digikala-build-sunscreen را اجرا کنید."


def format_price(value: object) -> str:
    if value is None:
        return "قیمت تاریخی استنباط‌شده: نامشخص"
    return f"قیمت تاریخی استنباط‌شده: \u2066{int(value):,}\u2069 ریال (نه قیمت فعلی)"


def format_search_result(row: dict[str, Any]) -> str:
    return (
        f"{format_price(row['historical_price_inferred_irr'])} | "
        f"\u2066{row['total_canonical_review_count']:,}\u2069 نظر"
    )


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
    first = {**evidence[0], "excerpt": str(evidence[0]["excerpt"])[:130].rstrip()}
    if len(str(evidence[0]["excerpt"])) > 130:
        first["excerpt"] += "…"
    return evidence_caption(first)


def _render_evidence(st: Any, evidence: list[dict[str, Any]], title: str = "مشاهده شواهد") -> None:
    with st.expander(title, expanded=False):
        if evidence:
            for item in evidence:
                st.caption(evidence_caption(item))
        else:
            st.caption("شواهد کوتاهِ بازیابی‌شده‌ای در دسترس نیست.")


def _display_answer(st: Any, answer: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown('<p class="answer-heading">جمع‌بندی دستیار</p>', unsafe_allow_html=True)
        if answer["source"] == "deterministic_fallback":
            st.markdown('<span class="provider-badge provider-badge--fallback">پاسخ محلی</span>', unsafe_allow_html=True)
            st.caption("مدل زبانی در این پاسخ استفاده نشده است.")
        else:
            st.markdown('<span class="provider-badge">Groq</span>', unsafe_allow_html=True)
        rendered = _CITATION.sub(
            lambda match: f"[نظر \u2066{match.group(1)}\u2069، ردیف \u2066{match.group(2)}\u2069]",
            answer["text"],
        )
        st.markdown(rendered)


def _render_cards(st: Any, result: dict[str, Any], selected: list[int]) -> list[int]:
    rows = result["results"][:MAX_PRIMARY_CARDS]
    for rank, (column, row) in enumerate(zip(st.columns(MAX_PRIMARY_CARDS), rows), start=1):
        with column, st.container(border=True):
            badge = "پیشنهاد اول" if rank == 1 else "گزینهٔ جایگزین"
            badge_class = "rank-badge" if rank == 1 else "rank-badge rank-badge--alternative"
            st.markdown(f'<span class="{badge_class}">{badge}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="product-title" dir="auto">\u2067{html.escape(row["title"])}\u2069</p>',
                unsafe_allow_html=True,
            )
            st.caption(f"برند: {row['brand'] or 'برند نامشخص'}")
            st.markdown(f'<p class="price-metric">{format_search_result(row)}</p>', unsafe_allow_html=True)
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
    st.markdown(APP_CSS, unsafe_allow_html=True)

    @st.cache_resource
    def load(path: str) -> SunscreenLexicalIndex:
        return SunscreenLexicalIndex(Path(path))

    with st.container(border=True):
        st.markdown('<span class="hero-mark" aria-hidden="true">☼</span>', unsafe_allow_html=True)
        st.markdown('<p class="hero-kicker">راهنمای انتخاب مبتنی بر شواهد</p>', unsafe_allow_html=True)
        st.title("دستیار انتخاب ضدآفتاب")
        st.markdown(
            '<p class="hero-subtitle">جست‌وجو در نظرهای کاربران و جمع‌بندی مستند برای انتخاب آگاهانه</p>',
            unsafe_allow_html=True,
        )
    st.info(f"ⓘ {SAFETY_NOTICE}")
    try:
        index = load(str(DEFAULT_DATA_DIR))
    except FileNotFoundError as error:
        st.error(missing_artifact_message(error))
        return

    if "sunscreen_query" not in st.session_state:
        st.session_state.sunscreen_query = EXAMPLE_QUESTIONS[0]
    with st.container(border=True):
        st.caption("نمونه پرسش‌ها")
        for example_columns, examples in (
            (st.columns(2), EXAMPLE_QUESTIONS[:2]),
            (st.columns(2), EXAMPLE_QUESTIONS[2:]),
        ):
            for column, example in zip(example_columns, examples):
                if column.button(example, key=f"example-{example}", use_container_width=True):
                    st.session_state.sunscreen_query = example
        query_column, action_column = st.columns([4, 1])
        query = query_column.text_input(
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
        run_search = action_column.button("پیشنهاد بده", type="primary", use_container_width=True)

    if run_search:
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
    selected_titles = [row["title"] for row in result["results"] if row["product_id"] in selected]
    with st.container(border=True):
        if selected_titles:
            st.caption(f"انتخاب‌شده برای مقایسه: {'، '.join(selected_titles)}")
        else:
            st.caption("برای مقایسه، دو تا چهار محصول را با دکمهٔ هر کارت انتخاب کنید.")
        run_comparison = len(selected) >= 2 and st.button("مقایسهٔ انتخاب‌ها", type="primary")
    if run_comparison:
        comparison = SunscreenComparisonService(index).compare(selected, query=query)
        answer = GroundedAssistant().answer_comparison(query, comparison)
        _render_comparison(st, comparison, answer)


if __name__ == "__main__":
    main()
