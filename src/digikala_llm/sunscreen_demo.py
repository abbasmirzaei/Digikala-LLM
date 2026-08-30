"""Conversational Streamlit presenter for the sunscreen-only MVP."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from digikala_llm.sunscreen_hybrid import DEFAULT_SEMANTIC_DIR, HybridSunscreenRetriever
from digikala_llm.sunscreen_llm import GroundedAssistant
from digikala_llm.sunscreen_retrieval import DEFAULT_DATA_DIR, SunscreenLexicalIndex

EXAMPLE_QUESTIONS = (
    "ضد آفتاب سبک و بدون چربی پیشنهاد بده",
    "ضد آفتاب بدون رنگ با نظر خریداران زیاد",
    "گزینه‌های با قیمت تاریخی مناسب را نشان بده",
    "برای پوست چرب چه شواهدی در نظر کاربران هست؟",
)
PRICE_NOTICE = "قیمت‌های نمایش‌داده‌شده تاریخی‌اند و قیمت فعلی بازار نیستند."
MAX_PRIMARY_CARDS = 3
_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_VISIBLE_ANSWER_CITATION = re.compile(
    r"\[[\s\u2066\u2069]*(?:(?:نظر|comment[_\s]*id)[\s:#\u2066\u2069]*)?"
    r"(?P<comment>[0-9۰-۹٠-٩]+)[\s\u2066\u2069]*[,،][\s\u2066\u2069]*"
    r"(?:(?:ردیف|source[_\s]*row)[\s:#\u2066\u2069]*)?"
    r"(?P<row>[0-9۰-۹٠-٩]+)[\s\u2066\u2069]*\]",
    flags=re.IGNORECASE,
)
_VISIBLE_PRICE_CLAUSE = re.compile(
    r"\(\s*قیمت(?:\s+(?:تاریخی|استنباط(?:‌|\s)*شده|داده(?:‌|\s)*ای))*\s*:?\s*"
    r"[0-9۰-۹٠-٩][0-9۰-۹٠-٩\s,٬،.\u00a0\u202f]*\s*ریال\s*\)",
)
_NORMALIZE_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_MARKDOWN_TITLE_ESCAPE = re.compile(r"([\\`*{}\[\]()#\-.!_>])")
# Streamlit test IDs below are presentation hooks; recheck them after a major Streamlit upgrade.
APP_CSS = """
<style>
:root { --navy:#17324d; --teal:#276b73; --muted:#667085; --line:#d9e0e2; --paper:#fffdf9; --canvas:#f7f4ee; }
.stApp { background: var(--canvas); }
html, body, [data-testid='stAppViewContainer'] { direction:rtl; text-align:right; color:var(--navy); }
header[data-testid='stHeader'] { background:transparent; }
[data-testid='stToolbar'] { visibility:hidden; }
.block-container { max-width:1240px; padding:1rem 1.5rem 5.25rem; }
h1 { font-size:clamp(1.65rem, 2.7vw, 2.05rem); margin:0; letter-spacing:-.025em; }
h2 { font-size:1.2rem; margin:1rem 0 .45rem; text-align:right; }
[data-testid='stCaptionContainer'] { color:var(--muted); }
.stButton > button { border-radius:.55rem; min-height:2.2rem; font-weight:650; }
.stButton > button[kind='primary'] { background:var(--teal); border-color:var(--teal); color:white; padding-inline:1.5rem; }
.stButton > button[kind='primary']:hover { background:#1e5960; border-color:#1e5960; }
[data-testid='stTextInput'] input { background:#fff; border-radius:.75rem; direction:rtl; font-size:1.1rem; min-height:3.45rem; padding-inline:1.05rem; text-align:right; }
[data-testid='stButton'] > button { transition:background-color .15s ease, border-color .15s ease, transform .15s ease; }
[data-testid='stButton'] > button:not([kind='primary']) { background:#fff; border-color:#d8e0e3; color:#29445b; }
[data-testid='stButton'] > button:not([kind='primary']):hover { background:#f0f7f6; border-color:#9dc5c4; color:#19565c; }
[data-testid='stVerticalBlockBorderWrapper'] { background:var(--paper); border-color:var(--line); border-radius:.85rem; box-shadow:0 3px 12px rgb(23 50 77 / 5%); padding:.72rem .85rem; }
[data-testid='stVerticalBlockBorderWrapper'] .stMarkdown { direction:rtl; line-height:1.85; max-width:82ch; text-align:right; unicode-bidi:plaintext; }
[data-testid='stExpander'] { border-color:var(--line); border-radius:.55rem; margin-top:.35rem; }
[data-testid='stExpander'] summary { align-items:center; font-size:.87rem; height:2.3rem; padding-block:.1rem; }
.provider-badge,.rank-badge { display:inline-block; border:1px solid #b9d6d7; border-radius:999px; color:#19565c; background:#edf7f6; font-size:.75rem; font-weight:750; padding:.16rem .52rem; }
.provider-badge--fallback { border-color:#d8dce2; color:#5d6775; background:#f5f6f8; }
.answer-badges { display:flex; flex-wrap:wrap; gap:7px; margin:.1rem 0 .45rem; }
.rank-badge--alternative { border-color:#dde1e6; color:#596575; background:#f8f9fa; }
.st-key-hero { padding-top:.35rem; }
.hero-subtitle { color:var(--muted); margin:.15rem 0 .95rem; text-align:center; }
.st-key-initial-composition { box-sizing:border-box; display:flex; flex-direction:column; justify-content:center; min-height:calc(100vh - 9rem); transform:translateY(-8.5rem); }
.st-key-compact-composition { padding-top:.25rem; }
.st-key-sample-menu { align-items:flex-start !important; direction:rtl; text-align:right; }
.st-key-sample-menu .stButton > button { direction:rtl; text-align:right; }
[data-testid='stPopoverBody'] [data-testid='stButton'] > button { direction:rtl; justify-content:flex-start; text-align:right; }
.answer-heading { color:var(--navy); font-size:1.02rem; font-weight:750; margin:0 0 .35rem; text-align:right; }
.st-key-answer-panel, .st-key-answer-panel [data-testid='stMarkdownContainer'] { direction:rtl; text-align:right; unicode-bidi:plaintext; }
.product-title, .product-title strong { color:var(--navy); font-weight:800 !important; }
.product-title { display:-webkit-box; font-size:1rem; line-height:1.65; margin:.5rem 0 .35rem; -webkit-box-orient:vertical; -webkit-line-clamp:2; overflow:hidden; }
.price-metric { color:var(--teal); font-size:.9rem; font-weight:750; line-height:1.65; margin:.55rem 0 .2rem; }
.footer-price-notice { bottom:.6rem; color:var(--muted); font-size:.78rem; left:0; line-height:1.65; margin:0; padding:0 .85rem; pointer-events:none; position:fixed; right:0; text-align:center; z-index:10; }
[data-testid='stHorizontalBlock'] > [data-testid='column'] > div > [data-testid='stVerticalBlockBorderWrapper'] { height:100%; }
@media (max-width: 640px) {
  .block-container { padding:1rem .85rem 2rem; }
  h1 { font-size:1.7rem; }
  [data-testid='stHorizontalBlock'] { flex-wrap:wrap; gap:.55rem; }
  [data-testid='stHorizontalBlock'] > [data-testid='column'] { flex:1 1 100%; min-width:100%; }
  [data-testid='stVerticalBlockBorderWrapper'] { padding:.7rem; }
  .st-key-initial-composition { min-height:calc(100vh - 7rem); transform:translateY(-2.5rem); }
}
</style>
"""


def missing_artifact_message(error: FileNotFoundError) -> str:
    return f"دادهٔ منتشرشده یافت نشد: {error}. ابتدا digikala-build-sunscreen را اجرا کنید."


def format_price(value: object) -> str:
    if value is None:
        return "قیمت: نامشخص"
    amount = f"{int(value):,}".translate(_PERSIAN_DIGITS).replace(",", "٬")
    return f"قیمت: \u2066{amount}\u2069 ریال"


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


def select_example(session_state: Any, example: str) -> None:
    """Copy an example to the input; retrieval remains an explicit user action."""
    session_state["sunscreen_query"] = example


def _normalise_trace_number(value: object) -> str | None:
    digits = str(value).translate(_NORMALIZE_DIGITS).replace("\u2066", "").replace("\u2069", "")
    if not digits.isdecimal():
        return None
    return str(int(digits))


def _supplied_evidence_pairs(context: object) -> frozenset[tuple[str, str]]:
    """Return only citation pairs included in the bounded grounded context."""
    if not isinstance(context, dict):
        return frozenset()
    pairs: set[tuple[str, str]] = set()
    for product in context.get("products", []):
        if not isinstance(product, dict):
            continue
        for evidence in product.get("user_review_evidence", []):
            if not isinstance(evidence, dict):
                continue
            comment_id = _normalise_trace_number(evidence.get("comment_id"))
            source_row = _normalise_trace_number(evidence.get("canonical_source_row_number"))
            if comment_id is not None and source_row is not None:
                pairs.add((comment_id, source_row))
    return frozenset(pairs)


def _markdown_product_title(title: str) -> str:
    escaped = _MARKDOWN_TITLE_ESCAPE.sub(r"\\\1", html.escape(title))
    return f"**\u2067{escaped}\u2069**"


def format_visible_answer(text: str, *, rows: list[dict[str, Any]], context: object) -> str:
    """Format the visible narrative without mutating the grounded result itself."""
    supplied_pairs = _supplied_evidence_pairs(context)

    def hide_supplied_citation(match: re.Match[str]) -> str:
        pair = (
            _normalise_trace_number(match.group("comment")),
            _normalise_trace_number(match.group("row")),
        )
        return "" if pair in supplied_pairs else match.group(0)

    visible = _VISIBLE_ANSWER_CITATION.sub(hide_supplied_citation, text)
    visible = _VISIBLE_PRICE_CLAUSE.sub("", visible)
    visible = re.sub(r"(?:^|\n)\s*شواهد قابل بررسی\s*:?\s*", "\n", visible)
    visible = re.sub(r"[ \t\u00a0\u202f]+", " ", visible)
    visible = re.sub(r"\s+([،؛.!؟])", r"\1", visible)
    visible = re.sub(r"([،؛])(?=\S)", r"\1 ", visible)
    visible = re.sub(r"(?:،\s*){2,}", "، ", visible).strip()
    titles = sorted({str(row["title"]) for row in rows if row.get("title")}, key=len, reverse=True)
    if titles:
        title_pattern = re.compile("|".join(re.escape(title) for title in titles))
        visible = title_pattern.sub(lambda match: _markdown_product_title(match.group(0)), visible)
    return visible


def _render_evidence(st: Any, evidence: list[dict[str, Any]], title: str = "مشاهده شواهد") -> None:
    with st.expander(title, expanded=False):
        if evidence:
            for item in evidence:
                st.caption(evidence_caption(item))
        else:
            st.caption("شواهد کوتاهِ بازیابی‌شده‌ای در دسترس نیست.")


def retrieval_mode_label(retrieval_mode: str | None) -> str | None:
    """Return a user-facing badge only for a retrieval mode actually returned by the retriever."""
    return {
        "hybrid": "Hybrid RAG: واژگانی + معنایی",
        "semantic": "بازیابی معنایی",
        "lexical_fallback": "بازیابی واژگانی (fallback)",
    }.get(retrieval_mode)


def _display_answer(
    st: Any,
    answer: dict[str, Any],
    retrieval_mode: str | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    with st.container(key="answer-panel", border=True):
        st.markdown('<p class="answer-heading">پاسخ دستیار</p>', unsafe_allow_html=True)
        provider = (
            '<span class="provider-badge provider-badge--fallback">پاسخ محلی</span>'
            if answer["source"] == "deterministic_fallback"
            else '<span class="provider-badge">Groq</span>'
        )
        retrieval = retrieval_mode_label(retrieval_mode)
        if retrieval is not None:
            st.markdown(
                f'<div class="answer-badges">{provider}'
                f'<span class="provider-badge">{retrieval}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(provider, unsafe_allow_html=True)
        if answer["source"] == "deterministic_fallback":
            st.caption("مدل زبانی در این پاسخ استفاده نشده است.")
        st.markdown(
            format_visible_answer(answer["text"], rows=rows or [], context=answer.get("context"))
        )


def _render_cards(st: Any, result: dict[str, Any]) -> None:
    rows = result["results"][:MAX_PRIMARY_CARDS]
    for rank, (column, row) in enumerate(zip(st.columns(MAX_PRIMARY_CARDS), rows), start=1):
        with column, st.container(border=True):
            badge = "پیشنهاد اول" if rank == 1 else "گزینهٔ جایگزین"
            badge_class = "rank-badge" if rank == 1 else "rank-badge rank-badge--alternative"
            st.markdown(f'<span class="{badge_class}">{badge}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="product-title" dir="auto"><strong>\u2067{html.escape(row["title"])}\u2069</strong></p>',
                unsafe_allow_html=True,
            )
            st.caption(f"برند: {row['brand'] or 'برند نامشخص'}")
            st.markdown(f'<p class="price-metric">{format_search_result(row)}</p>', unsafe_allow_html=True)
            st.caption(card_summary(row))
            _render_evidence(st, row["evidence"])


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="دستیار انتخاب ضدآفتاب", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    @st.cache_resource
    def load(path: str) -> HybridSunscreenRetriever:
        return HybridSunscreenRetriever(SunscreenLexicalIndex(Path(path)), DEFAULT_SEMANTIC_DIR)

    try:
        index = load(str(DEFAULT_DATA_DIR))
    except FileNotFoundError as error:
        st.error(missing_artifact_message(error))
        return

    if "sunscreen_query" not in st.session_state:
        st.session_state.sunscreen_query = ""

    with st.container(key="hero", horizontal_alignment="center"):
        st.title("دستیار انتخاب ضدآفتاب", text_alignment="center")
        st.markdown(
            '<p class="hero-subtitle">جست‌وجو در نظرهای کاربران و جمع‌بندی مستند برای انتخاب آگاهانه</p>',
            unsafe_allow_html=True,
        )

    result = st.session_state.get("sunscreen_result")
    composition_key = "compact-composition" if result else "initial-composition"
    with st.container(key=composition_key):
        _search_left, search_content, _search_right = st.columns([2, 6, 2])
        with search_content:
            query_column, action_column = st.columns([5, 2], vertical_alignment="bottom")
            query = query_column.text_input(
                "سؤال درباره ضدآفتاب",
                key="sunscreen_query",
                placeholder="چه ضدآفتابی می‌خواهید؟",
                label_visibility="collapsed",
            )
            with action_column:
                run_search = st.button("پیشنهاد بده", type="primary", width="stretch")

            sample_column, _sample_spacer = st.columns([5, 2])
            with sample_column, st.container(key="sample-menu", horizontal_alignment="right"), st.popover(
                "نمونه",
                type="tertiary",
                width=110,
                wrap=False,
            ):
                for example in EXAMPLE_QUESTIONS:
                    st.button(
                        example,
                        key=f"example-{example}",
                        on_click=select_example,
                        args=(st.session_state, example),
                        width="stretch",
                    )

    minimum_price = 0
    maximum_price = 0
    brand = "همه"
    minimum_reviews = 1

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
        st.rerun()

    if result:
        answer = st.session_state.sunscreen_answer
        _display_answer(st, answer, result.get("retrieval_mode"), result["results"])
        if not result["results"]:
            st.warning("برای این پرسش و تنظیمات، محصول منطبقی پیدا نشد.")
        else:
            st.header("محصولات پیشنهادی", text_alignment="right")
            st.caption("شواهد هر پیشنهاد از نظرهای کاربران همین محصول نمایش داده شده است.")
            _render_cards(st, result)
    st.markdown(f'<p class="footer-price-notice">{PRICE_NOTICE}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
