"""Bounded, grounded LLM synthesis for the sunscreen-only MVP.

This module never performs retrieval itself.  It receives only deterministic, already
bounded retrieval or comparison output and either asks Groq to summarize it in Persian or
returns an explicitly labelled local evidence fallback.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

from digikala_llm.sunscreen_retrieval import MAX_EXCERPT_CHARS, MAX_EXCERPTS

GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL_ENV = "GROQ_MODEL"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
MAX_PRODUCTS_IN_PROMPT = 3
MAX_COMPARISON_PRODUCTS_IN_PROMPT = 4
MAX_COMPLETION_TOKENS = 1_000
RETRY_COMPLETION_TOKENS = 1_200
MAX_APPENDED_CITATIONS = 3
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "token_limit"})
_BRACKETED_CITATION = re.compile(
    r"\[\s*(?:comment_id|نظر)\s*[:#]?\s*(\d+)\s*[,،]\s*"
    r"(?:source_row|ردیف)\s*[:#]?\s*(\d+)\s*\]",
    flags=re.IGNORECASE,
)
_BARE_COMMENT_ID = re.compile(r"\bcomment_id\s*[:#]?\s*(\d+)\b", flags=re.IGNORECASE)
_PERSIAN_COMMENT_REFERENCE = re.compile(r"نظر\s*(?:شماره)?\s*(\d+)")
_CURRENT_PRICE_LANGUAGE = re.compile(
    r"(?:"
    r"قیمت\s*(?:فعلی|روز|امروز|بازار)|بازار\s*(?:فعلی|امروز)|"
    r"قیمت\s*(?:کنونی|حال\s*حاضر)|افزایش\s*قیمت|کاهش\s*قیمت|"
    r"گران(?:\s*تر)?\s*(?:شده|است)|ارزان(?:\s*تر)?\s*(?:شده|است)|"
    r"current\s*(?:market\s*)?price|current\s*market|today['’]?s\s*price|"
    r"today\s*price|price\s*(?:increase|decrease)|(?:price\s*)?(?:increased|decreased)|"
    r"(?:is\s*)?(?:cheaper|more\s*expensive)\s+today"
    r")",
    flags=re.IGNORECASE,
)
_SAFE_CURRENT_PRICE_DISCLAIMER = re.compile(
    r"(?:"
    r"نه\s+قیمت\s*(?:فعلی|روز|امروز|بازار)|"
    r"قیمت\s*(?:فعلی|روز|امروز|بازار)(?:\s+بازار)?\s*(?:نیست|نداریم|موجود\s+نیست|"
    r"در\s+دسترس\s+نیست)|"
    r"(?:داده|اطلاعات).{0,40}?قیمت\s*(?:فعلی|روز|امروز|بازار).{0,30}?(?:نیست|نداریم|"
    r"موجود\s+نیست|در\s+دسترس\s+نیست)|"
    r"نمی\s*توان.{0,40}?قیمت\s*(?:فعلی|روز|امروز|بازار).{0,40}?(?:تعیین|محاسبه|"
    r"استنتاج|نتیجه\s*گیری)|"
    r"(?:not|isn't)\s+(?:the\s+)?current\s*(?:market\s*)?price|"
    r"no\s+(?:current|today['’]?s)\s*price\s*(?:data|is\s+available)|"
    r"cannot\s+determine.{0,40}?(?:current|today['’]?s)\s*price"
    r")",
    flags=re.IGNORECASE,
)

SYSTEM_INSTRUCTION = """تو دستیار انتخاب ضدآفتاب هستی و فقط به فارسی پاسخ می‌دهی.
فقط از کاتالوگ و شواهدی که در پیام کاربر آمده استفاده کن؛ محصول، قیمت، SPF، سازگاری با نوع پوست،
بافت یا ادعای پزشکی جدید نساز. فقط مقدارهای عرضه‌شدهٔ «قیمت تاریخی استنباط‌شده به ریال» را ذکر کن؛
قیمت، بازار یا تغییرات قیمت در زمان حال را نه نتیجه‌گیری کن، نه نام ببر و نه مقایسه کن. هر ادعای
برگرفته از نظر را به کاربران نسبت بده و با قالب دقیق
[نظر COMMENT_ID، ردیف SOURCE_ROW] ارجاع بده و هرگز برچسب انگلیسی COMMENT_ID ننویس. فقط نامزدهای
داده‌شده را مقایسه کن. ادعای بافت یا مناسب‌بودن را فقط به کاربران نسبت بده و تشخیص، سازگاری پزشکی
یا قطعیت اعلام نکن. اگر شواهد برای پاسخ کافی نیست، صادقانه کمبود شواهد را بگو. پاسخ ۲۵۰ تا ۴۵۰ کلمه
یا کوتاه‌تر اگر شواهد کم است، با جمله‌های کامل، خوانا و بدون JSON باشد."""


def llm_model() -> str:
    """Return only the model identifier; API keys are intentionally never returned."""
    return os.environ.get(GROQ_MODEL_ENV, DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def _citation(evidence: dict[str, Any]) -> str:
    return f"[نظر {evidence['comment_id']}، ردیف {evidence['canonical_source_row_number']}]"


def _bounded_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "comment_id": item["comment_id"],
            "canonical_source_row_number": item["canonical_source_row_number"],
            "excerpt": str(item.get("excerpt") or "")[:MAX_EXCERPT_CHARS],
        }
        for item in items[:MAX_EXCERPTS]
    ]


def retrieval_context(query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    """Make the only candidate/evidence payload the LLM is allowed to see."""
    products = []
    for row in retrieval.get("results", [])[:MAX_PRODUCTS_IN_PROMPT]:
        products.append(
            {
                "product_id": row["product_id"],
                "title": row["title"],
                "brand": row.get("brand"),
                "historical_price_inferred_irr": row.get("historical_price_inferred_irr"),
                "historical_price_label": row["historical_price_label"],
                "canonical_review_count": row["total_canonical_review_count"],
                "user_review_evidence": _bounded_evidence(row.get("evidence", [])),
            }
        )
    return {"kind": "recommendation", "query": query, "products": products}


def comparison_context(query: str | None, comparison: dict[str, Any]) -> dict[str, Any]:
    """Reduce deterministic comparison data to measurable fields and cited evidence."""
    products = []
    for row in comparison.get("products", [])[:MAX_COMPARISON_PRODUCTS_IN_PROMPT]:
        products.append(
            {
                "product_id": row["product_id"],
                "title": row["title"],
                "brand": row.get("brand"),
                "historical_price_inferred_irr": row.get("historical_price_inferred_irr"),
                "historical_price_label": row["historical_price_label"],
                "canonical_review_count": row["canonical_review_count"],
                "buyer_review_count": row["buyer_review_count"],
                "buyer_review_percentage": row["buyer_review_percentage"],
                "valid_average_rating": row["valid_average_rating"],
                "valid_rating_count": row["valid_rating_count"],
                "positive_recommendation_share": row["positive_recommendation_share"],
                "price_difference_vs_compared_inferred_irr": row[
                    "price_difference_vs_compared_inferred_irr"
                ],
                "positive_user_review_evidence": _bounded_evidence(
                    row.get("positive_evidence", [])
                ),
                "critical_user_review_evidence": _bounded_evidence(
                    row.get("critical_evidence", [])
                ),
            }
        )
    return {"kind": "comparison", "query": query, "products": products}


def _price(value: Any) -> str:
    return "نامشخص" if value is None else f"{int(value):,} ریال (تاریخی، نه قیمت فعلی)"


def _citations_by_comment_id(context: dict[str, Any]) -> dict[str, str]:
    """Map only supplied evidence IDs to their required display citation."""
    citations: dict[str, str] = {}
    for product in context["products"]:
        evidence_groups = (
            (product.get("user_review_evidence", []),)
            if context["kind"] == "recommendation"
            else (
                product.get("positive_user_review_evidence", []),
                product.get("critical_user_review_evidence", []),
            )
        )
        for evidence_group in evidence_groups:
            for evidence in evidence_group:
                citation = _citation(evidence)
                citations.setdefault(str(evidence["comment_id"]), citation)
    return citations


def normalize_citations(text: str, context: dict[str, Any]) -> str:
    """Normalize safe model citation variants without inventing IDs or source rows."""
    citations = _citations_by_comment_id(context)

    def bracketed(match: re.Match[str]) -> str:
        return f"[نظر {match.group(1)}، ردیف {match.group(2)}]"

    text = _BRACKETED_CITATION.sub(bracketed, text)
    return _BARE_COMMENT_ID.sub(
        lambda match: citations.get(match.group(1), match.group(0)), text
    )


def citation_appendix(text: str, context: dict[str, Any]) -> str:
    """Add at most a few supplied citations that prose references but does not display."""
    citations = _citations_by_comment_id(context)
    referenced_ids = set(_PERSIAN_COMMENT_REFERENCE.findall(text)) | set(
        _BARE_COMMENT_ID.findall(text)
    )
    missing = [
        citation
        for comment_id, citation in citations.items()
        if comment_id in referenced_ids and citation not in text
    ][:MAX_APPENDED_CITATIONS]
    return "شواهد قابل بررسی: " + "، ".join(missing) if missing else ""


def has_unsupported_current_price_claim(text: str) -> bool:
    """Reject present-market assertions, while allowing explicit absence disclaimers."""
    for sentence in re.split(r"[.!؟\n]+", text.replace("\u200c", " ")):
        if _CURRENT_PRICE_LANGUAGE.search(sentence) and not _SAFE_CURRENT_PRICE_DISCLAIMER.search(
            sentence
        ):
            return True
    return False


def _fallback_reason(error: Exception) -> str:
    """Classify expected provider failures without retaining provider error text."""
    status_code = getattr(error, "status_code", None)
    error_hints = f"{type(error).__name__} {error}".casefold()
    if isinstance(error, ImportError):
        return "missing_sdk"
    if isinstance(error, TimeoutError) or "timeout" in error_hints:
        return "timeout"
    if status_code == 429 or "rate" in error_hints or "quota" in error_hints:
        return "quota_or_rate_limit"
    if status_code in {401, 403} or isinstance(error, PermissionError):
        return "permission_or_api_error"
    if status_code == 404 or "model" in error_hints:
        return "unavailable_model"
    return "permission_or_api_error"


def deterministic_fallback(context: dict[str, Any], reason: str) -> str:
    """Polished non-LLM response based only on the same bounded context."""
    products = context["products"]
    if not products:
        return "شواهد بازیابی‌شده‌ای برای این پرسش پیدا نشد؛ بنابراین پیشنهاد قابل اتکایی ندارم."
    label = "پاسخ محلی مبتنی بر شواهد (بدون مدل زبانی):"
    if context["kind"] == "comparison":
        lines = [label, "مقایسهٔ زیر فقط بر پایهٔ شاخص‌های اندازه‌پذیر و نظرهای کاربران است."]
        for product in products:
            lines.append(
                f"{product['title']} ({product.get('brand') or 'برند نامشخص'}) — "
                f"{_price(product['historical_price_inferred_irr'])}؛ "
                f"{product['canonical_review_count']:,} نظر ثبت‌شده."
            )
            evidence = (
                product["positive_user_review_evidence"] + product["critical_user_review_evidence"]
            )
            if evidence:
                lines.append(f"نظر کاربر: «{evidence[0]['excerpt']}» {_citation(evidence[0])}")
        return "\n\n".join(lines)
    lines = [label, "کاندیداها با بازیابی قطعی انتخاب شده‌اند؛ قیمت‌ها تاریخی‌اند، نه قیمت فعلی."]
    for product in products:
        line = (
            f"{product['title']} ({product.get('brand') or 'برند نامشخص'}) — "
            f"{_price(product['historical_price_inferred_irr'])}؛ "
            f"{product['canonical_review_count']:,} نظر ثبت‌شده."
        )
        if product["user_review_evidence"]:
            evidence = product["user_review_evidence"][0]
            line += f" نظر کاربر: «{evidence['excerpt']}» {_citation(evidence)}"
        lines.append(line)
    return "\n\n".join(lines)


class GroundedAssistant:
    """Small Groq adapter with safe deterministic degradation and no model discovery."""

    def __init__(self, client_factory: Callable[[], Any] | None = None) -> None:
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client() -> Any:
        from groq import Groq

        # The official SDK reads GROQ_API_KEY itself. Keeping the key out of call
        # arguments also prevents it from entering client-factory mocks or logs.
        return Groq()

    def answer_recommendation(self, query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
        return self._answer(retrieval_context(query, retrieval))

    def answer_comparison(self, query: str | None, comparison: dict[str, Any]) -> dict[str, Any]:
        return self._answer(comparison_context(query, comparison))

    @staticmethod
    def _completion(client: Any, context: dict[str, Any], max_tokens: int) -> tuple[str, str | None]:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return str(choice.message.content or "").strip(), getattr(choice, "finish_reason", None)

    @staticmethod
    def _fallback(context: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "source": "deterministic_fallback",
            "reason": reason,
            "text": deterministic_fallback(context, reason),
            "context": context,
        }

    def _answer(self, context: dict[str, Any]) -> dict[str, Any]:
        if not os.environ.get(GROQ_API_KEY_ENV):
            return self._fallback(context, "missing_api_key")
        try:
            client = self._client_factory()
            text, finish_reason = self._completion(client, context, MAX_COMPLETION_TOKENS)
            if finish_reason and finish_reason.casefold() in _TRUNCATED_FINISH_REASONS:
                text, finish_reason = self._completion(client, context, RETRY_COMPLETION_TOKENS)
            if not text or (finish_reason and finish_reason.casefold() in _TRUNCATED_FINISH_REASONS):
                return self._fallback(context, "truncated_response")
            text = normalize_citations(text, context)
            if has_unsupported_current_price_claim(text):
                return self._fallback(context, "unsupported_current_price_claim")
            appendix = citation_appendix(text, context)
            if appendix:
                text = f"{text}\n\n{appendix}"
            return {
                "source": "groq",
                "reason": "api_success",
                "model": llm_model(),
                "text": text,
                "context": context,
            }
        except Exception as error:  # noqa: BLE001 -- SDK exposes several transport/provider exception types.
            reason = _fallback_reason(error)
            return self._fallback(context, reason)
