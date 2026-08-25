# Digikala LLM

نسخه‌ی اول یک سیستم هوشمند جست‌وجو و تحلیل محصول و نظر کاربران، مطابق معماری فاز صفر.

## وضعیت فعلی

فاز ۱ — راه‌اندازی Repository، شناخت Dataset و اولین EDA.

## شروع سریع

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

فایل CSV یا TSV را داخل `data/raw/` قرار دهید و اجرا کنید:

```bash
digikala-eda data/raw/your_dataset.csv
```

گزارش‌های JSON و Markdown در `reports/eda/` ساخته می‌شوند.

## ساختار پروژه

```text
data/raw/          داده‌ی اصلی؛ بدون تغییر و خارج از Git
data/interim/      داده‌ی میانی
data/processed/    داده‌ی پاک‌سازی‌شده و آماده‌ی مدل
reports/eda/       خروجی EDA
src/digikala_llm/  کد پروژه
tests/             تست‌ها
```

## اصول معماری V1

- Exact facts با فیلتر ساختاریافته، نه semantic approximation
- Evidence بازیابی‌شده جدا از aggregate نظر کل کاربران
- پاسخ grounded همراه با شناسه‌ی review و abstention در نبود evidence
- ابتدا baseline و metric؛ سپس پیچیدگی‌هایی مثل hybrid، reranker، LoRA یا agent

