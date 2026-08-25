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

برای اجرای تحلیل مشترک و بررسی اتصال محصولات به نظرها، هر دو فایل را بدهید. داده‌ها به‌صورت
قطعه‌ای خوانده می‌شوند و اندازهٔ پیش‌فرض هر قطعه ۱۰۰٬۰۰۰ ردیف است:

```bash
digikala-eda data/raw/digikala-products.csv data/raw/digikala-comments.csv --chunksize 100000
```

گزارش‌های JSON و Markdown در `reports/eda/` ساخته می‌شوند.

## نوت‌بوک آموزشی EDA

نوت‌بوک فارسی [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb)
برای بررسی تعاملی نمونه‌های حداکثر ۱۰۰٬۰۰۰ ردیفی آماده شده است و تحلیل کامل فایل‌های بزرگ را
اجرا نمی‌کند.

برای اجرای آن در VS Code:

1. پوشهٔ پروژه را در VS Code باز کنید و افزونه‌های Python و Jupyter را نصب کنید.
2. محیط را با `pip install -e ".[dev]"` آماده کنید.
3. فایل نوت‌بوک را باز کنید و از بالای صفحه روی **Select Kernel** بزنید.
4. گزینهٔ **Python Environments** و سپس مفسر `.venv/bin/python` پروژه را انتخاب کنید.
5. سلول‌ها را به‌ترتیب اجرا کنید. مسیرها و سقف نمونه در ابتدای نوت‌بوک قابل تنظیم‌اند.

## ساختار پروژه

```text
data/raw/          داده‌ی اصلی؛ بدون تغییر و خارج از Git
data/interim/      داده‌ی میانی
data/processed/    داده‌ی پاک‌سازی‌شده و آماده‌ی مدل
reports/eda/       خروجی EDA
notebooks/         تحلیل‌های آموزشی و تعاملی
src/digikala_llm/  کد پروژه
tests/             تست‌ها
```

## اصول معماری V1

- Exact facts با فیلتر ساختاریافته، نه semantic approximation
- Evidence بازیابی‌شده جدا از aggregate نظر کل کاربران
- پاسخ grounded همراه با شناسه‌ی review و abstention در نبود evidence
- ابتدا baseline و metric؛ سپس پیچیدگی‌هایی مثل hybrid، reranker، LoRA یا agent
