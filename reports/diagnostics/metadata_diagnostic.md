# Phase 1 bounded metadata diagnostic

## Scope and reproducibility

This is a read-only metadata diagnostic, not a full EDA or cleaning run. It does not change any
unresolved question to resolved.

| Item | Value |
|---|---|
| Execution timestamp | `2026-08-25T17:27:43.720812+00:00` (UTC) |
| Products source | `data/raw/digikala-products.csv` |
| Comments source | `data/raw/digikala-comments.csv` |
| Products rows scanned for currency sample | 1,283,496, in 100,000-row chunks |
| Comments rows for date diagnostic | Full 6,156,289-row scan, in 100,000-row chunks |
| Comments rows for `seller_code` diagnostic | Full 6,156,289-row scan, in 100,000-row chunks |
| Seller sentinel cross-column scan | `2026-08-25T22:27:34+03:30`; full 6,156,289 rows, three columns, 100,000-row chunks |
| Products row limit for price-history diagnostic | Exactly the first 100,000 rows |
| Python | `3.13.13` |
| pandas | `2.3.3` |

For the currency plausibility sample, complete product rows were fingerprinted in source-column
order with SHA-256 and exact repeats were removed. Rows with missing, non-numeric, fractional, or
non-positive `Price` were excluded from price quantiles and sampling. This left 960,169 distinct
positive-price offer rows. Quantiles use the deterministic nearest-rank definition: sort ascending
and select one-based rank `ceil(p × N)`.

For each of the six observed `sub_category` values, the sampler targeted p10, p25, p50, p75, and
p90. It chose a row at the target price where possible, preferred a nonblank Persian title of a
readable length, and broke ties deterministically by price distance, title properties,
`product_id`, and seller. A `product_id` could appear only once in the 30-row sample.

## 1. Currency plausibility sample

### Positive-price quantiles

All values below are raw source units. Currency is not asserted.

| Scope | N | p10 | p25 | p50 | p75 | p90 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global | 960,169 | 400,000 | 739,000 | 1,700,000 | 4,583,800 | 15,500,000 | 120,600,000 | 599,000,000 |
| clothe | 446,401 | 590,000 | 1,140,000 | 2,723,000 | 7,500,000 | 24,500,000 | 108,750,000 | 420,508,000 |
| book & stationary & art | 274,540 | 320,000 | 530,000 | 973,600 | 2,300,000 | 7,270,000 | 230,000,000 | 895,000,000 |
| beauty | 123,967 | 346,500 | 550,000 | 1,100,000 | 2,500,000 | 6,750,000 | 79,000,000 | 195,000,000 |
| toys and kids | 67,398 | 399,000 | 761,000 | 1,600,000 | 3,900,000 | 10,500,000 | 68,100,000 | 359,000,000 |
| rural goods | 24,117 | 447,000 | 700,000 | 1,600,000 | 4,880,000 | 13,552,000 | 75,600,000 | 355,000,000 |
| travel | 23,746 | 369,000 | 436,100 | 1,680,000 | 4,845,000 | 16,500,000 | 139,000,000 | 380,000,000 |

### Deterministic 30-offer review sample

`provisional_price_toman` is shown only as the requested arithmetic comparison
`price_raw / 10`. It is not a currency conversion claim.

| product_id | title_fa | sub_category | Brand | Seller | price_raw | provisional_price_toman | sample quantile |
|---:|---|---|---|---|---:|---:|---:|
| 1938182 | دستبند کد X17 | clothe | متفرقه | پانته گالری | 590,000 | 59,000 | p10 |
| 7566744 | جاکلیدی مدل K98 | clothe | متفرقه | بامسان | 1,140,000 | 114,000 | p25 |
| 5570240 | دستبند نقره زنانه ترمه ۱ مدل شکیبا | clothe | ترمه ۱ | لیردا گلد | 2,723,000 | 272,300 | p50 |
| 4128205 | کفش مردانه کد 029 | clothe | متفرقه | فروشگاه آذربایجان | 7,500,000 | 750,000 | p75 |
| 12257138 | مانتو زنانه مدل زِرآی | clothe | متفرقه | مادام کالکشن | 24,500,000 | 2,450,000 | p90 |
| 2009556 | پانچ مدل m07295 | book & stationary & art | متفرقه | سیتی نایس | 320,000 | 32,000 | p10 |
| 2141986 | پاکت هدیه کد 5 | book & stationary & art | متفرقه | فردزند | 530,000 | 53,000 | p25 |
| 9276876 | کتاب میکروب های ترسناک اثر نیک آرنولد انتشارات پیدایش | book & stationary & art | انتشارات پیدایش | کتابکالا | 973,600 | 97,360 | p50 |
| 1824879 | پرچم کد PM4 | book & stationary & art | متفرقه | بارمان دکور چاپ | 2,300,000 | 230,000 | p75 |
| 5885707 | پرچم مدل محرم کد c29 | book & stationary & art | متفرقه | بارمان دکور چاپ | 7,270,000 | 727,000 | p90 |
| 3323229 | لاک ناخن کامنت شماره 61 | beauty | کامنت | کامنت | 346,500 | 34,650 | p10 |
| 1714855 | کلاه مش کد 4020 | beauty | متفرقه | ارایشی شهرزاد | 550,000 | 55,000 | p25 |
| 343248 | سختی سنج آب مدل HM | beauty | متفرقه | فروشگاه جوی واتر | 1,100,000 | 110,000 | p50 |
| 1955490 | برس مو مدل EX1 | beauty | متفرقه | ارایشی شهرزاد | 2,500,000 | 250,000 | p75 |
| 9493257 | قوزبند طبی کد SH2020NAB | beauty | متفرقه | تن سلامت | 6,750,000 | 675,000 | p90 |
| 2899317 | ناخن گیر نوا مدل B7 | toys and kids | متفرقه | سام شاه | 399,000 | 39,900 | p10 |
| 657259 | کالسکه اسباب بازی کودک مدل GHASHANGE کد 1318 | toys and kids | متفرقه | اسباب بازی نیما | 761,000 | 76,100 | p25 |
| 8002032 | پودر کلر مدل شارک | toys and kids | متفرقه | استخر درخشان | 1,600,000 | 160,000 | p50 |
| 6001033 | ساختنی مدل قطار | toys and kids | متفرقه | نگار كالا | 3,900,000 | 390,000 | p75 |
| 3372021 | راکر کودک مدل فیل | toys and kids | متفرقه | گروه مهرآذر | 10,500,000 | 1,050,000 | p90 |
| 8446421 | پیاز پرک خشک روحبخش - 100 گرم | rural goods | متفرقه | بومیکده | 447,000 | 44,700 | p10 |
| 10510429 | لپه - 1 کیلوگرم | rural goods | متفرقه | خاتون استور | 700,000 | 70,000 | p25 |
| 10508746 | سویا - 2 کیلوگرم | rural goods | متفرقه | خاتون استور | 1,600,000 | 160,000 | p50 |
| 937334 | سشوار مسافرتی سام مدل HD-1281W | rural goods | سام | فروشگاه رزلند | 4,880,000 | 488,000 | p75 |
| 10707233 | چلغوز نچرال - 500 گرم | rural goods | متفرقه | کالاکو | 13,552,000 | 1,355,200 | p90 |
| 3180268 | کبریت طرح گل کد kbs371 | travel | متفرقه | استودیو هنری ژانو | 369,000 | 36,900 | p10 |
| 6978019 | چشم بند خواب مدل star | travel | متفرقه | نامی | 436,100 | 43,610 | p25 |
| 6523980 | کیف کمری مدل 100 | travel | متفرقه | خانه کیف | 1,680,000 | 168,000 | p50 |
| 9852194 | ساک ورزشی مدل Abr Criech | travel | متفرقه | گالری ورزشی حسام | 4,845,000 | 484,500 | p75 |
| 12075541 | آتشدان مدل تاشو MQ5767 | travel | متفرقه | مگ صنعت | 16,500,000 | 1,650,000 | p90 |

### Interpretation

The divided-by-ten figures produce values that a human reviewer may compare with recognizable
products, but this diagnostic does not establish a currency. Category assignments also contain
surprising products, so category labels are not reliable currency evidence by themselves. The
provisional IRR inference remains pending human validation against these examples or another
approved set. No web/current prices were consulted.

## 2. Full streaming `created_at` diagnostic

This extension ran at `2026-08-25T18:01:20.609759+00:00` (UTC). It read only `id` and `created_at`
from all comments in 100,000-row chunks. Peak analytical state consisted of one chunk, aggregate
counters, valid distinct date keys, and at most 50 invalid examples.

### Coverage and shape validation

| Measure | Count |
|---|---:|
| Total rows scanned | 6,156,289 |
| Missing `created_at` | 0 |
| Blank/whitespace-only `created_at` | 0 |
| Exact `D <recognized Persian Jalali month> YYYY` shape | 6,156,289 |
| Parse-valid recognized Jalali dates | 6,156,289 |
| Invalid or unmatched nonblank values | 0 |

Detected date-shape patterns were assigned exclusively. Only one distinct shape occurred:

| Detected shape | Count |
|---|---:|
| `D <recognized Persian Jalali month> YYYY` with ASCII digits | 6,156,289 |

### Requested anomaly checks

| Check | Count |
|---|---:|
| Unrecognized month names | 0 |
| Invalid day numbers for the stated Jalali month | 0 |
| Non-four-digit years | 0 |
| Time-of-day components | 0 |
| Timezone `Z` or numeric offset suffixes | 0 |
| Rows containing Persian digits (`۰`–`۹`) | 0 |
| Rows containing Arabic-Indic digits (`٠`–`٩`) | 0 |

All date digits were ASCII. Day validation used 31 days for months 1–6, 30 days for months 7–11,
and 29 or 30 days for Esfand according to the calculated Jalali year length.

### Explicit diagnostic conversion

No pandas automatic date inference was used. Recognized month names were mapped explicitly to
Jalali month numbers. Valid dates were converted with the integer-arithmetic Solar Hijri conversion
used by established Jalaali implementations. Before the scan, the implementation passed seven
known Nowruz anchors from 1397 through 1403 and 755 Jalali→Gregorian→Jalali round-trip cases for
years 1300–1450. Every distinct parsed source date also passed a round-trip assertion. Conversion
was diagnostic only; no raw value was rewritten.

| Boundary | Original Jalali representation | Converted Gregorian ISO date |
|---|---|---|
| Minimum comment date | `23 تیر 1395` | `2016-07-13` |
| Maximum comment date | `26 مهر 1402` | `2023-10-18` |

These are the true minimum and maximum **comment dates in this source file**. They must not be used
to infer the dataset snapshot date or the product/offer price capture date.

### Comment counts by Jalali year

| Jalali year | Comments |
|---:|---:|
| 1395 | 3,575 |
| 1396 | 6,357 |
| 1397 | 16,693 |
| 1398 | 138,645 |
| 1399 | 684,321 |
| 1400 | 1,253,484 |
| 1401 | 2,165,894 |
| 1402 | 1,887,320 |
| **Total** | **6,156,289** |

### Invalid or unmatched examples

There were no invalid, unmatched, missing, or blank values, so there are no examples to list. The
diagnostic was configured to retain at most 50 examples if any were encountered.

The scan validates a uniform raw date shape and a safe diagnostic conversion for this comments
file. It does not establish business timezone, comment publication time-of-day, dataset snapshot
date, or product/offer capture date.

## 3. `min_price_last_month` bounded diagnostic

Exactly the first 100,000 product rows were read for this section.

| Value-quality measure | Count |
|---|---:|
| Zero | 92,260 |
| Missing | 0 |
| Negative | 0 |
| Non-numeric | 0 |

Among 7,738 pairs where both `Price` and `min_price_last_month` were valid and positive:

| Relationship | Count |
|---|---:|
| `min_price_last_month < Price` | 6,013 |
| `min_price_last_month = Price` | 902 |
| `min_price_last_month > Price` | 823 |

### Twenty deterministic examples

Examples are the first five source rows in each of four deterministic groups: less than, equal,
greater than, and invalid/unavailable.

| Source row | product_id | title_fa | Price | min_price_last_month | Relationship |
|---:|---:|---|---:|---:|---|
| 58 | 10421284 | مداد ابرو پیپا مدل فالت لِس شماره 111 | 4,589,400 | 4,254,800 | less |
| 61 | 11430094 | سرم تقویت کننده مژه و ابرو اسنس مدل Grow Like A Boss حجم 6 میلی لیتر | 4,700,000 | 3,190,000 | less |
| 65 | 10421350 | مداد ابرو پیپا مدل فالت لِس شماره 112 | 5,078,900 | 4,500,000 | less |
| 71 | 10415181 | ریمل ابرو پیپا مدل دیفاین شماره 182 | 5,524,600 | 5,240,000 | less |
| 78 | 10415143 | ریمل ابرو پیپا مدل دیفاین شماره 181 | 5,682,200 | 5,150,000 | less |
| 56 | 5539371 | محلول تقویت کننده مژه و ابرو اکسیلیا مدل 2022 حجم 3 میلی لیتر | 4,390,000 | 4,390,000 | equal |
| 106 | 2197343 | ریمل ابرو آدور کوین شماره R06 | 930,000 | 930,000 | equal |
| 114 | 2643229 | مداد ابرو رویال اترنیتی شماره 600 | 940,000 | 940,000 | equal |
| 171 | 9663342 | ژل ابرو دفکتو مدل C1 بسته 3 عددی | 1,839,700 | 1,839,700 | equal |
| 427 | 1903487 | مداد ابرو دونادیا شماره 14 | 659,000 | 659,000 | equal |
| 34 | 11542467 | ژل ابرو شگلم مدل Set Me Up به همراه برس ابرو | 3,690,000 | 4,850,000 | greater |
| 57 | 4251106 | ماژیک ابرو پیپا شماره 172 | 4,577,900 | 4,637,500 | greater |
| 129 | 10674028 | هاشور ابرو ورشیپ شماره 30 | 1,610,000 | 1,880,000 | greater |
| 195 | 9265678 | ریمل ابرو نوت شماره 02 | 2,000,000 | 2,350,000 | greater |
| 215 | 10217416 | سرم تقویت کننده ابرو آیسول | 2,177,200 | 2,510,000 | greater |
| 2 | 7096438 | آبسلانگ مدل s5 بسته 250 عددی | 634,800 | 0 | unavailable |
| 3 | 2845119 | آبسلانگ مدل M-1 بسته 400 عددی | 818,800 | 0 | unavailable |
| 4 | 6117745 | آبسلانگ مدل m50 مجموعه 500 عددی | 920,000 | 0 | unavailable |
| 5 | 1912926 | استند ابسلانگ مدل S01 | 1,100,000 | 0 | unavailable |
| 6 | 6335462 | آبسلانگ نوری تسلامد مدل All-in-One | 1,530,000 | 0 | unavailable |

### Interpretation

The field is overwhelmingly zero in this bounded sample (92.26%), while positive values can be
below, equal to, or above `Price`. This is compatible with several collection mechanisms and does
not establish an exact lookback window, capture method, currency, or whether the two price fields
were captured simultaneously. The exact semantics remain unresolved. Treating zero as unavailable
history remains a conservative cleaning policy rather than a discovered temporal definition.

## 4. `seller_code` full streaming diagnostic

This read-only scan ran at `2026-08-25T22:15:45+03:30` and read only `seller_code` from all
6,156,289 comment rows in 100,000-row chunks. An exact temporary SQLite primary-key index was used
for distinct values and was closed and removed after the scan. Here, “missing” means an empty CSV
field, “blank” means a nonempty whitespace-only value, “numeric-only” means ASCII digits only,
“alphanumeric” means ASCII letters/digits with at least one letter, and “other” means any remaining
nonblank pattern.

| Measure | Count |
|---|---:|
| Total rows | 6,156,289 |
| Missing | 0 |
| Blank | 0 |
| Nonblank | 6,156,289 |
| Numeric-only | 0 |
| Alphanumeric | 6,156,289 |
| Other-character pattern | 0 |
| Distinct lexical nonblank values (exact) | 29,214 |
| Leading-zero values | 0 |
| Leading/trailing whitespace | 0 |
| Containing non-ASCII characters | 0 |
| Minimum nonblank length | 3 |
| Maximum nonblank length | 5 |

These are lexical counts only. The follow-up cross-column diagnostic below establishes that the
302,181 lowercase `nan` tokens are a semantic missing/not-applicable sentinel, leaving 5,854,108
real seller-code values and 29,213 distinct real codes. Every real code has length 5; the lexical
minimum of 3 is the sentinel itself.

### Thirty deterministic distinct examples

Examples were selected by ascending SHA-256 digest of the exact UTF-8 source value. All observed
values belong to the single alphanumeric pattern; therefore no numeric-only or other-pattern
examples exist to include.

| # | Pattern | Exact raw `seller_code` |
|---:|---|---|
| 1 | alphanumeric | `AWNAZ` |
| 2 | alphanumeric | `EU3WA` |
| 3 | alphanumeric | `CKYF3` |
| 4 | alphanumeric | `CZKCG` |
| 5 | alphanumeric | `DNVXR` |
| 6 | alphanumeric | `EKAV3` |
| 7 | alphanumeric | `5AEF2` |
| 8 | alphanumeric | `A7JX3` |
| 9 | alphanumeric | `E9TEZ` |
| 10 | alphanumeric | `CMTJ3` |
| 11 | alphanumeric | `EWCN9` |
| 12 | alphanumeric | `CRURW` |
| 13 | alphanumeric | `DJ6A2` |
| 14 | alphanumeric | `C3CWG` |
| 15 | alphanumeric | `EXKEX` |
| 16 | alphanumeric | `EKZRX` |
| 17 | alphanumeric | `CFTVX` |
| 18 | alphanumeric | `AJWG3` |
| 19 | alphanumeric | `C5F5S` |
| 20 | alphanumeric | `CC3CX` |
| 21 | alphanumeric | `5AKEC` |
| 22 | alphanumeric | `D4JMT` |
| 23 | alphanumeric | `DMUVT` |
| 24 | alphanumeric | `DRHRT` |
| 25 | alphanumeric | `CZTVC` |
| 26 | alphanumeric | `C7RA5` |
| 27 | alphanumeric | `EURSG` |
| 28 | alphanumeric | `EUUF9` |
| 29 | alphanumeric | `F2VE6` |
| 30 | alphanumeric | `AYERT` |

The lexical evidence establishes that real `seller_code` values are opaque business codes, not
numeric identifiers. Numeric parsing would discard them. However, nonblankness alone does not prove
that a token is real; the cross-column diagnostic below identifies one exact column-specific
sentinel. CSV readers must still disable broad default NA-token inference so raw tokens remain
available for explicit, case-sensitive field rules.

### Seller sentinel cross-column reconciliation

A second full scan read only `seller_code`, `seller_title`, and `is_buyer` with
`keep_default_na=False`. Token matching was exact and case-sensitive.

| Exact raw token | `seller_code` | `seller_title` |
|---|---:|---:|
| Empty string | 0 | 0 |
| Whitespace-only | 0 | 0 |
| `NAN` | 0 | 0 |
| `NaN` | 0 | 0 |
| `nan` | 302,181 | 302,181 |
| `NA` | 0 | 0 |
| `N/A` | 0 | 0 |
| `NULL` | 0 | 0 |
| `null` | 0 | 0 |
| `None` | 0 | 0 |
| All other values | 5,854,108 | 5,854,108 |

Thus the original hypothesis is confirmed with a casing correction: the raw sentinel is exactly
lowercase `nan`, not uppercase `NAN`.

| `seller_code == "nan"` | `is_buyer=True` | `is_buyer=False` | Total |
|---|---:|---:|---:|
| False | 5,854,108 | 0 | 5,854,108 |
| True | 0 | 302,181 | 302,181 |

The identical cross-tab applies to `seller_title == "nan"`.

| Seller-code status | Seller-title real | Seller-title sentinel | Total |
|---|---:|---:|---:|
| Real | 5,854,108 | 0 | 5,854,108 |
| Sentinel | 0 | 302,181 | 302,181 |

| Seller-code status | `is_buyer=True` | `is_buyer=False` | Total |
|---|---:|---:|---:|
| Real | 5,854,108 | 0 | 5,854,108 |
| Sentinel | 0 | 302,181 | 302,181 |

There are no empty or whitespace-only seller values in this source. The 302,181 lowercase-`nan`
rows are exactly the 302,181 non-buyer rows, both seller columns are sentinel on exactly the same
rows, no buyer row uses the sentinel, and no sentinel code is paired with a real title.

For the 5,854,108 non-sentinel code occurrences:

| Real-code property | Result |
|---|---:|
| Length 5 | 5,854,108 |
| Exact distinct real codes | 29,213 |
| Codes associated with exactly one title | 29,198 |
| Codes associated with two distinct titles | 15 |
| Maximum distinct titles for one code | 2 |

Therefore code-to-title mapping is almost, but not completely, one-to-one. The 15 exceptions must
not be silently reconciled by row-level cleaning. No real seller title is associated with sentinel
code `nan`.

## 5. Diagnostic conclusions

- Human assessment of the 30-product plausibility sample strongly infers that `Price` is IRR/rial,
  but this is not authoritatively confirmed. Raw prices must always be preserved. A derived display
  value may use `price_toman = price_raw_irr / 10` while retaining the raw IRR value.
- No inflation adjustment belongs in the core cleaning pipeline. Any future adjusted feature must
  remain explicitly estimated and must not be represented as a real current market price.
- The full comments scan consistently uses an ASCII-digit, recognized Persian Jalali month-name
  date shape with no invalid days, time component, or timezone suffix. The comment-date range is
  `23 تیر 1395`–`26 مهر 1402` (`2016-07-13`–`2023-10-18` Gregorian).
- Comment dates do not establish the dataset snapshot date or product/offer capture date.
- The bounded history sample supports distinguishing zero as unavailable from invalid current
  price, but it cannot establish exact `min_price_last_month` business semantics.
- No current web prices were used. Jalali conversion was diagnostic only; no source values were
  rewritten.
- Of 6,156,289 lexical nonblank `seller_code` values, exactly 302,181 are the lowercase `nan`
  missing/not-applicable sentinel and 5,854,108 are real opaque business codes. The real codes have
  29,213 distinct values, are all length 5, and must remain strings. Exact lowercase `nan` becomes
  null only through an explicit column-specific rule; other tokens are not interpreted through a
  broad pandas NA vocabulary.
