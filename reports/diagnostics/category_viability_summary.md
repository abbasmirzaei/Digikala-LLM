# Category viability inventory

## Methodology

- Sources: canonical `products_clean.parquet`, `offers_clean.parquet`, and one sequential exact-token pass over `digikala-comments.csv`.
- Authoritative scopes are `Category1` and hierarchical `Category1 > Category2`; `sub_category` is not used for selection.
- Product-level price is the lowest valid positive historical `price_raw` offer for that canonical product. Percentiles are calculated over those product-level prices.
- Comment text is never materialized as a dataset: only aggregate counters and a per-product comment counter are retained.
- The top-pair table applies transparent evidence thresholds (at least 150 products, 1,000 comments, 75 commented products, 20 products with 10+ comments, 800 non-empty bodies, 8 brands, and 60% price coverage) and excludes broad book, stationery, toy, clothing, accessory, jewellery, and child-category families.

## Top 20 coherent `Category1 > Category2` scopes

| Scope | Products | Comments | With comments | ≥5 / ≥10 / ≥20 / ≥50 | Median / p95 | Bodies | Brands | Historical-price coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| آرایش مو > رنگ مو | 11,195 | 83,729 | 6,321 | 2,799 / 1,751 / 956 / 369 | 4 / 57 | 83,720 | 129 | 99.99% |
| شامپو و مراقبت مو > شامپو مو | 3,667 | 114,066 | 2,666 | 1,890 / 1,507 / 1,116 / 647 | 13 / 200 | 114,060 | 379 | 99.89% |
| مراقبت پوست > کرم مرطوب کننده و نرم کننده | 3,386 | 115,153 | 2,628 | 1,806 / 1,454 / 1,103 / 661 | 12.5 / 201 | 115,143 | 357 | 100.00% |
| لوازم کمپینگ > تجهیزات کمپینگ | 11,517 | 69,235 | 2,646 | 1,226 / 923 / 659 / 390 | 4 / 200 | 69,223 | 213 | 99.94% |
| بهداشت و زیبایی ناخن > لاک ناخن | 4,113 | 54,961 | 2,173 | 1,118 / 779 / 515 / 303 | 5 / 190 | 54,956 | 68 | 99.95% |
| مراقبت پوست > پاک کننده آرایش صورت | 1,749 | 61,415 | 1,286 | 871 / 696 / 539 / 354 | 12 / 202 | 61,414 | 284 | 99.89% |
| کفش مردانه > کفش ورزشی مردانه | 5,260 | 39,450 | 2,649 | 999 / 666 / 407 / 192 | 3 / 72 | 39,442 | 120 | 100.00% |
| مراقبت پوست > ماسک صورت و بدن | 1,675 | 55,318 | 1,259 | 840 / 655 / 501 / 326 | 10 / 201 | 55,312 | 125 | 100.00% |
| برس‌ها و تجهیزات آرایشی > برس ها و تجهیزات آرایشی صورت | 9,355 | 57,067 | 1,585 | 810 / 646 / 505 / 329 | 5 / 200 | 57,065 | 101 | 99.99% |
| کفش زنانه و مردانه > مراقبت کفش و لوازم جانبی | 1,775 | 36,077 | 1,261 | 812 / 601 / 417 / 201 | 9 / 164 | 36,073 | 54 | 100.00% |
| مراقبت پوست > کرم ضد آفتاب | 1,048 | 53,522 | 892 | 685 / 584 / 459 / 316 | 21 / 202 | 53,517 | 175 | 99.81% |
| شامپو و مراقبت مو > ماسک و مراقبت مو | 1,848 | 41,121 | 1,186 | 761 / 570 / 415 / 234 | 9 / 200 | 41,119 | 268 | 99.89% |
| بهداشت و زیبایی ناخن > آرایش ناخن | 1,945 | 34,494 | 1,323 | 766 / 549 / 375 / 193 | 6 / 148 | 34,491 | 9 | 100.00% |
| بهداشت دهان و دندان > مسواک | 1,096 | 35,166 | 844 | 613 / 486 / 342 / 209 | 13 / 200 | 35,163 | 53 | 100.00% |
| بهداشت جنسی > کاندوم | 1,100 | 44,203 | 813 | 588 / 482 / 390 / 254 | 17 / 201 | 44,193 | 23 | 100.00% |
| سلامت محیط > فیلتر تصفیه کننده آب | 1,579 | 27,168 | 1,174 | 672 / 461 / 317 / 149 | 6 / 123 | 27,165 | 59 | 100.00% |
| لوازم اصلاح مو > تیغ و یدک اصلاح | 922 | 37,736 | 628 | 489 / 419 / 338 / 227 | 24 / 200 | 37,730 | 33 | 99.89% |
| کفش مردانه > کفش روزمره مردانه | 3,894 | 18,298 | 1,834 | 654 / 418 / 211 / 81 | 3 / 42 | 18,298 | 131 | 100.00% |
| بهداشت و مراقبت بدن > لوسیون و روغن بدن | 1,076 | 32,881 | 772 | 508 / 416 / 309 / 190 | 12 / 200 | 32,879 | 159 | 100.00% |
| کوه‌ نوردی و صخره نوردی > چراغ قوه | 1,070 | 28,070 | 822 | 532 / 412 / 283 / 157 | 10 / 200 | 28,066 | 74 | 99.63% |

## Category1 summary

| Scope | Products | Comments | With comments | ≥5 / ≥10 / ≥20 / ≥50 | Median / p95 | Bodies | Brands | Historical-price coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| اسباب بازی | 42,645 | 414,786 | 19,885 | 9,523 / 6,580 / 4,230 / 2,185 | 4 / 130 | 414,724 | 437 | 99.97% |
| مراقبت پوست | 11,125 | 392,257 | 8,518 | 5,870 / 4,711 / 3,587 / 2,255 | 12.5 / 201 | 392,226 | 660 | 99.96% |
| لباس مردانه | 55,970 | 275,273 | 19,656 | 7,438 / 4,672 / 2,773 / 1,330 | 3 / 70 | 275,242 | 489 | 100.00% |
| لباس زنانه | 77,928 | 246,104 | 24,931 | 7,601 / 4,299 / 2,426 / 1,067 | 2 / 42 | 246,079 | 685 | 99.99% |
| اکسسوری زنانه و مردانه | 52,644 | 238,226 | 16,845 | 6,229 / 3,901 / 2,327 / 1,151 | 3 / 71 | 238,195 | 543 | 100.00% |
| کتاب شعر و ادبیات | 20,343 | 172,174 | 11,841 | 5,637 / 3,402 / 1,892 / 781 | 4 / 65 | 172,158 | 1,190 | 99.99% |
| کتاب کودک و نوجوان | 14,495 | 134,530 | 9,999 | 5,126 / 3,128 / 1,656 / 538 | 5 / 53 | 134,520 | 501 | 99.96% |
| نوشت افزار | 10,386 | 226,662 | 6,517 | 4,026 / 3,035 / 2,160 / 1,277 | 8 / 200 | 226,644 | 144 | 99.99% |
| شامپو و مراقبت مو | 7,448 | 212,710 | 5,150 | 3,551 / 2,779 / 2,056 / 1,207 | 12 / 200 | 212,699 | 558 | 99.91% |
| برس‌ها و تجهیزات آرایشی | 15,872 | 213,644 | 5,278 | 3,363 / 2,637 / 1,988 / 1,214 | 9 / 200 | 213,631 | 248 | 99.97% |
| آرایش مو | 13,675 | 140,293 | 7,805 | 3,854 / 2,587 / 1,559 / 691 | 4 / 90 | 140,281 | 238 | 99.99% |
| دفتر و کاغذ و مقوا | 30,065 | 152,332 | 8,328 | 3,463 / 2,331 / 1,477 / 801 | 3 / 115 | 152,319 | 116 | 99.97% |
| کتاب فلسفه و روانشناسی | 8,921 | 127,039 | 6,283 | 3,433 / 2,222 / 1,315 / 640 | 5 / 110 | 127,027 | 787 | 99.93% |
| اکسسوری مردانه | 32,337 | 119,984 | 9,420 | 3,467 / 2,120 / 1,261 / 575 | 3 / 62.0 | 119,962 | 456 | 100.00% |
| بهداشت و زیبایی ناخن | 12,157 | 142,339 | 5,303 | 2,797 / 2,017 / 1,396 / 804 | 5 / 190 | 142,328 | 196 | 99.98% |
| اکسسوری زنانه | 57,514 | 124,474 | 13,348 | 3,585 / 1,993 / 1,134 / 517 | 2 / 36 | 124,454 | 600 | 99.99% |
| کفش مردانه | 14,633 | 109,805 | 7,583 | 3,039 / 1,987 / 1,189 / 540 | 3 / 70 | 109,789 | 259 | 99.99% |
| دخترانه | 35,260 | 92,266 | 9,569 | 2,920 / 1,673 / 896 / 391 | 2 / 38 | 92,255 | 377 | 100.00% |
| لوازم اداری | 7,129 | 127,806 | 3,463 | 2,163 / 1,655 / 1,236 / 733 | 9 / 200 | 127,794 | 118 | 100.00% |
| بهداشت و مراقبت بدن | 3,503 | 112,971 | 2,468 | 1,717 / 1,398 / 1,030 / 644 | 14 / 201 | 112,963 | 337 | 100.00% |

## Selected comparison scopes

| Scope | Products | Comments | With comments | ≥5 / ≥10 / ≥20 / ≥50 | Median / p95 | Bodies | Brands | Historical-price coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| مراقبت پوست > کرم ضد آفتاب | 1,048 | 53,522 | 892 | 685 / 584 / 459 / 316 | 21 / 202 | 53,517 | 175 | 99.81% |
| شامپو و مراقبت مو > شامپو مو | 3,667 | 114,066 | 2,666 | 1,890 / 1,507 / 1,116 / 647 | 13 / 200 | 114,060 | 379 | 99.89% |
| مراقبت پوست > پاک کننده آرایش صورت | 1,749 | 61,415 | 1,286 | 871 / 696 / 539 / 354 | 12 / 202 | 61,414 | 284 | 99.89% |
| مراقبت پوست > ماسک صورت و بدن | 1,675 | 55,318 | 1,259 | 840 / 655 / 501 / 326 | 10 / 201 | 55,312 | 125 | 100.00% |
| مراقبت پوست > کرم مرطوب کننده و نرم کننده | 3,386 | 115,153 | 2,628 | 1,806 / 1,454 / 1,103 / 661 | 12.5 / 201 | 115,143 | 357 | 100.00% |

## Why sunscreen was selected

`مراقبت پوست > کرم ضد آفتاب` has 1,048 products, 53,522 comments across 892 products, 175 brands, and 99.81% historical-price coverage. Its review-depth counts are 685 / 584 / 459 / 316 at the 5 / 10 / 20 / 50 thresholds.

Sunscreen was selected as the best deadline-constrained MVP scope because it combines sufficient scale, strong review coverage, brand and price diversity, narrow semantic coherence, and a clear comparison-oriented use case. It is not claimed to have the highest raw product or comment count.

## Rejected device scopes

- Laptop: no strictly categorized laptop products; title mentions were false or ambiguous.
- Mobile phone: no strict hardware category; the only mobile-labelled Category1 was training content, and title matches were non-phone items.
- Tablet: `Category1 = تبلت` has 81 products and zero matching comments.

## Limitations

- Categories are source labels and may contain occasional semantic surprises; no category correction was applied.
- Product prices are historical inferred IRR, not current, latest, or live prices. They must not be presented as current market prices.
- Raw comment rows are counted here; this report does not deduplicate or canonicalize comments.

## Run metadata

- Canonical product mapping: 948,352 products
- Runtime: 198.55 seconds
- Maximum RSS: 1,011,040 KiB
- Reconciliation: passed
