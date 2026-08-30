# Final sunscreen MVP evaluation

## Requirement-to-evidence

- **scoped data and prices**: verified — `data/processed/sunscreen_mvp/v1/manifest.json`
- **fixed retrieval correctness and determinism**: verified — `reports/evaluation/sunscreen_mvp_evaluation.json`
- **grounding/citation audit**: verified offline; live status recorded separately — `stored evidence-pair and optional live-audit report`
- **recommendation_status Macro F1**: verified — `reports/evaluation/recommendation_status_evaluation.json`
- **live answer quality, tokens, cost**: recorded when live report exists; human scores remain null — `opt-in sunscreen_live_audit report`

## Scope and artifacts

- Products: `1048`; canonical comments: `53365`; brands: `175`.
- Historical-price coverage: `99.81%` (1046 products).
- Semantic artifact: `53365` vectors, `384` dimensions, `intfloat/multilingual-e5-small`; build runtime `3579.350s`, RSS `2621812` KiB.

## Retrieval and latency

- 10 baseline + 2 semantic fixed cases; deterministic lexical, semantic, and hybrid outcomes are retained in the JSON report.
- Comparison mode observed the three retrieval channels; this establishes functional correctness/repeatability, not ranking superiority.
- Per-channel fixed-case latency: `{'count': 33, 'min': 24.322, 'max': 8766.229, 'mean': 652.4689393939393}`; semantic index load: `0.44631135900272056` seconds.
- No human relevance labels: Recall@K, MRR, nDCG, and statistical Hybrid-superiority claims are unavailable.

## Grounding and live-answer audit

- Stored evidence pairs checked: `813`; provenance failures: `0`.
- Bounded Groq context excludes raw scores/embeddings; historical-price and medical-safety guardrails are present.
- Provider request success: `{'completed_cases': 5, 'attempted_cases': 5}`; complete responses: `{'complete': 5, 'truncated': 0}`; citation membership: `{'citation_membership_passed': True, 'invalid': 0, 'total': 26, 'valid': 26}`.
- Human quality: `not scored: all human scores remain null`; monetary cost: `unavailable: no explicit verified tariff configured`.

## Recommendation-status prediction

- Labels: `['no_idea', 'not_recommended', 'recommended']`; distribution: `{'no_idea': 5107, 'not_recommended': 3730, 'recommended': 35771}`; excluded: `8757`.
- Split: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260830); first valid fold`; train/test `35687/8921`; product overlap `0`.
- Dummy Macro F1: `0.296693`; text Macro F1: `0.631486`; improvement `0.334794`.

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| no_idea | 0.325634 | 0.566112 | 0.413448 | 1021 |
| not_recommended | 0.512171 | 0.705094 | 0.593345 | 746 |
| recommended | 0.962739 | 0.823455 | 0.887667 | 7154 |

Confusion matrix (rows=true, columns=predicted; label order as above):

```text
578 252 191
183 526 37
1014 249 5891
```

## API usage, cost, and human review

- Local embedding compute is not API-token usage.
- Token counts and monetary cost are unavailable without the opt-in live report; no configured authoritative tariff or account-spend evidence exists, so the $5 budget is unverified.
- Human rubric scores remain null until a human completes the separate live audit.

## Failure analysis

- **full-comments SQLite/HDD bottleneck** — symptom: slow/bottlenecked historical attempt; cause: disk-backed full-corpus workflow; mitigation: scoped one-pass builder avoids SQLite; remaining risk: raw corpus rebuild remains expensive.
- **decimal SQLite REAL precision** — symptom: binary-float precision loss; cause: REAL representation; mitigation: exact Decimal/TEXT contract; remaining risk: future consumers must preserve decimal schema.
- **abandoned semantic staging** — symptom: pre-success staging directory; cause: interrupted prior build; mitigation: explicit cleanup and atomic success publication; remaining risk: future interrupted builds need the same review.
- **Gemini 403** — symptom: provider authorization failure; cause: provider permission; mitigation: not used by this MVP; remaining risk: external providers can deny access.
- **Groq model/permission failure** — symptom: unavailable model or permission; cause: provider configuration/access; mitigation: bounded local evidence fallback; remaining risk: live synthesis remains provider-dependent.
- **Hugging Face Xet/CAS download** — symptom: download failure; cause: remote transfer path; mitigation: HTTP retry and cached local model; remaining risk: initial model availability.
- **optional torchvision watcher noise** — symptom: watcher import noise; cause: optional dependency watcher; mitigation: file-watcher mitigation; remaining risk: environment-specific warnings.
- **semantic artifact/model unavailable** — symptom: semantic channel cannot load; cause: missing/corrupt artifact or optional model; mitigation: explicit lexical fallback; remaining risk: semantic recall unavailable during fallback.
- **weakest classifier class** — symptom: lowest text-model F1 for no_idea; cause: class ambiguity/imbalance in subset; mitigation: balanced class weights and Macro F1; remaining risk: no_idea remains weak.
- **unproven Hybrid ranking superiority** — symptom: fixed cases are not relevance judgements; cause: no human ranking labels; mitigation: report functional correctness separately; remaining risk: ranking uplift unknown.

## Limitations

- Sunscreen category only; no generalization claim.
- No raw CSV scan, embedding rebuild, live API call, or dataset mutation in this consolidation.
- No stored full test-count command evidence.
- Live Groq cost/usage requires explicit user audit.
