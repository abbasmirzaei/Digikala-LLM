# Digikala LLM

## Sunscreen MVP

Local scope: `مراقبت پوست > کرم ضد آفتاب` — 1,048 products, 53,522 raw matching reviews, 892 reviewed products, 175 brands, and 99.81% historical-price coverage.

Architecture: scoped builder → published Parquet → cached deterministic lexical index → evidence-backed search/comparison → bounded Groq synthesis → Streamlit. Retrieval and candidate selection stay local, CPU-only, and deterministic. Groq is used only after retrieval to turn bounded product/review evidence into Persian prose; it is never given the full dataset or raw CSV. If `GROQ_API_KEY` is absent or the API fails, the demo shows an explicitly labelled non-LLM evidence fallback.

Prices are **historical inferred IRR**, never current price, stock, or availability. Review excerpts are user opinions, not verified product facts. The MVP does not diagnose, treat, or guarantee skin suitability; unsupported attributes are omitted. The Groq key is read only from `GROQ_API_KEY`; do not put a real key in `.env` or commit it. The model defaults to `openai/gpt-oss-20b` and can be changed with `GROQ_MODEL`. The application never calls Groq's models-list endpoint.

```bash
pip install -e ".[dev,demo]"
digikala-build-sunscreen
digikala-search-sunscreen "ضد آفتاب بدون رنگ" --limit 5
digikala-compare-sunscreen 2331113 289763 --query "برای پوست چرب"
digikala-evaluate-sunscreen
streamlit run src/digikala_llm/sunscreen_demo.py
```

For a live grounded-answer smoke test (after exporting `GROQ_API_KEY` in your own terminal and
after the scoped artifacts exist), run:

```bash
GROQ_MODEL=openai/gpt-oss-20b streamlit run src/digikala_llm/sunscreen_demo.py --server.port=8502
```

The builder writes `data/processed/sunscreen_mvp/v1/`; later commands use only published data. The
fixed suite has 12 cases: the original 10 lexical/comparison baseline cases plus 2 documented
semantic Persian paraphrase cases. Evaluation reports are in `reports/evaluation/`. Screenshot
placeholder: capture the local presenter view when needed. Non-goals: all-category platform,
fine-tuning, live prices, accounts, deployment, and medical recommendation.

### Semantic artifact checkpoint

The optional `semantic` extra supports a CPU-only, published-artifact-only embedding build with
`intfloat/multilingual-e5-small`. It embeds canonical comments with `passage: ` prefixes into
normalized float32 vectors and writes an atomic, checksummed semantic artifact with explicit
vector-to-comment provenance. The application validates this artifact before using exact semantic
retrieval and deterministic hybrid fusion; unavailable semantic retrieval falls back to lexical.

```bash
pip install -e ".[semantic]"
python -m digikala_llm.sunscreen_semantic_builder --smoke
```

The evaluation command tests only deterministic retrieval and comparison semantics, so it never
requires Groq, API access, or a key. Groq request tests are mocked and assert the bounded context,
grounding instruction, visible citations, error fallback, and absence of model discovery.

### Recommendation-status baseline

The optional `ml` extra evaluates authoritative canonical `recommendation_status` labels using
text-only title/body features and a deterministic product-disjoint holdout. It reports Macro F1
for a most-frequent dummy and character-TF-IDF logistic-regression baseline; it neither changes
the application nor infers/relabels source targets.

```bash
pip install -e ".[ml]"
digikala-evaluate-recommendation-status
```

### Final evaluation

Create the offline consolidated report from existing published artifacts and evaluation reports:

```bash
digikala-evaluate-final
```

It writes [`final_evaluation.json`](reports/evaluation/final_evaluation.json) and
[`final_evaluation.md`](reports/evaluation/final_evaluation.md). It does not call Groq. To run the
separate, opt-in five-prompt live audit in your own terminal (with a configured key), use:

```bash
digikala-evaluate-sunscreen-live
```
