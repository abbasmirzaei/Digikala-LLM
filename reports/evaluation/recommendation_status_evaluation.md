# Recommendation-status baseline evaluation

Published sunscreen canonical comments only.

## Counts

- class_distribution: `{'no_idea': 5107, 'not_recommended': 3730, 'recommended': 35771}`
- excluded_rows: `8757`
- total_canonical_rows: `53365`
- usable_labelled_rows: `44608`

## Split

- algorithm: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260830); first valid fold`
- seed: `20260830`
- product overlap: `0`

## Metrics

### dummy_most_frequent

- Macro F1: `0.296693`
- Accuracy: `0.801928`
- Configuration: `{'strategy': 'most_frequent'}`

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| no_idea | 0.000000 | 0.000000 | 0.000000 | 1021 |
| not_recommended | 0.000000 | 0.000000 | 0.000000 | 746 |
| recommended | 0.801928 | 1.000000 | 0.890078 | 7154 |

Confusion matrix (rows=true, columns=predicted; ['no_idea', 'not_recommended', 'recommended']):

```text
0 0 1021
0 0 746
0 0 7154
```

### text_tfidf_logistic_regression

- Macro F1: `0.631486`
- Accuracy: `0.784105`
- Configuration: `{'feature_fields': ['title', 'body'], 'logistic_regression': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 20260830, 'solver': 'lbfgs'}, 'vectorizer': {'analyzer': 'char_wb', 'min_df': 2, 'ngram_range': [3, 5], 'sublinear_tf': True}}`

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| no_idea | 0.325634 | 0.566112 | 0.413448 | 1021 |
| not_recommended | 0.512171 | 0.705094 | 0.593345 | 746 |
| recommended | 0.962739 | 0.823455 | 0.887667 | 7154 |

Confusion matrix (rows=true, columns=predicted; ['no_idea', 'not_recommended', 'recommended']):

```text
578 252 191
183 526 37
1014 249 5891
```


Improvement over dummy Macro F1: `0.334794`

## Failure analysis

- Lowest text-model F1 label: `no_idea`
- Inspect the labelled confusion matrix; it is true rows by predicted columns.

## Limitations

- Single sunscreen-only holdout; no claim beyond this subset.
- No hyperparameter tuning was performed on the holdout.
- Text-only prediction does not establish product quality or medical suitability.
