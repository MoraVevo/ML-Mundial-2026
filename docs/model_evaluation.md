# Model evaluation

This report evaluates the current neutral parsimonious recipe with temporal guards designed to avoid training on future information.

Feature recipe: `parsimonious_live_fifa_drift_abs_match_script_bilateral_clinical_low_block_w12_guardrail_residual_friendly_weight_060_club_star_finisher_fd_fallback`

## Test policies

### World Cup 2026 holdout

Played World Cup 2026 matches are forced to test. Training uses only national-team matches before the first World Cup 2026 match; same-date and later rows are excluded.

- Train matches: 787
- Test matches: 54
- Test window: 2026-06-11 to 2026-06-25

### External random temporal holdout

Random holdout of 104 non-friendly, non-World-Cup-2026 national matches from the recent official-match pool, seed=42. Training uses only matches before the earliest selected test date; same-date and later non-selected rows are excluded.

- Train matches: 565
- Test matches: 104
- Test window: 2024-09-07 to 2026-03-26

## Metrics

| Evaluation | Accuracy | Correct | Log loss | MAE team A | MAE team B | MAE avg |
|---|---:|---:|---:|---:|---:|---:|
| World Cup 2026 holdout | 0.5741 | 31/54 | 0.9453 | 1.1564 | 0.8650 | 1.0107 |
| External random temporal holdout | 0.6635 | 69/104 | 0.8949 | 0.9929 | 1.0552 | 1.0240 |

![Metrics summary](assets/model_evaluation/metrics_summary.png)

## Technical interpretation

The model is directionally useful on winners, but the current decision threshold is conservative around draws. In these holdouts it assigns draw probability for log-loss calibration, yet the top class rarely becomes `draw`.

| Evaluation | Actual draws | Predicted draws as top class |
|---|---:|---:|
| World Cup 2026 holdout | 14 | 0 |
| External random temporal holdout | 22 | 0 |

This is why log loss is shown next to accuracy: accuracy alone hides whether the model is placing useful probability mass on draws and close matches. The MAE values are reported separately because the goal regressors can be directionally acceptable even when the 1X2 classifier chooses the wrong class.

## Confusion matrices

### World Cup 2026 holdout

![Confusion matrix](assets/model_evaluation/confusion_worldcup_2026.png)

### External random temporal holdout

![Confusion matrix](assets/model_evaluation/confusion_external_random_temporal.png)

## Feature importance

### World Cup 2026 holdout

![Feature importance](assets/model_evaluation/feature_importance_worldcup_2026.png)

| Feature | Importance |
|---|---:|
| `draw_pressure_index` | 2567.67 |
| `rating_guardrail_edge` | 2207.00 |
| `quality_form_edge` | 2166.00 |
| `score_control_value_edge` | 2162.33 |
| `clinical_low_block_matchup_edge` | 2079.00 |
| `rating_drift_abs` | 2047.67 |
| `match_script_compatibility_edge` | 1998.67 |
| `rating_threat_edge` | 1894.00 |

### External random temporal holdout

![Feature importance](assets/model_evaluation/feature_importance_external_random_temporal.png)

| Feature | Importance |
|---|---:|
| `draw_pressure_index` | 2014.67 |
| `rating_guardrail_edge` | 1664.00 |
| `rating_drift_abs` | 1568.67 |
| `match_script_compatibility_edge` | 1532.67 |
| `quality_form_edge` | 1521.67 |
| `score_control_value_edge` | 1494.33 |
| `rating_threat_edge` | 1446.33 |
| `clinical_low_block_matchup_edge` | 1423.00 |

## Feature construction summary

The active model uses only pre-match features. The 12 production features are:

- `competition_family`
- `stage_or_round`
- `rating_threat_edge`
- `quality_form_edge`
- `goal_balance_edge`
- `draw_pressure_index`
- `score_control_value_edge`
- `rating_guardrail_edge`
- `rating_drift_abs`
- `match_script_compatibility_edge`
- `clinical_low_block_matchup_edge`
- `club_star_finisher_edge`

High-level groups:

- Rating strength: FIFA/ranking, Elo-style team strength, rating guardrail and drift.
- Recent form: opponent-adjusted recent points and goal-balance signals.
- Match context: competition family, stage/round and draw-pressure context.
- Tactical/attacking profile: match-script compatibility, clinical low-block matchup and club star-finisher signal.

Excluded from model features: target goals/results, raw identifiers, raw dates, source names and post-match statistics from the evaluated match.

## Leakage controls

The World Cup holdout is the primary model accuracy because it matches the target domain. The external random temporal holdout is a robustness diagnostic: it samples non-World-Cup-2026 official national-team matches but still trains only on matches before the first selected test date.

Both evaluations rebuild features before fitting and keep the following columns out of the model: `match_id`, `source`, raw `date`, teams, goals, final result and provider identifiers. Date is used only to define chronological splits and pre-match rolling context.
