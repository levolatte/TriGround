# Quant grounding learning-signal diagnosis

GPU hours: 6.85694641889694
Answer mode: short

## Rollout groups

- Unique queries: 227
- Groups: 236 (236 complete, 0 incomplete; denominator includes all groups)
- Outcomes: allhit 187, allwrong 9, mixed 40
- Same reward: 182; differing reward: 54 (7.8752 per GPU hour)
- Invalid answers: 1; truncated answers: 0

## Answer diversity

- Status: available (raw parsed boxes available)
- Groups with raw samples: 59; with parsed boxes: 59; varying boxes: 59

## D failure categories

- Status: not_applicable
- unfinished_thinking: 0, token_limit_after_thinking: 0, invalid_final_json: 0, valid_bbox_wrong_target: 0, unknown_raw_text: 0, unknown_target_result: 0

## Existing KL metrics

- Status: available; count: 59
- Mean/min/max: 0.00673343/0/0.0247573; first/last: 0/0.0116416

## by_source

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| city | 93 | 93 | 0 | 61 | 7 | 25 | 64 | 29 | 0 | 0 |
| rgbdt | 89 | 89 | 0 | 86 | 0 | 3 | 76 | 13 | 0 | 0 |
| robo | 54 | 54 | 0 | 40 | 2 | 12 | 42 | 12 | 1 | 0 |

## by_negative_kind

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extent | 57 | 57 | 0 | 56 | 0 | 1 | 46 | 11 | 0 | 0 |
| extent+wrong_object | 5 | 5 | 0 | 4 | 0 | 1 | 4 | 1 | 0 | 0 |
| none | 110 | 110 | 0 | 71 | 9 | 30 | 76 | 34 | 1 | 0 |
| wrong_object | 64 | 64 | 0 | 56 | 0 | 8 | 56 | 8 | 0 | 0 |

## by_sampling_pool

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extent | 52 | 52 | 0 | 51 | 0 | 1 | 44 | 8 | 0 | 0 |
| regular_city | 93 | 93 | 0 | 61 | 7 | 25 | 64 | 29 | 0 | 0 |
| regular_rgbdt | 15 | 15 | 0 | 14 | 0 | 1 | 11 | 4 | 0 | 0 |
| regular_robo | 12 | 12 | 0 | 5 | 2 | 5 | 7 | 5 | 1 | 0 |
| wrong_object | 64 | 64 | 0 | 56 | 0 | 8 | 56 | 8 | 0 | 0 |
