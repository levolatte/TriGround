# Quant grounding learning-signal diagnosis

GPU hours: 6.76181558218992
Answer mode: short

## Rollout groups

- Unique queries: 227
- Groups: 236 (236 complete, 0 incomplete; denominator includes all groups)
- Outcomes: allhit 183, allwrong 9, mixed 44
- Same reward: 187; differing reward: 49 (7.2466 per GPU hour)
- Invalid answers: 1; truncated answers: 0

## Answer diversity

- Status: available (raw parsed boxes available)
- Groups with raw samples: 236; with parsed boxes: 236; varying boxes: 236

## D failure categories

- Status: not_applicable
- unfinished_thinking: 0, token_limit_after_thinking: 0, invalid_final_json: 0, valid_bbox_wrong_target: 0, unknown_raw_text: 0, unknown_target_result: 0

## Existing KL metrics

- Status: available; count: 59
- Mean/min/max: 0.00684272/0/0.030274; first/last: 0/0.0171371

## by_source

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| city | 93 | 93 | 0 | 60 | 8 | 25 | 63 | 30 | 0 | 0 |
| rgbdt | 89 | 89 | 0 | 83 | 0 | 6 | 83 | 6 | 0 | 0 |
| robo | 54 | 54 | 0 | 40 | 1 | 13 | 41 | 13 | 1 | 0 |

## by_negative_kind

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extent | 57 | 57 | 0 | 55 | 0 | 2 | 55 | 2 | 0 | 0 |
| extent+wrong_object | 5 | 5 | 0 | 4 | 0 | 1 | 4 | 1 | 0 | 0 |
| none | 110 | 110 | 0 | 69 | 9 | 32 | 73 | 37 | 1 | 0 |
| wrong_object | 64 | 64 | 0 | 55 | 0 | 9 | 55 | 9 | 0 | 0 |

## by_sampling_pool

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extent | 52 | 52 | 0 | 50 | 0 | 2 | 50 | 2 | 0 | 0 |
| regular_city | 93 | 93 | 0 | 60 | 8 | 25 | 63 | 30 | 0 | 0 |
| regular_rgbdt | 15 | 15 | 0 | 13 | 0 | 2 | 13 | 2 | 0 | 0 |
| regular_robo | 12 | 12 | 0 | 5 | 1 | 6 | 6 | 6 | 1 | 0 |
| wrong_object | 64 | 64 | 0 | 55 | 0 | 9 | 55 | 9 | 0 | 0 |
