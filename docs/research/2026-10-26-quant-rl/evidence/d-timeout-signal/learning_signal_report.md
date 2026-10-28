# Quant grounding learning-signal diagnosis

GPU hours: 3.0000015637661432
Answer mode: thinking

## Rollout groups

- Unique queries: 32
- Groups: 32 (32 complete, 0 incomplete; denominator includes all groups)
- Outcomes: allhit 0, allwrong 32, mixed 0
- Same reward: 32; differing reward: 0 (0.0000 per GPU hour)
- Invalid answers: 256; truncated answers: 230

## Answer diversity

- Status: unavailable (no_matched_parsed_bbox)
- Groups with raw samples: 32; with parsed boxes: 0; varying boxes: None

## D failure categories

- Status: available
- unfinished_thinking: 230, token_limit_after_thinking: 0, invalid_final_json: 26, valid_bbox_wrong_target: 0, unknown_raw_text: 0, unknown_target_result: 0

## Existing KL metrics

- Status: available; count: 8
- Mean/min/max: 0/0/0; first/last: 0/0

## by_source

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| city | 11 | 11 | 0 | 0 | 11 | 0 | 11 | 0 | 88 | 83 |
| rgbdt | 15 | 15 | 0 | 0 | 15 | 0 | 15 | 0 | 120 | 119 |
| robo | 6 | 6 | 0 | 0 | 6 | 0 | 6 | 0 | 48 | 28 |

## by_negative_kind

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extent | 12 | 12 | 0 | 0 | 12 | 0 | 12 | 0 | 96 | 95 |
| none | 13 | 13 | 0 | 0 | 13 | 0 | 13 | 0 | 104 | 89 |
| wrong_object | 7 | 7 | 0 | 0 | 7 | 0 | 7 | 0 | 56 | 46 |

## by_sampling_pool

| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extent | 10 | 10 | 0 | 0 | 10 | 0 | 10 | 0 | 80 | 79 |
| regular_city | 11 | 11 | 0 | 0 | 11 | 0 | 11 | 0 | 88 | 83 |
| regular_rgbdt | 3 | 3 | 0 | 0 | 3 | 0 | 3 | 0 | 24 | 24 |
| regular_robo | 2 | 2 | 0 | 0 | 2 | 0 | 2 | 0 | 16 | 6 |
| wrong_object | 6 | 6 | 0 | 0 | 6 | 0 | 6 | 0 | 48 | 38 |
