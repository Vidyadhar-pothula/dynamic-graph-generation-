# v1 (loss-selected, unweighted) -> v2 (class-balanced, macro-F1-selected)

| Model | Test set | Accuracy | Macro-F1 | Predicates predicted | Predicates with recall>0 (of present) |
|---|---|---|---|---|---|
| Relation transformer | old val | 0.529 -> 0.314 | 0.084 -> 0.064 | 8 -> 14 | 2 -> 3 (of 11) |
| Relation transformer | new videos | 0.407 -> 0.406 | 0.124 -> 0.189 | 11 -> 23 | 10 -> 14 (of 23) |
| Temporal LSTM | old val | 0.586 -> 0.322 | 0.099 -> 0.059 | 4 -> 10 | 3 -> 3 (of 11) |
| Temporal LSTM | new videos | 0.372 -> 0.339 | 0.074 -> 0.145 | 8 -> 17 | 6 -> 13 (of 23) |
| Temporal GRU | old val | 0.628 -> 0.396 | 0.113 -> 0.130 | 3 -> 13 | 3 -> 7 (of 11) |
| Temporal GRU | new videos | 0.390 -> 0.333 | 0.063 -> 0.172 | 4 -> 22 | 4 -> 14 (of 23) |
