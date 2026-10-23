# Results (before -> after fine-tuning)

| Model | Test set | Accuracy | Macro-F1 | Loss |
|---|---|---|---|---|
| Relation transformer | old val | 0.429 -> 0.529 | 0.068 -> 0.077 | 1.758 -> 1.490 |
| Relation transformer | new videos | 0.303 -> 0.407 | 0.098 -> 0.124 | 2.264 -> 1.781 |
| Temporal LSTM | old val | 0.645 -> 0.586 | 0.072 -> 0.099 | 1.317 -> 1.375 |
| Temporal LSTM | new videos | 0.278 -> 0.372 | 0.026 -> 0.074 | 2.100 -> 1.891 |
| Temporal GRU | old val | 0.576 -> 0.628 | 0.075 -> 0.113 | 1.510 -> 1.372 |
| Temporal GRU | new videos | 0.281 -> 0.390 | 0.032 -> 0.063 | 2.284 -> 1.809 |

Detector (same 3,594 validation images): mAP50 0.2635 -> 0.2602, mAP50-95 0.1526 -> 0.1548, recall 0.285 -> 0.303.
