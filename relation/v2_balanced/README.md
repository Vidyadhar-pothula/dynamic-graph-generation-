# relation/v2_balanced - graphs after fixing the predicate imbalance

Why v2 exists: `next_to` and `in_front_of` are 72% of all positive training labels, the v1 loss was unweighted and v1 checkpoints
were chosen by validation loss (which favours majority guesses), so v1 predicted only 11 of 32 predicates.
v2 uses class-balanced loss weights (inverse square root of frequency, capped at 12x) and picks the checkpoint by macro-F1.

* `weights/` v2 relation, LSTM and GRU. v1 files are untouched in `../weights/`.
* `graphs/` 10 examples (frame | relation model | dynamic graph with temporal GRU) and `contact_sheet.png`.
  Examples are the held-out frames with the most DIFFERENT predicates (not a random sample).
* `relations/` per-frame predicted triplets for the 38 held-out new videos.
* `results/` v1 -> v2 table, metrics JSON, predicted-relation counts.
Edge rule: top class is not background and probability >= 0.25, at most 8 edges per frame.
