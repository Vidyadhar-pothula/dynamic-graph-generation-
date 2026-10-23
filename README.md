# Dynamic Scene Graph Generation from Video (VidOR)

Detect objects in video, track them through time and predict how they relate (for example *adult - holds - baby*),
producing a scene graph for every frame that follows the same objects across frames. Built on the VidOR dataset
(frames sampled at 1 fps) with a YOLOv8s detector, ByteTrack tracking, a relation transformer and LSTM/GRU temporal models.

## Pipeline

| Step | Script | Module in `src/` |
|---|---|---|
| 1-4 | `01_prepare_dataset.py` ... `04_prepare_yolo.py`: extract videos, parse annotations, sample frames, build YOLO data | `dataset/`, `preprocessing/` |
| 5 | `05_train_yolo.py`: train the detector | `detection/` |
| 6-7 | `06_run_tracking.py`, `07_extract_features.py`: ByteTrack tracks, appearance / spatial / motion features | `tracking/`, `features/` |
| 8-9 | `08_prepare_relation_data.py`, `09_train_relation_transformer.py` | `relations/` |
| 10-11 | `10_prepare_temporal_data.py`, `11_train_temporal_model.py` (LSTM, GRU) | `temporal/` |
| 12-15 | full pipeline, validation, evaluation, packaging | `graph/` |

The scripts in `scripts/` are the original pipeline (written for an Apple Silicon Mac, MPS device).

## Review 3 extension (`review3_extension/`)

Three more groups of VidOR videos (750 videos, 27,311 frames) were added to the earlier training data
(33,277 frames; 60,588 in total, same 3,594 validation frames) and the models were fine-tuned, not retrained from scratch.

* **Detector (78 classes):** the Colab-trained YOLOv8s was fine-tuned for 3 epochs on all frames.
* **Class expansion:** the relation model had 28 object classes; it was expanded to 78 by copying weights by class name
  (verified lossless: identical output on the old validation set).
* **Relation and temporal models** were fine-tuned on old + new data; 38 unseen new videos (every 10th) were held out.
* **Predicate imbalance:** `next_to` and `in_front_of` are 72% of all positive labels, so v1 predicted only 11 of 32 predicates.
  v2 retrains with class-balanced loss and picks checkpoints by macro-F1 (`scripts/15_diagnose_predicates.py`, `16_train_balanced.py`).

### Results

Detector, same 3,594 validation images:

| | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Colab run 3 (starting point) | 0.2635 | 0.1526 | 0.381 | 0.285 |
| Fine-tuned on all parts | 0.2602 | 0.1548 | 0.331 | 0.303 |

More data did not raise mAP50. Relation / temporal models on the 38 unseen new videos (v1 = loss-selected, v2 = class-balanced):

| Model | Macro-F1 v1 -> v2 | Accuracy v1 -> v2 | Predicates used v1 -> v2 |
|---|---|---|---|
| Relation transformer | 0.124 -> 0.189 | 0.407 -> 0.406 | 11 -> 23 |
| Temporal LSTM | 0.074 -> 0.145 | 0.372 -> 0.339 | 8 -> 17 |
| Temporal GRU | 0.063 -> 0.172 | 0.390 -> 0.333 | 4 -> 22 |

On the small old validation set v2 is mixed (relation and LSTM worse, GRU better). Scores remain low overall.

## Folders

| Folder | Contents |
|---|---|
| `configs/` | pipeline configuration |
| `src/`, `scripts/` | original pipeline code |
| `review3_extension/scripts/` | patched data scripts (04, 06, 08 keep the trained class and predicate IDs) and the fine-tune, diagnosis, v2 training and graph-generation scripts |
| `review3_extension/results/` | detector fine-tune results and curves, relation / temporal metrics (v1 and v2) |
| `relation/` | v1 weights, dynamic scene graph images, per-video predicted relations. `relation/v2_balanced/` holds the same for the balanced v2 models |
| `deck/` | Review 3 presentation: `presentation.tex` / `.pdf` (LaTeX) and the editable `Computer_Vision_Review3.pptx` with the scripts that build them |

## Notes and caveats

* VidOR videos and the extracted frames are not included; download VidOR from its official source.
* The graph examples in `relation/` are held-out frames picked for the number of predicted relations (v2: for variety), not a random sample.
* Several scripts contain absolute paths from the author's machine (for example the path of the original project in
  `12b_finetune_relation_temporal.py` and `16_train_balanced.py`); edit them before running elsewhere.
* The example frames show people from the VidOR dataset; check the dataset terms before redistributing.
