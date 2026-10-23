# relation/ - dynamic scene graph outputs (78-class pipeline)

| Folder | What is in it |
|---|---|
| `weights/` | fine-tuned relation transformer (78 classes), temporal LSTM and GRU, the original 28-class models (`original_28class/`), and the fine-tuned YOLOv8s detector (`detector/`) |
| `graphs/` | `qual_01.png` ... `qual_10.png`: reference frame | scene graph from the relation model | dynamic scene graph with the temporal GRU. `contact_sheet.png` has them all |
| `relations/` | one JSON per held-out video with the predicted triplets for every frame (relation model and temporal GRU) |
| `results/` | metrics (`results.md`, `relation_temporal_results.json`), detector per-epoch results, predicate counts, class and predicate vocabularies |

## How the graphs were made
* Videos: the 38 NEW videos that were held out of relation/temporal training (every 10th), so the models never trained on them.
* Objects and tracks come from the fine-tuned 78-class YOLOv8s detector with ByteTrack.
* For every pair of objects in a frame the relation model predicts a predicate; edges are kept when the top class is not background and its probability is at least 0.30, at most 8 edges per frame.
* The temporal GRU sees the last 10 frames of each object pair and gives the dynamic prediction.
* The 10 shown examples are the held-out frames with the most predicted relations (3 to 8 objects), so they are favourable examples, not a random sample.

## Honest caveats
* Relation macro-F1 is about 0.12 and the temporal models are about 0.06 to 0.07 on new videos, so these graphs contain many wrong or missing relations.
* Class names come from the 78-class detector; the relation model was expanded from 28 to 78 classes by class name.
