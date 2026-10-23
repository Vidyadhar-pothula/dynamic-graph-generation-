"""Generate dynamic scene graphs with the fine-tuned models and collect everything in cv_work/relation/.

Uses the NEW videos that were held out of relation/temporal training (every 10th, 38 videos), so the graphs are
predictions on videos the models never trained on. Tracks and features come from the fine-tuned 78-class detector.

Outputs (cv_work/relation/):
  weights/    relation model (78-class), temporal LSTM + GRU, originals (28-class) and the fine-tuned detector
  graphs/     qual_01..10.png (frame | relation-model graph | dynamic graph with temporal GRU) + contact_sheet.png
  relations/  <video>.json with per-frame predicted triplets (relation model and temporal GRU)
  results/    metrics json/csv/markdown
  README.md
"""
import sys, json, pickle, shutil, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.relations.relation_transformer import RelationTransformer
from src.temporal.temporal_gru import TemporalGRU
from src.features.feature_fusion import fuse_features

OUT = ROOT / "relation"
for d in ("weights", "graphs", "relations", "results"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
MAX_OBJECTS, MAX_EDGES, WINDOW, N_EXAMPLES, CONF_MIN = 8, 8, 10, 10, 0.30

vocab = json.load(open(ROOT / "data/processed/relation_vocabulary.json"))
id2pred = {v: k for k, v in vocab.items()}
cat = json.load(open(ROOT / "data/processed/category_mapping.json"))
id2cls = {v: k for k, v in cat.items()}

rel = RelationTransformer(num_object_classes=78, num_relation_classes=len(vocab))
rel.load_state_dict(torch.load(ROOT / "models/relation/best_relation_model_78class.pt", map_location="cpu")); rel.eval()
gru = TemporalGRU(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=len(vocab))
gru.load_state_dict(torch.load(ROOT / "models/temporal/best_temporal_gru_model_finetuned.pt", map_location="cpu")); gru.eval()

# ---- the 38 held-out new videos (same rule as the fine-tune script) ----
new_rel = pickle.load(open(ROOT / "data/processed/relation_train.pkl", "rb"))
vids = sorted({s["video_id"] for s in new_rel})
hold = [v for i, v in enumerate(vids) if i % 10 == 0]
feat_path = {p.stem: p for p in (ROOT / "data/processed/features/train").glob("*/*.pkl")}
print(f"{len(hold)} held-out videos")


def pair_vec(a, b):
    return np.concatenate([fuse_features(a["appearance"], a["spatial"], a["motion"]),
                           fuse_features(b["appearance"], b["spatial"], b["motion"])]).astype(np.float32)


def analyse_video(vid):
    """Per frame: objects, relation-model edges, temporal-GRU edges."""
    ff = pickle.load(open(feat_path[vid], "rb"))["frame_features"]
    sub = feat_path[vid].parent.name
    fids = sorted(ff)
    hist = {}   # (a,b) -> list of pair vectors up to the current frame
    frames = []
    for fid in fids:
        tr = ff[fid]
        tids = sorted(tr, key=lambda t: -tr[t]["confidence"])[:MAX_OBJECTS]
        pairs = [(a, b) for a in tids for b in tids if a != b]
        if not pairs:
            continue
        vecs = np.stack([pair_vec(tr[a], tr[b]) for a, b in pairs])
        sf = torch.tensor(vecs[:, :527]); of = torch.tensor(vecs[:, 527:])
        sc = torch.tensor([tr[a]["class_id"] for a, _ in pairs]); oc = torch.tensor([tr[b]["class_id"] for _, b in pairs])
        with torch.no_grad():
            p_rel = torch.softmax(rel(sf, of, sc, oc), 1).numpy()
        # temporal window per pair (history grows frame by frame)
        seqs = []
        for k, pr in enumerate(pairs):
            hist.setdefault(pr, []).append(vecs[k])
            h = hist[pr][-WINDOW:]
            pad = [np.zeros(1054, dtype=np.float32)] * (WINDOW - len(h))
            seqs.append(np.stack(pad + h))
        with torch.no_grad():
            p_tmp = torch.softmax(gru(torch.tensor(np.stack(seqs)))[:, -1, :], 1).numpy()

        def edges(P):
            out = []
            for k, (a, b) in enumerate(pairs):
                c = int(P[k].argmax())
                if c != 0 and P[k][c] >= CONF_MIN:
                    out.append({"subject": int(a), "predicate": id2pred[c], "object": int(b), "confidence": round(float(P[k][c]), 3)})
            return sorted(out, key=lambda e: -e["confidence"])[:MAX_EDGES]

        frames.append({"frame_id": int(fid),
                       "objects": [{"track_id": int(t), "class": id2cls[int(tr[t]["class_id"])], "bbox": [int(x) for x in tr[t]["bbox"]], "conf": round(float(tr[t]["confidence"]), 3)} for t in tids],
                       "relation_model": edges(p_rel), "temporal_gru": edges(p_tmp)})
    return sub, frames


def draw_graph(ax, objects, edge_list, title):
    names = {o["track_id"]: f'ID {o["track_id"]} | {o["class"]}' for o in objects}
    G = nx.DiGraph()
    used = {e["subject"] for e in edge_list} | {e["object"] for e in edge_list}
    for t in (used or set(names)):
        G.add_node(t)
    for e in edge_list:
        G.add_edge(e["subject"], e["object"], label=e["predicate"])
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.axis("off")
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no relations predicted", ha="center", va="center", fontsize=9); return
    pos = nx.circular_layout(G) if G.number_of_nodes() > 2 else nx.spring_layout(G, seed=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#cfe3f1", edgecolors="#34495e", node_size=900)
    nx.draw_networkx_labels(G, pos, labels={n: names[n].replace(" | ", "\n") for n in G.nodes}, ax=ax, font_size=6, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12, node_size=900, connectionstyle="arc3,rad=0.18", edge_color="#555")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), ax=ax, font_size=6,
                                 bbox=dict(boxstyle="round,pad=0.15", fc="#fff4d6", ec="#c9a227", lw=0.5), connectionstyle="arc3,rad=0.18")


def make_figure(k, vid, sub, fr, path):
    img = Image.open(ROOT / f"data/frames/train/{sub}/{vid}/frame_{fr['frame_id']:06d}.jpg")
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    axs[0].imshow(img); axs[0].axis("off")
    axs[0].set_title(f"REFERENCE FRAME\nVideo {vid} | Frame {fr['frame_id']}", fontsize=9, fontweight="bold")
    for o in fr["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        axs[0].add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.6))
        axs[0].text(x1 + 2, y1 + 11, f'ID {o["track_id"]} | {o["class"]}', color="white", fontsize=6.5, fontweight="bold", bbox=dict(fc="red", ec="none", pad=1))
    draw_graph(axs[1], fr["objects"], fr["relation_model"], "PREDICTED SCENE GRAPH\n(relation model)")
    draw_graph(axs[2], fr["objects"], fr["temporal_gru"], "DYNAMIC SCENE GRAPH\n(+ temporal GRU)")
    preds = sorted({e["predicate"] for e in fr["temporal_gru"]} | {e["predicate"] for e in fr["relation_model"]})
    fig.suptitle(f"Qualitative Result #{k:02d}  -  Detected relations: {' | '.join(preds) if preds else 'none'}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path, dpi=110); plt.close(fig)


results = {}; best = []
for n, vid in enumerate(hold, 1):
    sub, frames = analyse_video(vid)
    results[vid] = {"subfolder": sub, "frames": frames}
    json.dump({"video_id": vid, "note": "predictions on a held-out new video, fine-tuned 78-class models", "frames": frames},
              open(OUT / "relations" / f"{vid}.json", "w"), indent=1)
    cand = [f for f in frames if 3 <= len(f["objects"]) <= MAX_OBJECTS and len(f["temporal_gru"]) >= 2 and len(f["relation_model"]) >= 2]
    if cand:
        fr = max(cand, key=lambda f: (min(len(f["relation_model"]), 6) + min(len(f["temporal_gru"]), 6), -f["frame_id"]))
        best.append((min(len(fr["relation_model"]), 6) + min(len(fr["temporal_gru"]), 6), vid, sub, fr))
    print(f"[{n}/{len(hold)}] {vid}: {len(frames)} frames", flush=True)

best.sort(key=lambda x: (-x[0], x[1]))
chosen = best[:N_EXAMPLES]
imgs = []
for k, (_, vid, sub, fr) in enumerate(chosen, 1):
    p = OUT / "graphs" / f"qual_{k:02d}.png"
    make_figure(k, vid, sub, fr, p); imgs.append(Image.open(p)); print("graph", p.name, vid, fr["frame_id"])
w = max(i.width for i in imgs); h = max(i.height for i in imgs)
sheet = Image.new("RGB", (w * 2, h * ((len(imgs) + 1) // 2)), "white")
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % 2) * w, (i // 2) * h))
sheet.save(OUT / "graphs" / "contact_sheet.png")

# ---- summary of what was predicted ----
cnt_rel, cnt_tmp = {}, {}
for v in results.values():
    for f in v["frames"]:
        for e in f["relation_model"]: cnt_rel[e["predicate"]] = cnt_rel.get(e["predicate"], 0) + 1
        for e in f["temporal_gru"]: cnt_tmp[e["predicate"]] = cnt_tmp.get(e["predicate"], 0) + 1
with open(OUT / "results" / "predicted_relation_counts.csv", "w", newline="") as f:
    wtr = csv.writer(f); wtr.writerow(["predicate", "relation_model_edges", "temporal_gru_edges"])
    for p in sorted(set(cnt_rel) | set(cnt_tmp), key=lambda p: -(cnt_rel.get(p, 0) + cnt_tmp.get(p, 0))):
        wtr.writerow([p, cnt_rel.get(p, 0), cnt_tmp.get(p, 0)])

# ---- weights ----
W = OUT / "weights"
(W / "original_28class").mkdir(exist_ok=True); (W / "detector").mkdir(exist_ok=True)
for src, dst in [("models/relation/best_relation_model_78class.pt", W / "best_relation_model_78class.pt"),
                 ("models/temporal/best_temporal_lstm_model_finetuned.pt", W / "best_temporal_lstm_model_finetuned.pt"),
                 ("models/temporal/best_temporal_gru_model_finetuned.pt", W / "best_temporal_gru_model_finetuned.pt"),
                 ("models/relation/best_relation_model.pt", W / "original_28class/best_relation_model_28class_original.pt"),
                 ("models/temporal/best_temporal_lstm_model.pt", W / "original_28class/best_temporal_lstm_model_original.pt"),
                 ("models/temporal/best_temporal_gru_model.pt", W / "original_28class/best_temporal_gru_model_original.pt"),
                 ("models/detection_project/finetune/weights/best.pt", W / "detector/yolov8s_78class_finetuned_best.pt")]:
    if (ROOT / src).exists():
        shutil.copy2(ROOT / src, dst)

# ---- results ----
R = json.load(open(ROOT / "logs/relation_temporal_results.json"))
shutil.copy2(ROOT / "logs/relation_temporal_results.json", OUT / "results/relation_temporal_results.json")
shutil.copy2(ROOT / "models/detection_project/finetune/results.csv", OUT / "results/detector_finetune_results.csv")
shutil.copy2(ROOT / "relation_vocabulary.json" if (ROOT / "relation_vocabulary.json").exists() else ROOT / "data/processed/relation_vocabulary.json", OUT / "results/relation_vocabulary.json")
shutil.copy2(ROOT / "data/processed/category_mapping.json", OUT / "results/class_mapping_78.json")
md = ["# Results (before -> after fine-tuning)", "", "| Model | Test set | Accuracy | Macro-F1 | Loss |", "|---|---|---|---|---|"]
def line(name, key, node):
    for ts, tk in (("old val", "old_val"), ("new videos", "new_val")):
        b, a = node[f"baseline_{tk}"], node[f"finetuned_{tk}"]
        md.append(f"| {name} | {ts} | {b['acc']:.3f} -> {a['acc']:.3f} | {b['macro_F1']:.3f} -> {a['macro_F1']:.3f} | {b['loss']:.3f} -> {a['loss']:.3f} |")
line("Relation transformer", "relation", R["relation"]); line("Temporal LSTM", "lstm", R["temporal"]["lstm"]); line("Temporal GRU", "gru", R["temporal"]["gru"])
md += ["", "Detector (same 3,594 validation images): mAP50 0.2635 -> 0.2602, mAP50-95 0.1526 -> 0.1548, recall 0.285 -> 0.303."]
(OUT / "results/results.md").write_text("\n".join(md) + "\n")

(OUT / "README.md").write_text(f"""# relation/ - dynamic scene graph outputs (78-class pipeline)

| Folder | What is in it |
|---|---|
| `weights/` | fine-tuned relation transformer (78 classes), temporal LSTM and GRU, the original 28-class models (`original_28class/`), and the fine-tuned YOLOv8s detector (`detector/`) |
| `graphs/` | `qual_01.png` ... `qual_{len(chosen):02d}.png`: reference frame | scene graph from the relation model | dynamic scene graph with the temporal GRU. `contact_sheet.png` has them all |
| `relations/` | one JSON per held-out video with the predicted triplets for every frame (relation model and temporal GRU) |
| `results/` | metrics (`results.md`, `relation_temporal_results.json`), detector per-epoch results, predicate counts, class and predicate vocabularies |

## How the graphs were made
* Videos: the {len(hold)} NEW videos that were held out of relation/temporal training (every 10th), so the models never trained on them.
* Objects and tracks come from the fine-tuned 78-class YOLOv8s detector with ByteTrack.
* For every pair of objects in a frame the relation model predicts a predicate; edges are kept when the top class is not background and its probability is at least {CONF_MIN:.2f}, at most {MAX_EDGES} edges per frame.
* The temporal GRU sees the last {WINDOW} frames of each object pair and gives the dynamic prediction.
* The {len(chosen)} shown examples are the held-out frames with the most predicted relations (3 to {MAX_OBJECTS} objects), so they are favourable examples, not a random sample.

## Honest caveats
* Relation macro-F1 is about 0.12 and the temporal models are about 0.06 to 0.07 on new videos, so these graphs contain many wrong or missing relations.
* Class names come from the 78-class detector; the relation model was expanded from 28 to 78 classes by class name.
""")
print("done ->", OUT)
