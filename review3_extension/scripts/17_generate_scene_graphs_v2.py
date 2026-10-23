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

OUT = ROOT / "relation" / "v2_balanced"
for d in ("weights", "graphs", "relations", "results"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
MAX_OBJECTS, MAX_EDGES, WINDOW, N_EXAMPLES, CONF_MIN = 8, 8, 10, 10, 0.25

vocab = json.load(open(ROOT / "data/processed/relation_vocabulary.json"))
id2pred = {v: k for k, v in vocab.items()}
cat = json.load(open(ROOT / "data/processed/category_mapping.json"))
id2cls = {v: k for k, v in cat.items()}

rel = RelationTransformer(num_object_classes=78, num_relation_classes=len(vocab))
rel.load_state_dict(torch.load(ROOT / "models/relation/best_relation_model_78class_v2_balanced.pt", map_location="cpu")); rel.eval()
gru = TemporalGRU(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=len(vocab))
gru.load_state_dict(torch.load(ROOT / "models/temporal/best_temporal_gru_model_v2_balanced.pt", map_location="cpu")); gru.eval()

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
    def variety(f):
        d = len({e["predicate"] for e in f["relation_model"]} | {e["predicate"] for e in f["temporal_gru"]})
        return 3 * d + min(len(f["relation_model"]), 6) + min(len(f["temporal_gru"]), 6)
    cand = [f for f in frames if 3 <= len(f["objects"]) <= MAX_OBJECTS and len(f["temporal_gru"]) >= 2 and len(f["relation_model"]) >= 2]
    if cand:
        fr = max(cand, key=lambda f: (variety(f), -f["frame_id"]))
        best.append((variety(fr), vid, sub, fr))
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
for src, dst in [("models/relation/best_relation_model_78class_v2_balanced.pt", W / "best_relation_model_78class_v2_balanced.pt"),
                 ("models/temporal/best_temporal_lstm_model_v2_balanced.pt", W / "best_temporal_lstm_model_v2_balanced.pt"),
                 ("models/temporal/best_temporal_gru_model_v2_balanced.pt", W / "best_temporal_gru_model_v2_balanced.pt")]:
    shutil.copy2(ROOT / src, dst)

# ---- results ----
R = json.load(open(ROOT / "logs/relation_temporal_results_v2.json"))
shutil.copy2(ROOT / "logs/relation_temporal_results_v2.json", OUT / "results/relation_temporal_results_v2.json")
shutil.copy2(ROOT / "logs/relation_temporal_results.json", OUT / "results/v1_relation_temporal_results.json")
md = ["# v1 (loss-selected, unweighted) -> v2 (class-balanced, macro-F1-selected)", "",
      "| Model | Test set | Accuracy | Macro-F1 | Predicates predicted | Predicates with recall>0 (of present) |", "|---|---|---|---|---|---|"]
def line(name, node):
    for ts, tk in (("old val", "old_val"), ("new videos", "new_val")):
        b, a = node[f"v1_{tk}"], node[f"v2_{tk}"]
        md.append(f"| {name} | {ts} | {b['acc']:.3f} -> {a['acc']:.3f} | {b['macro_F1']:.3f} -> {a['macro_F1']:.3f} | {b['classes_predicted']} -> {a['classes_predicted']} | {b['classes_recall_gt0']} -> {a['classes_recall_gt0']} (of {a['classes_present']}) |")
line("Relation transformer", R["relation"]); line("Temporal LSTM", R["temporal"]["lstm"]); line("Temporal GRU", R["temporal"]["gru"])
(OUT / "results/results_v2.md").write_text("\n".join(md) + "\n")
(OUT / "README.md").write_text(f"""# relation/v2_balanced - graphs after fixing the predicate imbalance

Why v2 exists: `next_to` and `in_front_of` are 72% of all positive training labels, the v1 loss was unweighted and v1 checkpoints
were chosen by validation loss (which favours majority guesses), so v1 predicted only 11 of 32 predicates.
v2 uses class-balanced loss weights (inverse square root of frequency, capped at 12x) and picks the checkpoint by macro-F1.

* `weights/` v2 relation, LSTM and GRU. v1 files are untouched in `../weights/`.
* `graphs/` {len(chosen)} examples (frame | relation model | dynamic graph with temporal GRU) and `contact_sheet.png`.
  Examples are the held-out frames with the most DIFFERENT predicates (not a random sample).
* `relations/` per-frame predicted triplets for the {len(hold)} held-out new videos.
* `results/` v1 -> v2 table, metrics JSON, predicted-relation counts.
Edge rule: top class is not background and probability >= {CONF_MIN:.2f}, at most {MAX_EDGES} edges per frame.
""")
print("done ->", OUT)
