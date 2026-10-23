"""End-to-end dynamic scene graph generation on ONE video.

    video -> YOLOv8s detector + ByteTrack -> appearance / spatial / motion features
          -> Relation Transformer -> temporal GRU -> scene graph per sampled frame

Uses the trained weights stored in this repository (relation/), so it runs straight from a clone:

    python scripts/22_run_dynamic_scene_graph.py --video path/to/video.mp4

Outputs (default: outputs/scene_graphs/<video name>/):
    relations.json   every sampled frame: tracked objects and the relations predicted by the relation model and by the GRU
    graph_01.png ... frame | scene graph (relation model) | dynamic scene graph (+ temporal GRU)
    summary.txt      how many relations of each type were predicted

Frames are sampled at 1 fps, like the training data. Scores of this system are low (see comparison/), so the graphs
contain wrong and missing relations. The example graphs are the frames with the most varied predicted relations.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features.appearance import AppearanceExtractor
from src.features.feature_fusion import fuse_features
from src.features.motion import extract_motion_features
from src.features.spatial import extract_spatial_features
from src.relations.relation_transformer import RelationTransformer
from src.temporal.temporal_gru import TemporalGRU


def parse_args():
    p = argparse.ArgumentParser(description="Run the full dynamic scene graph pipeline on one video.")
    p.add_argument("--video", required=True, help="path to an .mp4 video")
    p.add_argument("--out", default=None, help="output folder (default outputs/scene_graphs/<video name>)")
    p.add_argument("--detector", default=str(ROOT / "relation/weights/detector/yolov8s_78class_finetuned_best.pt"))
    p.add_argument("--relation", default=str(ROOT / "relation/v2_balanced/weights/best_relation_model_78class_v2_balanced.pt"))
    p.add_argument("--temporal", default=str(ROOT / "relation/v2_balanced/weights/best_temporal_gru_model_v2_balanced.pt"))
    p.add_argument("--vocab", default=str(ROOT / "relation/results/relation_vocabulary.json"))
    p.add_argument("--classes", default=str(ROOT / "relation/results/class_mapping_78.json"))
    p.add_argument("--fps", type=float, default=1.0, help="frames per second to analyse (training used 1)")
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="detector NMS IoU")
    p.add_argument("--max-objects", type=int, default=8, help="most confident objects kept per frame")
    p.add_argument("--max-edges", type=int, default=8, help="relations drawn per frame")
    p.add_argument("--min-prob", type=float, default=0.25, help="minimum probability of a drawn relation")
    p.add_argument("--max-graphs", type=int, default=6, help="number of example graph images to draw")
    p.add_argument("--window", type=int, default=10, help="temporal window (frames) used by the GRU")
    p.add_argument("--device", default="cpu", help="device for features and relation models (cpu or mps)")
    p.add_argument("--track-device", default=None, help="device for detection and tracking (default mps if available, else cpu)")
    return p.parse_args()


def load_models(a, n_rel):
    rel = RelationTransformer(num_object_classes=78, num_relation_classes=n_rel)
    rel.load_state_dict(torch.load(a.relation, map_location="cpu")); rel.to(a.device).eval()
    gru = TemporalGRU(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=n_rel)
    gru.load_state_dict(torch.load(a.temporal, map_location="cpu")); gru.to(a.device).eval()
    return rel, gru


def edges_from(P, pairs, id2pred, min_prob, max_edges):
    """One relation per object pair (the most likely non-background class), strongest first."""
    out = []
    for k, (a, b) in enumerate(pairs):
        q = P[k].copy(); q[0] = -1.0
        c = int(q.argmax())
        if q[c] >= min_prob:
            out.append({"subject": int(a), "predicate": id2pred[c], "object": int(b), "probability": round(float(q[c]), 3)})
    return sorted(out, key=lambda e: -e["probability"])[:max_edges]


def analyse_video(a, id2pred, id2cls):
    from ultralytics import YOLO

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    step = max(1, int(round(fps / a.fps)))
    print(f"Video: {width}x{height}, {fps:.1f} fps, {n_frames} frames -> analysing every {step}th frame")

    track_dev = a.track_device or ("mps" if torch.backends.mps.is_available() else "cpu")
    detector = YOLO(a.detector)
    appearance = AppearanceExtractor(device=a.device)
    rel, gru = load_models(a, len(id2pred))

    prev_spatial = {}      # track id -> (spatial vector, frame index)
    history = {}           # (subject, object) -> list of pair vectors over sampled frames
    frames_out = []
    stream = detector.track(source=a.video, tracker="bytetrack.yaml", persist=True, stream=True,
                            conf=a.conf, iou=a.iou, device=track_dev, verbose=False)
    for idx, r in enumerate(stream):
        if idx % step or r.boxes is None or r.boxes.id is None:
            continue
        frame = r.orig_img
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int); ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int); conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-conf)[: a.max_objects]
        boxes = [[int(v) for v in xyxy[i]] for i in order]
        app = appearance.extract_batch_crops(frame, boxes)
        obj = {}
        for j, i in enumerate(order):
            tid = int(ids[i]); sp = extract_spatial_features(boxes[j], width, height)
            mo = extract_motion_features(sp, *(prev_spatial[tid][0], idx - prev_spatial[tid][1])) if tid in prev_spatial else np.zeros(6, np.float32)
            prev_spatial[tid] = (sp, idx)
            obj[tid] = {"app": app[j], "sp": sp, "mo": mo, "cls": int(cls[i]), "bbox": boxes[j], "conf": float(conf[i])}
        tids = list(obj)
        pairs = [(s, o) for s in tids for o in tids if s != o]
        if not pairs:
            continue
        vec = np.stack([np.concatenate([fuse_features(obj[s]["app"], obj[s]["sp"], obj[s]["mo"]),
                                        fuse_features(obj[o]["app"], obj[o]["sp"], obj[o]["mo"])]) for s, o in pairs]).astype(np.float32)
        with torch.no_grad():
            p_rel = torch.softmax(rel(torch.tensor(vec[:, :527]).to(a.device), torch.tensor(vec[:, 527:]).to(a.device),
                                      torch.tensor([obj[s]["cls"] for s, _ in pairs]).to(a.device),
                                      torch.tensor([obj[o]["cls"] for _, o in pairs]).to(a.device)), 1).cpu().numpy()
            seqs = []
            for k, pr in enumerate(pairs):
                history.setdefault(pr, []).append(vec[k]); h = history[pr][-a.window:]
                seqs.append(np.stack([np.zeros(1054, np.float32)] * (a.window - len(h)) + h))
            p_tmp = torch.softmax(gru(torch.tensor(np.stack(seqs)).to(a.device))[:, -1, :], 1).cpu().numpy()
        frames_out.append({
            "frame_index": idx, "time_sec": round(idx / fps, 2),
            "objects": [{"track_id": t, "class": id2cls[o["cls"]], "bbox": o["bbox"], "confidence": round(o["conf"], 3)} for t, o in obj.items()],
            "relation_model": edges_from(p_rel, pairs, id2pred, a.min_prob, a.max_edges),
            "temporal_gru": edges_from(p_tmp, pairs, id2pred, a.min_prob, a.max_edges)})
        if len(frames_out) % 10 == 0:
            print(f"  analysed {len(frames_out)} frames", flush=True)
    return frames_out, {"width": width, "height": height, "fps": round(fps, 2), "frames": n_frames, "step": step}


def draw_graph(ax, objects, edge_list, title):
    import networkx as nx
    names = {o["track_id"]: f'ID {o["track_id"]}\n{o["class"]}' for o in objects}
    G = nx.DiGraph()
    used = {e["subject"] for e in edge_list} | {e["object"] for e in edge_list}
    G.add_nodes_from(used or set(names))
    for e in edge_list:
        G.add_edge(e["subject"], e["object"], label=e["predicate"])
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.axis("off")
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no relations predicted", ha="center", va="center", fontsize=9); return
    pos = nx.circular_layout(G) if G.number_of_nodes() > 2 else nx.spring_layout(G, seed=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#cfe3f1", edgecolors="#34495e", node_size=900)
    nx.draw_networkx_labels(G, pos, labels={n: names[n] for n in G.nodes}, ax=ax, font_size=6, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12, node_size=900, connectionstyle="arc3,rad=0.18", edge_color="#555")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), ax=ax, font_size=6,
                                 bbox=dict(boxstyle="round,pad=0.15", fc="#fff4d6", ec="#c9a227", lw=0.5), connectionstyle="arc3,rad=0.18")


def draw_examples(a, frames_out, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    def variety(f):
        d = {e["predicate"] for e in f["relation_model"]} | {e["predicate"] for e in f["temporal_gru"]}
        return 3 * len(d) + min(len(f["relation_model"]), 6) + min(len(f["temporal_gru"]), 6)
    cand = [f for f in frames_out if len(f["objects"]) >= 3 and f["relation_model"]]
    chosen = sorted(cand, key=lambda f: (-variety(f), f["frame_index"]))[: a.max_graphs]
    cap = cv2.VideoCapture(a.video); n = 0
    for f in sorted(chosen, key=lambda f: f["frame_index"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame_index"]); ok, img = cap.read()
        if not ok:
            continue
        n += 1
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].axis("off")
        axs[0].set_title(f'FRAME {f["frame_index"]}  (t = {f["time_sec"]} s)', fontsize=9, fontweight="bold")
        for o in f["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            axs[0].add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.6))
            axs[0].text(x1 + 2, y1 + 11, f'ID {o["track_id"]} | {o["class"]}', color="white", fontsize=6.5, fontweight="bold", bbox=dict(fc="red", ec="none", pad=1))
        draw_graph(axs[1], f["objects"], f["relation_model"], "SCENE GRAPH\n(relation model)")
        draw_graph(axs[2], f["objects"], f["temporal_gru"], "DYNAMIC SCENE GRAPH\n(+ temporal GRU)")
        fig.tight_layout(); fig.savefig(out_dir / f"graph_{n:02d}.png", dpi=110); plt.close(fig)
    cap.release()
    return n


def main():
    a = parse_args()
    vocab = json.load(open(a.vocab)); id2pred = {v: k for k, v in vocab.items()}
    id2cls = {v: k for k, v in json.load(open(a.classes)).items()}
    out_dir = Path(a.out) if a.out else ROOT / "outputs/scene_graphs" / Path(a.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_out, info = analyse_video(a, id2pred, id2cls)
    if not frames_out:
        sys.exit("No tracked objects were found in the sampled frames.")
    json.dump({"video": str(a.video), "video_info": info, "frames": frames_out}, open(out_dir / "relations.json", "w"), indent=1)
    n = draw_examples(a, frames_out, out_dir)

    cnt_rel = Counter(e["predicate"] for f in frames_out for e in f["relation_model"])
    cnt_tmp = Counter(e["predicate"] for f in frames_out for e in f["temporal_gru"])
    lines = [f"video: {a.video}", f"frames analysed: {len(frames_out)}", f"graph images: {n}", "",
             f'{"predicate":<20}{"relation model":>16}{"temporal GRU":>16}']
    for p in sorted(set(cnt_rel) | set(cnt_tmp), key=lambda p: -(cnt_rel[p] + cnt_tmp[p])):
        lines.append(f"{p:<20}{cnt_rel[p]:>16}{cnt_tmp[p]:>16}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
