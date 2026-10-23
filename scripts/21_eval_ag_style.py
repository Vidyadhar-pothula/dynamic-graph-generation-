"""Per-frame mean Recall@K (K=10,20,50) for three tasks, in the layout of the base paper's Action Genome table
(PredCLS / SGCLS / SGDET, 'with constraint': each object pair gets at most one predicate).

Our approximation (GT boxes are replaced by detector boxes matched to the ground truth, IoU >= 0.5, per frame):
  SGDET : subject and object tracks must match GT objects AND their categories; predicate must be right.
          Denominator = every ground-truth triplet active in the frame (missed objects count as misses).
  SGCLS : same matching, denominator = only triplets whose subject and object both have a detector box (IoU >= 0.5).
  PredCLS: denominator as SGCLS, but object categories are NOT required to match (labels treated as given).
mR@K = recall of each predicate class (over frames where it occurs, top-K pairs per frame by score), averaged over classes."""
import sys, json, pickle, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.relations.relation_transformer import RelationTransformer
from src.temporal.temporal_gru import TemporalGRU
from src.temporal.temporal_lstm import TemporalLSTM
from src.features.feature_fusion import fuse_features
MAX_OBJ, WINDOW, KS = 12, 10, (10, 20, 50)
vocab = json.load(open(ROOT/"data/processed/relation_vocabulary.json")); N = len(vocab)
cat = json.load(open(ROOT/"data/processed/category_mapping.json")); id2cls = {v:k for k,v in cat.items()}
def rel(path):
    m = RelationTransformer(num_object_classes=78, num_relation_classes=N); m.load_state_dict(torch.load(ROOT/path, map_location="cpu")); return m.eval()
def tmp(cls, path):
    m = cls(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=N); m.load_state_dict(torch.load(ROOT/path, map_location="cpu")); return m.eval()
MODELS = {
 "Relation Transformer": ("rel", rel("models/relation/best_relation_model_78class.pt")),
 "Relation Transformer + balanced loss": ("rel", rel("models/relation/best_relation_model_78class_v2_balanced.pt")),
 "Temporal GRU + balanced loss": ("tmp", tmp(TemporalGRU, "models/temporal/best_temporal_gru_model_v2_balanced.pt")),
 "Temporal LSTM + balanced loss": ("tmp", tmp(TemporalLSTM, "models/temporal/best_temporal_lstm_model_v2_balanced.pt")),
}
new_rel = pickle.load(open(ROOT/"data/processed/relation_train.pkl","rb")); vids = sorted({s["video_id"] for s in new_rel}); hold = [v for i,v in enumerate(vids) if i%10==0]
feat_path = {p.stem: p for p in (ROOT/"data/processed/features/train").glob("*/*.pkl")}
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); i=max(0,x2-x1)*max(0,y2-y1)
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i; return i/u if u>0 else 0
TASKS = ("PredCLS", "SGCLS", "SGDET")
# acc[model][task][K][predicate] -> list of per-frame recalls
acc = {m: {t: {K: collections.defaultdict(list) for K in KS} for t in TASKS} for m in MODELS}
nframes = 0
for vi, vid in enumerate(hold, 1):
    ff = pickle.load(open(feat_path[vid],"rb"))["frame_features"]; fids = sorted(ff)
    ann = json.load(open(ROOT/f"data/processed/annotations/train/{feat_path[vid].parent.name}/{vid}.json"))
    gt_cls = {int(k): v for k,v in ann["objects"].items()}
    traj = {int(k): {p["frame_id"]: p["bbox"] for p in v} for k,v in ann["trajectories"].items()}
    rels = [r for r in ann["relations"] if r["predicate"] in vocab]
    hist = {m: {} for m in MODELS}
    for fid in fids:
        tr = ff[fid]; tids = sorted(tr, key=lambda t: -tr[t]["confidence"])[:MAX_OBJ]
        pairs = [(a,b) for a in tids for b in tids if a != b]
        # GT triplets active at this frame
        gts = [(r["subject_id"], vocab[r["predicate"]], r["object_id"]) for r in rels
               if r["begin_fid"] <= fid <= r["end_fid"] and fid in traj.get(r["subject_id"], {}) and fid in traj.get(r["object_id"], {})]
        if not gts or not pairs: continue
        # per-frame track -> GT (best IoU >= 0.5), and whether the class agrees
        t2g = {}
        for t in tids:
            best, bi = None, 0.5
            for g, trj in traj.items():
                if fid in trj:
                    v = iou(tr[t]["bbox"], trj[fid])
                    if v >= bi: best, bi = g, v
            if best is not None: t2g[t] = (best, id2cls[int(tr[t]["class_id"])] == gt_cls[best])
        covered = [x for x in gts if any(t2g.get(t, (None,))[0] == x[0] for t in tids) and any(t2g.get(t, (None,))[0] == x[2] for t in tids)]
        nframes += 1
        vecs = np.stack([np.concatenate([fuse_features(tr[a]["appearance"],tr[a]["spatial"],tr[a]["motion"]), fuse_features(tr[b]["appearance"],tr[b]["spatial"],tr[b]["motion"])]) for a,b in pairs]).astype(np.float32)
        for name, (kind, model) in MODELS.items():
            with torch.no_grad():
                if kind == "rel":
                    P = torch.softmax(model(torch.tensor(vecs[:,:527]), torch.tensor(vecs[:,527:]), torch.tensor([tr[a]["class_id"] for a,_ in pairs]), torch.tensor([tr[b]["class_id"] for _,b in pairs])),1).numpy()
                else:
                    seqs = []
                    for k, pr in enumerate(pairs):
                        hist[name].setdefault(pr, []).append(vecs[k]); h = hist[name][pr][-WINDOW:]
                        seqs.append(np.stack([np.zeros(1054,np.float32)]*(WINDOW-len(h)) + h))
                    P = torch.softmax(model(torch.tensor(np.stack(seqs)))[:,-1,:],1).numpy()
            nb = P.copy(); nb[:,0] = -1; best_p = nb.argmax(1); score = nb.max(1)       # one predicate per pair ("with constraint")
            order = np.argsort(-score)
            for K in KS:
                top = [(pairs[i], int(best_p[i])) for i in order[:K]]
                hit_det, hit_cls, hit_pred = set(), set(), set()
                for (a,b), p in top:
                    ga, gb = t2g.get(a), t2g.get(b)
                    if ga is None or gb is None: continue
                    for gi, (gs, gp, go) in enumerate(gts):
                        if gp == p and ga[0] == gs and gb[0] == go:
                            hit_pred.add(gi)
                            if ga[1] and gb[1]: hit_det.add(gi)
                cov_idx = {gts.index(x) for x in covered}
                for pc in {x[1] for x in gts}:
                    idx_all = [i for i,x in enumerate(gts) if x[1] == pc]
                    idx_cov = [i for i in idx_all if i in cov_idx]
                    acc[name]["SGDET"][K][pc].append(len([i for i in idx_all if i in hit_det]) / len(idx_all))
                    if idx_cov:
                        acc[name]["SGCLS"][K][pc].append(len([i for i in idx_cov if i in hit_det]) / len(idx_cov))
                        acc[name]["PredCLS"][K][pc].append(len([i for i in idx_cov if i in hit_pred]) / len(idx_cov))
    print(f"[{vi}/{len(hold)}] {vid}", flush=True)
out = {"n_videos": len(hold), "n_frames_evaluated": nframes, "models": {}}
for name in MODELS:
    out["models"][name] = {t: {f"mR@{K}": round(100*float(np.mean([np.mean(v) for v in acc[name][t][K].values()])), 2) if acc[name][t][K] else 0.0 for K in KS} for t in TASKS}
json.dump(out, open(ROOT/"logs/paper_comparison_ag_style.json","w"), indent=2)
print(json.dumps({k:v for k,v in out.items() if k!="models"}))
for name,d in out["models"].items():
    print(f"{name:<38}" + " | ".join(f"{t} " + "/".join(f"{d[t][f'mR@{K}']:5.2f}" for K in KS) for t in TASKS))
