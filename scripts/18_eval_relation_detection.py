"""Proxy of the base paper's VidOR metric: video relation detection, mean Recall@K (mR@K) over predicate classes.

A ground-truth relation <subject, predicate, object, [begin,end]> counts as recalled if, among the video's top-K
predicted relation instances, one has: subject and object tracks that match the GT subject/object boxes (IoU >= 0.5 on the
frames they share) AND the same object categories, the same predicate, and temporal IoU >= 0.5 (also reported with any overlap).
Run on the 38 held-out new videos. NOT the paper's exact protocol (no I3D / Deep SORT, 38 videos instead of the 835-video val set)."""
import sys, json, pickle, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.relations.relation_transformer import RelationTransformer
from src.temporal.temporal_gru import TemporalGRU
from src.temporal.temporal_lstm import TemporalLSTM
from src.features.feature_fusion import fuse_features
MAX_OBJ, WINDOW, KS = 12, 10, (50, 100)
vocab = json.load(open(ROOT/"data/processed/relation_vocabulary.json")); id2p = {v:k for k,v in vocab.items()}; N = len(vocab)
cat = json.load(open(ROOT/"data/processed/category_mapping.json")); id2cls = {v:k for k,v in cat.items()}
def rel(path):
    m = RelationTransformer(num_object_classes=78, num_relation_classes=N); m.load_state_dict(torch.load(ROOT/path, map_location="cpu")); return m.eval()
def tmp(cls, path):
    m = cls(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=N); m.load_state_dict(torch.load(ROOT/path, map_location="cpu")); return m.eval()
MODELS = {
 "Relation v1 (loss-selected)": ("rel", rel("models/relation/best_relation_model_78class.pt")),
 "Relation v2 (class-balanced)": ("rel", rel("models/relation/best_relation_model_78class_v2_balanced.pt")),
 "Temporal LSTM v1": ("tmp", tmp(TemporalLSTM, "models/temporal/best_temporal_lstm_model_finetuned.pt")),
 "Temporal LSTM v2": ("tmp", tmp(TemporalLSTM, "models/temporal/best_temporal_lstm_model_v2_balanced.pt")),
 "Temporal GRU v1": ("tmp", tmp(TemporalGRU, "models/temporal/best_temporal_gru_model_finetuned.pt")),
 "Temporal GRU v2": ("tmp", tmp(TemporalGRU, "models/temporal/best_temporal_gru_model_v2_balanced.pt")),
}
new_rel = pickle.load(open(ROOT/"data/processed/relation_train.pkl","rb")); vids = sorted({s["video_id"] for s in new_rel}); hold = [v for i,v in enumerate(vids) if i%10==0]
feat_path = {p.stem: p for p in (ROOT/"data/processed/features/train").glob("*/*.pkl")}
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); i=max(0,x2-x1)*max(0,y2-y1)
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i; return i/u if u>0 else 0
def tiou(a,b):
    i=max(0,min(a[1],b[1])-max(a[0],b[0])+1); u=(a[1]-a[0]+1)+(b[1]-b[0]+1)-i; return i/u if u>0 else 0

stats = {m: {"tiou": collections.defaultdict(lambda: collections.defaultdict(list)), "any": collections.defaultdict(lambda: collections.defaultdict(list)),
             "hit_t": collections.Counter(), "hit_a": collections.Counter()} for m in MODELS}
gt_total = 0; gt_trackable = 0; gt_pred_count = collections.Counter()
for vi, vid in enumerate(hold, 1):
    ff = pickle.load(open(feat_path[vid],"rb"))["frame_features"]; fids = sorted(ff)
    ann = json.load(open(ROOT/f"data/processed/annotations/train/{feat_path[vid].parent.name}/{vid}.json"))
    gt_cls = {int(k): v for k,v in ann["objects"].items()}
    traj = {int(k): {p["frame_id"]: p["bbox"] for p in v} for k,v in ann["trajectories"].items()}
    # track -> GT object (vote by IoU>=0.5 and same category)
    votes = collections.defaultdict(collections.Counter)
    for fid in fids:
        for tid, info in ff[fid].items():
            for g, tr in traj.items():
                if fid in tr and gt_cls[g] == id2cls[int(info["class_id"])] and iou(info["bbox"], tr[fid]) >= 0.5: votes[tid][g] += 1
    t2g = {t: c.most_common(1)[0][0] for t, c in votes.items()}
    g2t = collections.defaultdict(list)
    for t, g in t2g.items(): g2t[g].append(t)
    gts = [r for r in ann["relations"]]
    gt_total += len(gts)
    # predictions per model: avg prob and interval per (a,b)
    per_model = {}
    for name, (kind, model) in MODELS.items():
        acc = {}; hist = {}
        for fid in fids:
            tr = ff[fid]; tids = sorted(tr, key=lambda t: -tr[t]["confidence"])[:MAX_OBJ]
            pairs = [(a,b) for a in tids for b in tids if a != b]
            if not pairs: continue
            vecs = np.stack([np.concatenate([fuse_features(tr[a]["appearance"],tr[a]["spatial"],tr[a]["motion"]), fuse_features(tr[b]["appearance"],tr[b]["spatial"],tr[b]["motion"])]) for a,b in pairs]).astype(np.float32)
            with torch.no_grad():
                if kind == "rel":
                    P = torch.softmax(model(torch.tensor(vecs[:,:527]), torch.tensor(vecs[:,527:]), torch.tensor([tr[a]["class_id"] for a,_ in pairs]), torch.tensor([tr[b]["class_id"] for _,b in pairs])),1).numpy()
                else:
                    seqs=[]
                    for k,pr in enumerate(pairs):
                        hist.setdefault(pr,[]).append(vecs[k]); h=hist[pr][-WINDOW:]; seqs.append(np.stack([np.zeros(1054,np.float32)]*(WINDOW-len(h))+h))
                    P = torch.softmax(model(torch.tensor(np.stack(seqs)))[:,-1,:],1).numpy()
            for k,pr in enumerate(pairs):
                a_ = acc.setdefault(pr, {"sum": np.zeros(N), "n": 0, "first": fid, "last": fid, "top": collections.defaultdict(list)})
                a_["sum"] += P[k]; a_["n"] += 1; a_["last"] = fid
                nb = P[k].copy(); nb[0] = -1; a_["top"][int(nb.argmax())].append(fid)
        inst = []
        for (a,b), a_ in acc.items():
            avg = a_["sum"]/a_["n"]
            for p in range(1,N):
                fr = a_["top"].get(p); iv = (min(fr),max(fr)) if fr else (a_["first"],a_["last"])
                inst.append((float(avg[p]), a, b, p, iv))
        inst.sort(key=lambda x: -x[0]); per_model[name] = inst
    # evaluate
    for gi, r in enumerate(gts):
        sub, obj, pr = r["subject_id"], r["object_id"], vocab.get(r["predicate"])
        gt_pred_count[r["predicate"]] += 1
        if pr is None: continue
        if sub in g2t and obj in g2t: gt_trackable += 1
    present = collections.Counter(r["predicate"] for r in gts)
    for name in MODELS:
        inst = per_model[name]
        for K in KS:
            top = inst[:K]
            for pname in present:
                tot = present[pname]; ht = ha = 0
                for r in gts:
                    if r["predicate"] != pname: continue
                    p = vocab.get(pname)
                    if p is None: continue
                    for sc,a,b,pp,iv in top:
                        if pp==p and t2g.get(a)==r["subject_id"] and t2g.get(b)==r["object_id"]:
                            ti = tiou(iv,(r["begin_fid"],r["end_fid"]))
                            if ti >= 0.5: ht += 1; break
                    for sc,a,b,pp,iv in top:
                        if pp==p and t2g.get(a)==r["subject_id"] and t2g.get(b)==r["object_id"] and tiou(iv,(r["begin_fid"],r["end_fid"]))>0:
                            ha += 1; break
                stats[name]["tiou"][K][pname].append(ht/tot); stats[name]["any"][K][pname].append(ha/tot)
    print(f"[{vi}/{len(hold)}] {vid}: {len(gts)} GT relations", flush=True)

out = {"n_videos": len(hold), "n_gt_relations": gt_total, "gt_relations_with_both_objects_tracked": gt_trackable,
       "ceiling_recall_given_tracking": round(gt_trackable/max(gt_total,1),3), "n_predicates_present": len(gt_pred_count), "models": {}}
for name in MODELS:
    d = {}
    for K in KS:
        for key,label in (("tiou","tIoU>=0.5"),("any","any_overlap")):
            per = {p: float(np.mean(v)) for p,v in stats[name][key][K].items()}
            d[f"mR@{K}_{label}"] = round(100*float(np.mean(list(per.values()))),2) if per else 0.0
            d[f"R@{K}_{label}_per_predicate"] = {p: round(100*v,1) for p,v in per.items()}
    out["models"][name] = d
json.dump(out, open(ROOT/"logs/paper_comparison_metrics.json","w"), indent=2)
print(json.dumps({"n_videos": out["n_videos"], "n_gt": out["n_gt_relations"], "ceiling": out["ceiling_recall_given_tracking"], "predicates": out["n_predicates_present"]}))
for name,d in out["models"].items(): print(f"{name:<30} mR@50 {d['mR@50_tIoU>=0.5']:5.2f}  mR@100 {d['mR@100_tIoU>=0.5']:5.2f}   | any-overlap mR@50 {d['mR@50_any_overlap']:5.2f} mR@100 {d['mR@100_any_overlap']:5.2f}")
