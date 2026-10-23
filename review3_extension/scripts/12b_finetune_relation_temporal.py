"""Expand the relation model from 28 to 78 object classes (old weights copied by class name) and fine-tune
the relation transformer plus the LSTM/GRU temporal models on OLD + NEW VidOR data.

Evaluation is reported separately on the old validation set and on a hold-out of the NEW videos.
Originals in the main project folder are never touched; outputs go to cv_work/models/*.
"""
import sys, json, pickle, time, argparse, copy
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MAIN = Path("/Users/vidyadhar/Desktop/dynamic scene graph generation /data/processed")
NEW = ROOT / "data/processed"

ap = argparse.ArgumentParser()
ap.add_argument("--sanity", action="store_true", help="only check old model vs expanded model on old validation data")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
dev = args.device
torch.manual_seed(0); np.random.seed(0)

from src.relations.relation_transformer import RelationTransformer
from src.relations.relation_dataset import RelationDataset
from src.temporal.temporal_dataset import TemporalDataset
from src.temporal.temporal_lstm import TemporalLSTM
from src.temporal.temporal_gru import TemporalGRU


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------- class remap 28 -> 78 by name ----------
from ultralytics import YOLO
names28 = YOLO(str(ROOT / "models/detection/best_original.pt")).names
map78 = json.load(open(ROOT / "data/processed/category_mapping.json"))
REMAP = {}
for i, n in names28.items():
    assert n in map78, f"class {n!r} of the 28-class detector not found in the 78-class list"
    REMAP[int(i)] = int(map78[n])
print(f"28 -> 78 class remap OK ({len(REMAP)} classes), e.g. baby: {[k for k,v in names28.items() if v=='baby']} -> {map78['baby']}")

vocab = json.load(open(ROOT / "data/processed/relation_vocabulary.json"))
N_REL = len(vocab)


def remap_samples(samples):
    out = []
    for s in samples:
        s = dict(s)
        s["subject_class_id"] = REMAP[int(s["subject_class_id"])]
        s["object_class_id"] = REMAP[int(s["object_class_id"])]
        out.append(s)
    return out


def metrics_from(labels, preds):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = float(np.mean(np.array(labels) == np.array(preds))) if len(labels) else 0.0
    return dict(acc=round(acc, 4), macro_P=round(float(p), 4), macro_R=round(float(r), 4), macro_F1=round(float(f), 4), n=len(labels))


# ================= RELATION =================
def rel_eval(model, samples, bs=512):
    ds = RelationDataset.__new__(RelationDataset); ds.samples = samples
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    model.eval(); crit = nn.CrossEntropyLoss(reduction="sum"); tot = 0.0; P = []; L = []
    with torch.no_grad():
        for sf, of, sc, oc, lb in dl:
            lg = model(sf.to(dev), of.to(dev), sc.to(dev), oc.to(dev))
            tot += crit(lg, lb.to(dev)).item()
            P += torch.argmax(lg, 1).cpu().tolist(); L += lb.tolist()
    m = metrics_from(L, P); m["loss"] = round(tot / max(len(L), 1), 4)
    return m


old_state = torch.load(ROOT / "models/relation/best_relation_model.pt", map_location="cpu")
m28 = RelationTransformer(num_object_classes=28, num_relation_classes=N_REL); m28.load_state_dict(old_state); m28.to(dev)

old_rel_val_raw = load(MAIN / "relation_val.pkl")
old_rel_val = remap_samples(old_rel_val_raw)

# expanded 78-class model: copy every weight, class_embed rows copied by class name
m78 = RelationTransformer(num_object_classes=78, num_relation_classes=N_REL)
sd = {k: v.clone() for k, v in old_state.items()}
emb = m78.class_embed.weight.data.clone()
for i28, i78 in REMAP.items():
    emb[i78] = old_state["class_embed.weight"][i28]
sd["class_embed.weight"] = emb
m78.load_state_dict(sd); m78.to(dev)

a = rel_eval(m28, old_rel_val_raw)
b = rel_eval(m78, old_rel_val)
print("\nRELATION sanity (old validation set, same samples):")
print("  old 28-class model      :", a)
print("  expanded 78-class model :", b)
assert abs(a["loss"] - b["loss"]) < 1e-3 and a["acc"] == b["acc"], "expansion changed the model's behaviour!"
print("  -> identical: expansion is lossless\n")
if args.sanity:
    sys.exit(0)

# ---------- new data, hold out every 10th NEW video ----------
new_rel = load(NEW / "relation_train.pkl")
vids = sorted({s["video_id"] for s in new_rel})
hold = {v for i, v in enumerate(vids) if i % 10 == 0}
new_rel_tr = [s for s in new_rel if s["video_id"] not in hold]
new_rel_va = [s for s in new_rel if s["video_id"] in hold]
old_rel_tr = remap_samples(load(MAIN / "relation_train.pkl"))
train_rel = old_rel_tr + new_rel_tr
print(f"relation data: old train {len(old_rel_tr)} + new train {len(new_rel_tr)} = {len(train_rel)}; "
      f"val: old {len(old_rel_val)}, new hold-out {len(new_rel_va)} ({len(hold)} videos)")

results = {"relation": {}, "temporal": {}}
results["relation"]["baseline_old_val"] = b
results["relation"]["baseline_new_val"] = rel_eval(m78, new_rel_va)
print("expanded model BEFORE fine-tune on NEW-video hold-out:", results["relation"]["baseline_new_val"])


def finetune(model, train_fn, eval_fn, best_path, label):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf"); bad = 0; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl = train_fn(model, opt)
        vo, vn = eval_fn(model)
        v = (vo["loss"] * vo["n"] + vn["loss"] * vn["n"]) / max(vo["n"] + vn["n"], 1)
        print(f"[{label}] epoch {ep}/{args.epochs} train_loss {tl:.4f} | val_loss(old+new) {v:.4f} | "
              f"old acc {vo['acc']} F1 {vo['macro_F1']} | new acc {vn['acc']} F1 {vn['macro_F1']} | {time.time()-t0:.0f}s", flush=True)
        if v < best:
            best = v; bad = 0; torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{label}] early stop"); break
    model.load_state_dict(torch.load(best_path, map_location=dev))
    return eval_fn(model)


def rel_train(model, opt):
    ds = RelationDataset.__new__(RelationDataset); ds.samples = train_rel
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True); crit = nn.CrossEntropyLoss(); tot = 0.0
    model.train()
    for sf, of, sc, oc, lb in dl:
        opt.zero_grad(); loss = crit(model(sf.to(dev), of.to(dev), sc.to(dev), oc.to(dev)), lb.to(dev)); loss.backward(); opt.step()
        tot += loss.item() * sf.size(0)
    return tot / len(train_rel)


(ROOT / "models/relation").mkdir(parents=True, exist_ok=True)
rel_path = ROOT / "models/relation/best_relation_model_78class.pt"
ro, rn = finetune(m78, rel_train, lambda m: (rel_eval(m, old_rel_val), rel_eval(m, new_rel_va)), rel_path, "relation")
results["relation"]["finetuned_old_val"] = ro; results["relation"]["finetuned_new_val"] = rn
print("RELATION fine-tuned  old val:", ro, "\n                     new val:", rn)


# ================= TEMPORAL =================
def tmp_eval(model, seqs, bs=256):
    ds = TemporalDataset.__new__(TemporalDataset); ds.sequences = seqs
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    model.eval(); crit = nn.CrossEntropyLoss(reduction="none"); tot = 0.0; steps = 0.0; P = []; L = []
    with torch.no_grad():
        for f, l, mk in dl:
            lg = model(f.to(dev)); lf = l.view(-1).to(dev); mf = mk.view(-1).to(dev)
            tot += (crit(lg.view(-1, lg.size(-1)), lf) * mf).sum().item(); steps += mf.sum().item()
            pr = torch.argmax(lg.view(-1, lg.size(-1)), 1).cpu().numpy(); idx = mf.cpu().numpy() > 0
            P += pr[idx].tolist(); L += lf.cpu().numpy()[idx].tolist()
    m = metrics_from(L, P); m["loss"] = round(tot / max(steps, 1), 4); m["n"] = int(steps)
    return m


old_tmp_tr = load(MAIN / "temporal_train.pkl"); old_tmp_va = load(MAIN / "temporal_val.pkl")
new_tmp = load(NEW / "temporal_train.pkl")
new_tmp_tr = [s for s in new_tmp if s["video_id"] not in hold]; new_tmp_va = [s for s in new_tmp if s["video_id"] in hold]
train_tmp = old_tmp_tr + new_tmp_tr
print(f"\ntemporal data: old train {len(old_tmp_tr)} + new train {len(new_tmp_tr)} = {len(train_tmp)}; val: old {len(old_tmp_va)}, new hold-out {len(new_tmp_va)}")

(ROOT / "models/temporal").mkdir(parents=True, exist_ok=True)
for kind, cls in (("lstm", TemporalLSTM), ("gru", TemporalGRU)):
    model = cls(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=N_REL)
    model.load_state_dict(torch.load(ROOT / f"models/temporal/best_temporal_{kind}_model.pt", map_location="cpu")); model.to(dev)
    results["temporal"][kind] = {"baseline_old_val": tmp_eval(model, old_tmp_va), "baseline_new_val": tmp_eval(model, new_tmp_va)}
    print(f"{kind.upper()} BEFORE: old val {results['temporal'][kind]['baseline_old_val']} | new val {results['temporal'][kind]['baseline_new_val']}")

    def t_train(model, opt):
        ds = TemporalDataset.__new__(TemporalDataset); ds.sequences = train_tmp
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True); crit = nn.CrossEntropyLoss(reduction="none"); tot = 0.0; steps = 0.0
        model.train()
        for f, l, mk in dl:
            opt.zero_grad(); lg = model(f.to(dev)); mf = mk.view(-1).to(dev)
            loss = (crit(lg.view(-1, lg.size(-1)), l.view(-1).to(dev)) * mf).sum() / (mf.sum() + 1e-8)
            loss.backward(); opt.step(); tot += loss.item() * mf.sum().item(); steps += mf.sum().item()
        return tot / max(steps, 1)

    path = ROOT / f"models/temporal/best_temporal_{kind}_model_finetuned.pt"
    to, tn = finetune(model, t_train, lambda m: (tmp_eval(m, old_tmp_va), tmp_eval(m, new_tmp_va)), path, kind)
    results["temporal"][kind]["finetuned_old_val"] = to; results["temporal"][kind]["finetuned_new_val"] = tn
    print(f"{kind.upper()} fine-tuned: old val {to} | new val {tn}")

json.dump(results, open(ROOT / "logs/relation_temporal_results.json", "w"), indent=2)
print("\nSaved results to logs/relation_temporal_results.json")
