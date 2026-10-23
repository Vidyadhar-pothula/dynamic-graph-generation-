"""Why do two predicates dominate? Prints label balance and per-class behaviour of the v1 relation model."""
import sys, json, pickle, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
MAIN = Path("/Users/vidyadhar/Desktop/dynamic scene graph generation /data/processed")
from src.relations.relation_transformer import RelationTransformer
from sklearn.metrics import precision_recall_fscore_support
vocab = json.load(open(ROOT/"data/processed/relation_vocabulary.json")); id2p = {v:k for k,v in vocab.items()}; N=len(vocab)
new = pickle.load(open(ROOT/"data/processed/relation_train.pkl","rb")); old = pickle.load(open(MAIN/"relation_train.pkl","rb"))
vids = sorted({s["video_id"] for s in new}); hold = {v for i,v in enumerate(vids) if i%10==0}
tr_new = [s for s in new if s["video_id"] not in hold]; va_new = [s for s in new if s["video_id"] in hold]
def dist(S):
    c = collections.Counter(s["label_id"] for s in S); n=sum(c.values()); return c,n
c_old,n_old = dist(old); c_new,n_new = dist(tr_new)
print(f"TRAINING LABELS  old={n_old}  new={n_new}")
print(f"  background share: old {c_old[0]/n_old:.1%}  new {c_new[0]/n_new:.1%}")
pos = collections.Counter(); 
for c in (c_old,c_new):
    for k,v in c.items():
        if k!=0: pos[k]+=v
tp = sum(pos.values()); print(f"  positive samples: {tp}  distinct predicates with >=1 sample: {len(pos)}/{N-1}")
print("  positive-label distribution (top 10):")
for k,v in pos.most_common(10): print(f"    {id2p[k]:<16}{v:>8}  {v/tp:6.1%}")
print("  rarest 8:", [(id2p[k],v) for k,v in sorted(pos.items(), key=lambda kv: kv[1])[:8]])
top2 = sum(v for k,v in pos.most_common(2))/tp; print(f"  top-2 predicates share of all positives: {top2:.1%}")

# v1 model behaviour on the new-video hold-out (remap not needed: new data is already 78-class)
m = RelationTransformer(num_object_classes=78, num_relation_classes=N); m.load_state_dict(torch.load(ROOT/"models/relation/best_relation_model_78class.pt", map_location="cpu")); m.eval()
S = va_new; sf=torch.tensor(np.stack([s["subject_features"] for s in S])); of=torch.tensor(np.stack([s["object_features"] for s in S]))
sc=torch.tensor([s["subject_class_id"] for s in S]); oc=torch.tensor([s["object_class_id"] for s in S]); y=np.array([s["label_id"] for s in S])
with torch.no_grad(): P=torch.softmax(m(sf,of,sc,oc),1).numpy()
pred=P.argmax(1)
print(f"\nV1 MODEL on {len(S)} new-video hold-out pairs")
pc=collections.Counter(pred.tolist()); print("  predicted-label distribution (top 6):", [(id2p[k], f"{v/len(S):.1%}") for k,v in pc.most_common(6)])
print(f"  classes ever predicted: {len(pc)}/{N}   classes in ground truth: {len(set(y.tolist()))}")
p,r,f,sup = precision_recall_fscore_support(y,pred,labels=list(range(N)),zero_division=0)
print("  per-class (precision / recall / F1 / support), classes with support:")
for k in np.argsort(-sup)[:12]:
    if sup[k]: print(f"    {id2p[k]:<16} P {p[k]:.2f}  R {r[k]:.2f}  F1 {f[k]:.2f}  n={int(sup[k])}")
print(f"  classes with recall > 0: {(r[sup>0]>0).sum()}/{(sup>0).sum()}")
# does the loss-based checkpoint under-train? compare to a 5-epoch-style view from the earlier log
