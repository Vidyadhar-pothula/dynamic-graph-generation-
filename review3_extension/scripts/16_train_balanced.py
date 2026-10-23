"""v2: retrain relation + temporal models with class-balanced loss and macro-F1 checkpoint selection.
v1 files are never overwritten; outputs get a _v2_balanced suffix."""
import sys, json, pickle, collections, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
MAIN = Path("/Users/vidyadhar/Desktop/dynamic scene graph generation /data/processed"); NEW = ROOT/"data/processed"
from src.relations.relation_transformer import RelationTransformer
from src.relations.relation_dataset import RelationDataset
from src.temporal.temporal_dataset import TemporalDataset
from src.temporal.temporal_lstm import TemporalLSTM
from src.temporal.temporal_gru import TemporalGRU
from ultralytics import YOLO
torch.manual_seed(0); np.random.seed(0)
EPOCHS, PATIENCE, LR, BATCH, WCAP = 15, 6, 1e-4, 128, 12.0
vocab = json.load(open(NEW/"relation_vocabulary.json")); id2p = {v:k for k,v in vocab.items()}; N = len(vocab)
names28 = YOLO(str(ROOT/"models/detection/best_original.pt")).names; map78 = json.load(open(NEW/"category_mapping.json"))
REMAP = {int(i): int(map78[n]) for i,n in names28.items()}
def load(p): return pickle.load(open(p,"rb"))
def remap(S):
    o=[]
    for s in S:
        s=dict(s); s["subject_class_id"]=REMAP[int(s["subject_class_id"])]; s["object_class_id"]=REMAP[int(s["object_class_id"])]; o.append(s)
    return o

def summarize(y, p):
    y=np.array(y); p=np.array(p)
    pr,rc,f1,sup = precision_recall_fscore_support(y,p,labels=list(range(N)),zero_division=0)
    present = sup>0
    pc = collections.Counter(p.tolist()); nonbg = [k for k in pc if k!=0]; tot_nonbg = sum(pc[k] for k in nonbg)
    top2 = sum(v for _,v in collections.Counter({k:pc[k] for k in nonbg}).most_common(2))/max(tot_nonbg,1)
    return dict(acc=round(float((y==p).mean()),4), macro_F1=round(float(f1[present].mean()),4), macro_R=round(float(rc[present].mean()),4),
                classes_predicted=len(pc), classes_recall_gt0=int((rc[present]>0).sum()), classes_present=int(present.sum()),
                top2_share_of_nonbg_predictions=round(float(top2),3), n=int(len(y)))

def weights_from(counts):
    c = np.array([counts.get(i,0) for i in range(N)], dtype=np.float64); c[c==0]=c.max()   # unseen -> harmless
    w = (c.max()/c)**0.5; w = np.minimum(w, WCAP); w = w/ (w*c).sum()*c.sum()               # keep average sample weight 1
    return torch.tensor(w, dtype=torch.float32)

# ---------------- data ----------------
new = load(NEW/"relation_train.pkl"); vids = sorted({s["video_id"] for s in new}); hold = {v for i,v in enumerate(vids) if i%10==0}
tr = remap(load(MAIN/"relation_train.pkl")) + [s for s in new if s["video_id"] not in hold]
va_old = remap(load(MAIN/"relation_val.pkl")); va_new = [s for s in new if s["video_id"] in hold]
cnt = collections.Counter(s["label_id"] for s in tr); W_rel = weights_from(cnt)
print("relation class weights (selected):", {id2p[k]: round(float(W_rel[k]),2) for k in (0,*[p for p,_ in cnt.most_common(3)], 20, 30)})
def mk(S): ds=RelationDataset.__new__(RelationDataset); ds.samples=S; return ds
def rel_pred(m,S):
    m.eval(); P=[]; Y=[]
    with torch.no_grad():
        for sf,of,sc,oc,lb in DataLoader(mk(S),batch_size=512):
            P+=m(sf,of,sc,oc).argmax(1).tolist(); Y+=lb.tolist()
    return Y,P
def rel_loss(m,S):
    m.eval(); crit=nn.CrossEntropyLoss(reduction="sum"); t=0.0
    with torch.no_grad():
        for sf,of,sc,oc,lb in DataLoader(mk(S),batch_size=512): t+=crit(m(sf,of,sc,oc),lb).item()
    return t/len(S)

out = {"relation": {}, "temporal": {}}
def new_model():
    m=RelationTransformer(num_object_classes=78,num_relation_classes=N); m.load_state_dict(torch.load(ROOT/"models/relation/best_relation_model_78class.pt",map_location="cpu")); return m
m1 = new_model()
for nm,S in (("old_val",va_old),("new_val",va_new)): out["relation"][f"v1_{nm}"]=summarize(*rel_pred(m1,S))
m = new_model(); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4); crit=nn.CrossEntropyLoss(weight=W_rel)
best=-1; bad=0; t0=time.time(); path=ROOT/"models/relation/best_relation_model_78class_v2_balanced.pt"
for ep in range(1,EPOCHS+1):
    m.train(); tl=0.0
    for sf,of,sc,oc,lb in DataLoader(mk(tr),batch_size=BATCH,shuffle=True):
        opt.zero_grad(); l=crit(m(sf,of,sc,oc),lb); l.backward(); opt.step(); tl+=l.item()*len(lb)
    Yo,Po=rel_pred(m,va_old); Yn,Pn=rel_pred(m,va_new); f=summarize(Yo+Yn,Po+Pn)
    print(f"[relation v2] epoch {ep}/{EPOCHS} train_wloss {tl/len(tr):.3f} | val macro-F1 {f['macro_F1']} acc {f['acc']} classes_predicted {f['classes_predicted']} | {time.time()-t0:.0f}s",flush=True)
    if f["macro_F1"]>best: best=f["macro_F1"]; bad=0; torch.save(m.state_dict(),path)
    else:
        bad+=1
        if bad>=PATIENCE: print("[relation v2] early stop on macro-F1"); break
m.load_state_dict(torch.load(path))
for nm,S in (("old_val",va_old),("new_val",va_new)): out["relation"][f"v2_{nm}"]=summarize(*rel_pred(m,S))
print("RELATION v1 vs v2 (new videos):", out["relation"]["v1_new_val"], "->", out["relation"]["v2_new_val"])

# ---------------- temporal ----------------
tn = load(NEW/"temporal_train.pkl"); ttr = load(MAIN/"temporal_train.pkl") + [s for s in tn if s["video_id"] not in hold]
tva_old = load(MAIN/"temporal_val.pkl"); tva_new = [s for s in tn if s["video_id"] in hold]
tc = collections.Counter(); 
for s in ttr:
    for l,mk_ in zip(s["labels"],s["mask"]):
        if mk_>0: tc[int(l)]+=1
W_t = weights_from(tc)
def tds(S): d=TemporalDataset.__new__(TemporalDataset); d.sequences=S; return d
def t_pred(model,S):
    model.eval(); P=[]; Y=[]
    with torch.no_grad():
        for f_,l,mk_ in DataLoader(tds(S),batch_size=256):
            lg=model(f_); idx=mk_.view(-1)>0; P+=lg.view(-1,lg.size(-1)).argmax(1)[idx].tolist(); Y+=l.view(-1)[idx].tolist()
    return Y,P
for kind,cls in (("lstm",TemporalLSTM),("gru",TemporalGRU)):
    def mkm(): 
        mm=cls(input_dim=1054,hidden_dim=256,num_layers=2,num_classes=N); mm.load_state_dict(torch.load(ROOT/f"models/temporal/best_temporal_{kind}_model_finetuned.pt",map_location="cpu")); return mm
    out["temporal"][kind]={}; t1=mkm()
    for nm,S in (("old_val",tva_old),("new_val",tva_new)): out["temporal"][kind][f"v1_{nm}"]=summarize(*t_pred(t1,S))
    mm=mkm(); opt=torch.optim.AdamW(mm.parameters(),lr=LR,weight_decay=1e-4); crit=nn.CrossEntropyLoss(weight=W_t,reduction="none")
    best=-1; bad=0; t0=time.time(); path=ROOT/f"models/temporal/best_temporal_{kind}_model_v2_balanced.pt"
    for ep in range(1,EPOCHS+1):
        mm.train()
        for f_,l,mk_ in DataLoader(tds(ttr),batch_size=BATCH,shuffle=True):
            opt.zero_grad(); lg=mm(f_); mf=mk_.view(-1)
            loss=(crit(lg.view(-1,lg.size(-1)),l.view(-1))*mf).sum()/(mf.sum()+1e-8); loss.backward(); opt.step()
        Yo,Po=t_pred(mm,tva_old); Yn,Pn=t_pred(mm,tva_new); f=summarize(Yo+Yn,Po+Pn)
        print(f"[{kind} v2] epoch {ep}/{EPOCHS} val macro-F1 {f['macro_F1']} acc {f['acc']} classes_predicted {f['classes_predicted']} | {time.time()-t0:.0f}s",flush=True)
        if f["macro_F1"]>best: best=f["macro_F1"]; bad=0; torch.save(mm.state_dict(),path)
        else:
            bad+=1
            if bad>=PATIENCE: print(f"[{kind} v2] early stop on macro-F1"); break
    mm.load_state_dict(torch.load(path))
    for nm,S in (("old_val",tva_old),("new_val",tva_new)): out["temporal"][kind][f"v2_{nm}"]=summarize(*t_pred(mm,S))
    print(f"{kind.upper()} v1 vs v2 (new videos):", out["temporal"][kind]["v1_new_val"], "->", out["temporal"][kind]["v2_new_val"])
json.dump(out, open(ROOT/"logs/relation_temporal_results_v2.json","w"), indent=2)
print("saved logs/relation_temporal_results_v2.json")
