#!/usr/bin/env python3
"""Live progress dashboard for the VidOR 78-class pipeline. Ctrl-C closes the view; jobs keep running."""
import re, time, os, json
from pathlib import Path

ROOT = Path.home() / "Downloads/cv_work"
LOG = ROOT / "logs"
W = 34


def bar(frac, width=W):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def read(p):
    try:
        return Path(p).read_text(errors="ignore")
    except Exception:
        return ""


def fmt(sec):
    sec = int(sec)
    return f"{sec//3600}h{(sec%3600)//60:02d}m" if sec >= 3600 else f"{sec//60}m{sec%60:02d}s"


def row(name, frac, detail, state):
    mark = {"done": "✔", "run": "▶", "wait": "·"}[state]
    print(f" {mark} {name:<34} {bar(frac)} {frac*100:5.1f}%  {detail}")


start_track = None
while True:
    os.system("clear")
    print("VidOR 78-class pipeline - live progress   (Ctrl-C closes this view, jobs keep running)\n")

    # 1-3 finished stages
    row("1 New data: frames + YOLO dataset", 1, "27,311 new frames, 60,588 train images", "done")
    res = read(ROOT / "models/detection_project/finetune/results.csv").strip().splitlines()
    last = res[-1].split(",") if len(res) > 1 else None
    row("2 YOLO fine-tune (3 epochs)", 1, f"final mAP50 {float(last[7]):.4f}  mAP50-95 {float(last[8]):.4f}" if last else "", "done")

    # 3 tracking
    t = read(LOG / "06_tracking.log")
    nums = re.findall(r"\[(\d+)/(\d+)\] TRACK", t)
    secs = [float(x) for x in re.findall(r"\(([\d.]+)s\)", t)]
    chain = read(LOG / "data_chain.log")
    if nums:
        n, tot = int(nums[-1][0]), int(nums[-1][1])
        done_track = "tracking done" in chain
        avg = sum(secs) / len(secs) if secs else 0
        eta = (tot - n) * avg
        row("3 Tracking 375 new videos", 1 if done_track else n / tot,
            "complete" if done_track else f"{n}/{tot} videos, ~{fmt(eta)} left", "done" if done_track else "run")
    else:
        row("3 Tracking 375 new videos", 0, "not started", "wait")

    # 4 features / relation / temporal data
    steps = ["features done", "relation data done", "temporal data done"]
    got = sum(s in chain for s in steps)
    if got == 3:
        row("4 Features, relation, temporal data", 1, "complete", "done")
    elif "tracking done" in chain:
        names = ["extracting features", "building relation pairs", "building temporal sequences"]
        row("4 Features, relation, temporal data", got / 3, names[got], "run")
    else:
        row("4 Features, relation, temporal data", 0, "waiting for tracking", "wait")

    # 5 relation + temporal fine-tune
    f = read(LOG / "finetune_rel_temp.log")
    stages = [("relation", "Relation transformer"), ("lstm", "Temporal LSTM"), ("gru", "Temporal GRU")]
    started = "starting relation" in f
    for key, label in stages:
        eps = re.findall(rf"\[{key}\] epoch (\d+)/(\d+)", f)
        stopped = f"[{key}] early stop" in f
        finished = bool(re.search(rf"{key.upper()} fine-tuned|RELATION fine-tuned", f)) if key != "relation" else "RELATION fine-tuned" in f
        if eps:
            e, tot = int(eps[-1][0]), int(eps[-1][1])
            frac = 1 if (finished or stopped) else e / tot
            row(f"5 {label} fine-tune", frac, f"epoch {e}/{tot}" + (" (early stop)" if stopped else ""), "done" if finished else "run")
        else:
            row(f"5 {label} fine-tune", 0, "waiting" if not started else "starting", "wait")

    # overall
    parts = [1, 1]
    parts.append(1 if "tracking done" in chain else (int(nums[-1][0]) / int(nums[-1][1]) if nums else 0))
    parts.append(got / 3)
    fin = 0
    for key, _ in stages:
        eps = re.findall(rf"\[{key}\] epoch (\d+)/(\d+)", f)
        if eps:
            fin += (1 if (f"[{key}] early stop" in f or (key == 'relation' and 'RELATION fine-tuned' in f) or (key != 'relation' and f'{key.upper()} fine-tuned' in f)) else int(eps[-1][0]) / int(eps[-1][1])) / 3
    parts.append(fin)
    # weights: tracking is the big remaining chunk
    w = [0.2, 0.3, 0.3, 0.05, 0.15]
    overall = sum(p * x for p, x in zip(parts, w))
    print("\n" + "-" * 78)
    print(f" OVERALL  {bar(overall, 46)} {overall*100:5.1f}%")
    print(f"\n updated {time.strftime('%H:%M:%S')}")
    time.sleep(3)
