"""Fine-tune the already-trained YOLOv8s detector on old + new VidOR data (cv_work/data/yolo_all)."""
import os, argparse
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr0", type=float, default=0.003)
ap.add_argument("--name", default="finetune")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

root = Path(__file__).resolve().parent.parent
if a.resume:
    model = YOLO(str(root / "models/detection_project" / a.name / "weights/last.pt"))
    model.train(resume=True)
else:
    model = YOLO(str(root / "models/detection/best_colab_run3.pt"))
    model.train(data=str(root / "data/data_all.yaml"), epochs=a.epochs, imgsz=640, batch=a.batch,
                lr0=a.lr0, lrf=0.1, warmup_epochs=0.5, patience=5, cache=False, workers=4, device="mps",
                fraction=a.fraction, project=str(root / "models/detection_project"), name=a.name,
                exist_ok=True, val=True, plots=True, verbose=True)
