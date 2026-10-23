import os
import time
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

def on_pretrain_routine_start(trainer):
    trainer.args.workers = 4
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def on_train_epoch_start(trainer):
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    trainer.epoch_start_time = time.time()
    if not hasattr(trainer, 'train_start_time'):
        trainer.train_start_time = time.time()


def on_train_epoch_end(trainer):
    epoch_time = time.time() - trainer.epoch_start_time
    total_elapsed = time.time() - trainer.train_start_time
    
    epoch_idx = trainer.epoch + 1
    total_epochs = trainer.epochs
    
    # Sum up loss items (can be a dict of values, list, or single number)
    if trainer.loss_items is None:
        loss_val = 0.0
    elif isinstance(trainer.loss_items, dict):
        loss_val = float(sum(trainer.loss_items.values()))
    elif isinstance(trainer.loss_items, (list, tuple)):
        loss_val = float(sum(trainer.loss_items))
    else:
        loss_val = float(trainer.loss_items)
    
    # Retrieve metrics from trainer
    metrics_dict = getattr(trainer, 'metrics', {})
    precision = metrics_dict.get('metrics/precision(B)', 0.0)
    recall = metrics_dict.get('metrics/recall(B)', 0.0)
    map50 = metrics_dict.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics_dict.get('metrics/mAP50-95(B)', 0.0)
    box_loss = metrics_dict.get('val/box_loss', 0.0)
    
    # Time estimation
    epochs_remaining = total_epochs - epoch_idx
    avg_epoch_time = total_elapsed / epoch_idx
    est_remaining = avg_epoch_time * epochs_remaining
    
    print(f"\n========================================")
    print(f"EPOCH [{epoch_idx}/{total_epochs}] METRICS & PROGRESS")
    print(f"========================================")
    print(f"Train Loss    : {loss_val:.4f}")
    print(f"Val Box Loss  : {box_loss:.4f}")
    print(f"Precision     : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall        : {recall:.4f} ({recall*100:.2f}%)")
    print(f"mAP50         : {map50:.4f} ({map50*100:.2f}%)")
    print(f"mAP50-95      : {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"Epoch Time    : {format_time(epoch_time)}")
    print(f"Elapsed Time  : {format_time(total_elapsed)}")
    print(f"Est. Remaining: {format_time(est_remaining)}")
    print(f"========================================\n")


def train_yolo_model(data_yaml, epochs, batch_size, imgsz, lr, patience, device="mps", resume=False):
    current_batch = batch_size
    while current_batch >= 1:
        try:
            print(f"Starting YOLOv8 training with batch size = {current_batch}...")
            model_to_load = 'yolov8s.pt'
            
            # Check for existing checkpoint to resume
            last_path = Path("models/detection_project/yolo_train/weights/last.pt")
            if not last_path.exists():
                last_path = Path("models/detection/last.pt")
                
            if resume and last_path.exists():
                model_to_load = str(last_path)
                print(f"Resuming YOLOv8 training from checkpoint: {model_to_load}")
                
            model = YOLO(model_to_load)
            
            # Register callbacks
            model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            # Train model
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                device=device,
                batch=current_batch,
                patience=patience,
                lr0=lr,
                cache='disk',
                workers=4,
                project=str(Path("models/detection_project").resolve()),
                name="yolo_train",
                exist_ok=True,
                resume=resume,
                val=True,
                verbose=True # Display progress bars and visualizations in terminal
            )
            
            # Copy checkpoints to target models/detection directory
            best_ckpt = Path("models/detection_project/yolo_train/weights/best.pt")
            last_ckpt = Path("models/detection_project/yolo_train/weights/last.pt")
            
            models_dir = Path("models/detection")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            if best_ckpt.exists():
                shutil.copy(best_ckpt, models_dir / "best.pt")
            if last_ckpt.exists():
                shutil.copy(last_ckpt, models_dir / "last.pt")
                
            return results
            
        except Exception as e:
            error_str = str(e)
            # Catch out of memory errors
            is_oom = any(phrase in error_str.lower() for phrase in [
                "out of memory", "allocator", "mps", "cuda", "memory allocation"
            ])
            
            if is_oom:
                print(f"MPS memory issue detected: {error_str}")
                new_batch = current_batch // 2
                if new_batch < 1:
                    print("Batch size cannot be reduced further. Training failed.")
                    raise e
                print(f"Reducing batch size from {current_batch} to {new_batch}.")
                current_batch = new_batch
            else:
                raise e
