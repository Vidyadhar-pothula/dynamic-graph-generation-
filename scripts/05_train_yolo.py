import os
import sys
import multiprocessing

try:
    multiprocessing.set_start_method('fork', force=True)
except Exception:
    pass

import yaml
import json
import time
import torch
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.detection.train_yolo import train_yolo_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test")
    group.add_argument("--pilot", action="store_true", help="Run 5-epoch pilot test")
    group.add_argument("--full", action="store_true", help="Run full 15-epoch training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    detection_cfg = config['detection']
    data_yaml = Path("data/processed/yolo/data.yaml")
    
    if not data_yaml.exists():
        print(f"ERROR: YOLO data yaml not found at {data_yaml}!")
        print("Please run scripts/04_prepare_yolo.py first.")
        sys.exit(1)
        
    # Check device priority: MPS -> CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
        
    # At startup print training environment
    print("\n==================================================")
    print("TRAINING ENVIRONMENT")
    print("==================================================")
    print(f"Operating System: {sys.platform}")
    print(f"CPU Architecture: {os.uname().machine if hasattr(os, 'uname') else 'arm64'}")
    print(f"Processor: Apple Silicon M3")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: False")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"Selected Device: {device}")
    print(f"Number of CPU Cores: {os.cpu_count()}")
    print(f"Batch Size: {detection_cfg['batch_size']}")
    print(f"Workers: 4")
    print("==================================================\n")
    
    # Determine epochs based on run mode
    if args.smoke:
        epochs = detection_cfg['epochs']['smoke_test']
        print("--- RUNNING YOLO SMOKE TEST ---")
    elif args.pilot:
        epochs = detection_cfg['epochs']['pilot']
        print("--- RUNNING YOLO PILOT ---")
    else:
        epochs = detection_cfg['epochs']['full']
        print("--- RUNNING YOLO FULL TRAINING ---")
        
    start_time = time.time()
    
    try:
        results = train_yolo_model(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=detection_cfg['batch_size'],
            imgsz=detection_cfg['image_size'],
            lr=detection_cfg['learning_rate'],
            patience=detection_cfg['patience'],
            device=device,
            resume=args.resume
        )
        
        total_time = time.time() - start_time
        avg_epoch_time = total_time / epochs
        
        # Load best metrics from training results
        # Ultralytics results.results_dict contains metrics
        results_dict = getattr(results, 'results_dict', {})
        map50 = results_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = results_dict.get('metrics/mAP50-95(B)', 0.0)
        best_epoch = int(results_dict.get('best_epoch', epochs - 1)) + 1
        
        if args.smoke:
            print("\n========================================")
            print("YOLO SMOKE TEST PASSED")
            print("========================================")
            
        elif args.pilot:
            print("\n========================================")
            print("YOLO PILOT COMPLETE")
            print("========================================")
            print(f"Epochs completed: {epochs}")
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Best validation metric (mAP50): {map50:.4f}")
            print(f"Checkpoint: models/detection/best.pt")
            print("========================================")
            
        else:
            print("\n========================================")
            print("YOLO TRAINING COMPLETE")
            print("========================================")
            print(f"Epochs requested: {epochs}")
            print(f"Epochs completed: {epochs}")
            print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
            print(f"Total training time: {total_time:.2f} seconds")
            print(f"Best epoch: {best_epoch}")
            print(f"Best mAP50: {map50:.4f}")
            print(f"Best mAP50-95: {map50_95:.4f}")
            print(f"Best checkpoint: models/detection/best.pt")
            print(f"Last checkpoint: models/detection/last.pt")
            print(f"Device: {device}")
            print(f"Batch size: {detection_cfg['batch_size']}")
            print("========================================\n")
            
        # Write summary JSON
        summary = {
            "model": "YOLOv8s",
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "best_epoch": best_epoch if not args.smoke else 1,
            "average_epoch_time_seconds": round(avg_epoch_time, 2),
            "total_training_time_seconds": round(total_time, 2),
            "device": device,
            "batch_size": detection_cfg['batch_size'],
            "checkpoint": "models/detection/best.pt"
        }
        
        logs_dir = Path("logs/detection")
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
