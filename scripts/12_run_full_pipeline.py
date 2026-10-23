import subprocess
import sys
import yaml
import json
import time
from pathlib import Path

def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    elapsed = time.time() - t0
    print(f">>> Completed in {elapsed:.2f} seconds\n")

def count_detections_and_tracks():
    print("Calculating final tracking statistics...")
    data_dir = Path("data/processed/tracks")
    
    stats = {}
    for split in ["train", "val"]:
        split_dir = data_dir / split
        total_boxes = 0
        total_tracks = set()
        file_count = 0
        
        if split_dir.exists():
            for json_path in split_dir.rglob("*.json"):
                file_count += 1
                try:
                    with open(json_path, 'r') as f:
                        track_data = json.load(f)
                    tracks_list = track_data.get("tracks", [])
                    total_boxes += len(tracks_list)
                    for t in tracks_list:
                        total_tracks.add((json_path.stem, t.get("track_id")))
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                    
        stats[split] = {
            "videos": file_count,
            "detections": total_boxes,
            "trajectories": len(total_tracks)
        }
        
    print("\n==================================================")
    print("DETECTION & TRACKING SUMMARY STATISTICS")
    print("==================================================")
    for split, data in stats.items():
        print(f"{split.upper()} SPLIT:")
        print(f"  Processed Videos: {data['videos']}")
        print(f"  Total Object Detections (Boxes): {data['detections']}")
        print(f"  Unique Trajectories (Tracks): {data['trajectories']}")
    print("==================================================\n")
    return stats

def main():
    config_path = "configs/config.yaml"
    print("Initializing final pipeline runner...")
    
    # 1. Verify YOLO training completed
    yolo_best = Path("models/detection_project/yolo_train/weights/best.pt")
    if not yolo_best.exists():
        print(f"ERROR: YOLO best.pt not found at {yolo_best}!")
        print("Please wait for YOLOv8 training to finish.")
        sys.exit(1)
        
    # 2. Copy best.pt to models/detection/best.pt
    dest_dir = Path("models/detection")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_best = dest_dir / "best.pt"
    print(f"Copying {yolo_best} to {dest_best}...")
    subprocess.run(["cp", str(yolo_best), str(dest_best)])
    
    # 3. Run downstream pipeline scripts sequentially
    # Run tracking
    run_cmd(["python3", "scripts/06_run_tracking.py", "--config", config_path])
    
    # Run feature extraction
    run_cmd(["python3", "scripts/07_extract_features.py", "--config", config_path])
    
    # Run relation preparation
    run_cmd(["python3", "scripts/08_prepare_relation_data.py", "--config", config_path])
    
    # Train Relation Transformer
    run_cmd(["python3", "scripts/09_train_relation_transformer.py", "--config", config_path, "--full", "--device", "cpu"])
    
    # Run temporal preparation
    run_cmd(["python3", "scripts/10_prepare_temporal_data.py", "--config", config_path])
    
    # Train Temporal LSTM
    run_cmd(["python3", "scripts/11_train_temporal_model.py", "--config", config_path, "--model", "lstm", "--full", "--device", "cpu"])
    
    # Train Temporal GRU
    run_cmd(["python3", "scripts/11_train_temporal_model.py", "--config", config_path, "--model", "gru", "--full", "--device", "cpu"])
    
    # 4. Count final detections and tracks
    count_detections_and_tracks()
    
    print("Master pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
