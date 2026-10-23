import os
import sys
import yaml
import json
import shutil
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset.annotation_utils import bbox_to_yolo

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for YOLOv8 training.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def collect_categories(ann_dir):
    """
    Scans parsed annotations to collect all unique object categories.
    """
    categories = set()
    ann_path = Path(ann_dir)
    if not ann_path.exists():
        return list(categories)
        
    for root, dirs, files in os.walk(ann_path):
        for f in files:
            if f.endswith('.json'):
                try:
                    with open(Path(root) / f, 'r') as json_f:
                        data = json.load(json_f)
                        for cat in data.get("objects", {}).values():
                            categories.add(cat)
                except Exception:
                    pass
    return sorted(list(categories))

def prepare_split(proc_ann_dir, frames_dir, yolo_dir, split, category_to_id):
    proc_ann_path = Path(proc_ann_dir)
    frames_path = Path(frames_dir)
    yolo_path = Path(yolo_dir)
    
    img_dest = yolo_path / "images" / split
    lbl_dest = yolo_path / "labels" / split
    
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    if not proc_ann_path.exists() or not frames_path.exists():
        print(f"Warning: Directories not found for split {split}")
        return 0
        
    count = 0
    # Walk through parsed annotations
    for root, dirs, files in os.walk(proc_ann_path):
        for f in files:
            if f.endswith('.json'):
                ann_file = Path(root) / f
                video_id = ann_file.stem
                
                # Get relative folder path (subfolder)
                rel_parts = ann_file.relative_to(proc_ann_path).parts
                if len(rel_parts) >= 2:
                    subfolder = rel_parts[0]
                else:
                    subfolder = ""
                    
                # Load normalized annotation
                with open(ann_file, 'r') as json_f:
                    ann_data = json.load(json_f)
                    
                width = ann_data["width"]
                height = ann_data["height"]
                objects = ann_data["objects"]
                trajectories = ann_data["trajectories"]
                
                # Locate video frames folder
                video_frame_dir = frames_path / subfolder / video_id
                if not video_frame_dir.exists():
                    continue
                    
                # We only process frames that were actually sampled and extracted
                sampled_frames = sorted(list(video_frame_dir.glob("*.jpg")))
                
                for frame_file in sampled_frames:
                    # Get frame ID from filename (e.g. "frame_000120.jpg" -> 120)
                    try:
                        frame_id = int(frame_file.stem.split("_")[1])
                    except Exception:
                        continue
                        
                    # Find which object trajectories have a bounding box in this frame
                    lines = []
                    for tid, bbox_list in trajectories.items():
                        # Find if there is a bbox for this frame_id
                        bbox_for_frame = None
                        for item in bbox_list:
                            if item["frame_id"] == frame_id:
                                bbox_for_frame = item["bbox"]
                                break
                                
                        if bbox_for_frame is not None:
                            category = objects[tid]
                            class_id = category_to_id[category]
                            # Convert bbox to YOLO
                            yolo_box = bbox_to_yolo(bbox_for_frame, width, height)
                            # format: class cx cy w h
                            lines.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                            
                    # Save labels file
                    label_name = f"{video_id}_frame_{frame_id:06d}.txt"
                    label_file = lbl_dest / label_name
                    
                    with open(label_file, 'w') as lbl_f:
                        lbl_f.write("\n".join(lines))
                        
                    # Copy frame image
                    image_name = f"{video_id}_frame_{frame_id:06d}.jpg"
                    shutil.copy(frame_file, img_dest / image_name)
                    count += 1
                    
    print(f"Processed {count} frames for split {split}")
    return count

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    data_dir = Path("data")
    proc_ann_train = data_dir / "processed" / "annotations" / "train"
    proc_ann_val = data_dir / "processed" / "annotations" / "val"
    
    frames_train = data_dir / "frames" / "train"
    frames_val = data_dir / "frames" / "val"
    
    yolo_dir = data_dir / "processed" / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect all unique categories to build map
    train_cats = collect_categories(proc_ann_train)
    val_cats = collect_categories(proc_ann_val)
    all_categories = sorted(list(set(train_cats + val_cats)))
    
    if len(all_categories) == 0:
        print("ERROR: No categories found in parsed annotations!")
        print("Please check that the annotations are parsed correctly.")
        sys.exit(1)
        
    category_to_id = {cat: i for i, cat in enumerate(all_categories)}
    
    print(f"Discovered {len(all_categories)} object categories.")
    print("Mapping:")
    for cat, idx in category_to_id.items():
        print(f"  {cat} -> class ID {idx}")
        
    # Write categories map to processed folder for reference
    with open(data_dir / "processed" / "category_mapping.json", "w") as f:
        json.dump(category_to_id, f, indent=4)
        
    # 2. Prepare train split
    print("Preparing training YOLO split...")
    prepare_split(proc_ann_train, frames_train, yolo_dir, "train", category_to_id)
    
    # 3. Prepare val split
    print("Preparing validation YOLO split...")
    prepare_split(proc_ann_val, frames_val, yolo_dir, "val", category_to_id)
    
    # 4. Write data.yaml file
    absolute_yolo_dir = str(yolo_dir.resolve())
    data_yaml_content = {
        "path": absolute_yolo_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {i: cat for i, cat in enumerate(all_categories)}
    }
    
    with open(yolo_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml_content, f, default_flow_style=False)
        
    print(f"\nYOLO data preparation complete! Config written to: {yolo_dir}/data.yaml")

if __name__ == "__main__":
    main()
