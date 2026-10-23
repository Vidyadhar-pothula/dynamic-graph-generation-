import os
import sys
import json
import shutil
import yaml
from pathlib import Path

def copy_with_progress(src, dst, label=""):
    """Copy a file and print progress."""
    shutil.copy2(src, dst)

def package_active_dataset():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "configs/config.yaml"
    
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    per_part = config['dataset'].get('per_part_limit_train', 250)
    fps = config['frame_sampling'].get('fps', 1)
    
    export_dir = root_dir / "data/dataset_export"
    print(f"\nCreating clean dataset export at: {export_dir}")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    
    # Create directory structure
    dirs = {
        "videos_train":    export_dir / "videos" / "train",
        "videos_val":      export_dir / "videos" / "val",
        "annotations_train": export_dir / "annotations" / "train",
        "annotations_val":   export_dir / "annotations" / "val",
        "yolo_images_train": export_dir / "yolo_format" / "images" / "train",
        "yolo_images_val":   export_dir / "yolo_format" / "images" / "val",
        "yolo_labels_train": export_dir / "yolo_format" / "labels" / "train",
        "yolo_labels_val":   export_dir / "yolo_format" / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────
    # 1. COPY TRAINING VIDEOS (mp4 files from data/extracted/train)
    # ─────────────────────────────────────────────────────────────
    print("\n[1/5] Copying training videos (mp4)...")
    train_extracted = root_dir / "data/extracted/train"
    train_video_count = 0
    for folder in sorted(train_extracted.iterdir()):
        if folder.is_dir():
            for mp4 in folder.glob("*.mp4"):
                dst = dirs["videos_train"] / mp4.name
                shutil.copy2(mp4, dst)
                train_video_count += 1
    print(f"   -> Copied {train_video_count} training videos.")
    
    # ─────────────────────────────────────────────────────────────
    # 2. COPY VALIDATION VIDEOS (mp4 files from data/extracted/val)
    # ─────────────────────────────────────────────────────────────
    print("\n[2/5] Copying validation videos (mp4)...")
    val_extracted = root_dir / "data/extracted/val"
    val_video_count = 0
    for folder in sorted(val_extracted.iterdir()):
        if folder.is_dir():
            for mp4 in folder.glob("*.mp4"):
                dst = dirs["videos_val"] / mp4.name
                shutil.copy2(mp4, dst)
                val_video_count += 1
    print(f"   -> Copied {val_video_count} validation videos.")
    
    # ─────────────────────────────────────────────────────────────
    # 3. COPY ANNOTATIONS (train & val from data/processed/annotations)
    # ─────────────────────────────────────────────────────────────
    print("\n[3/5] Copying annotations (JSON)...")
    proc_ann = root_dir / "data/processed/annotations"
    
    train_ann_count = 0
    train_ann_dir = proc_ann / "train"
    if train_ann_dir.exists():
        for subfolder in sorted(train_ann_dir.iterdir()):
            if subfolder.is_dir():
                for jf in subfolder.glob("*.json"):
                    dst_dir = dirs["annotations_train"] / subfolder.name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(jf, dst_dir / jf.name)
                    train_ann_count += 1
    print(f"   -> Copied {train_ann_count} training annotation JSONs.")
    
    val_ann_count = 0
    val_ann_dir = proc_ann / "val"
    if val_ann_dir.exists():
        for subfolder in sorted(val_ann_dir.iterdir()):
            if subfolder.is_dir():
                for jf in subfolder.glob("*.json"):
                    dst_dir = dirs["annotations_val"] / subfolder.name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(jf, dst_dir / jf.name)
                    val_ann_count += 1
    print(f"   -> Copied {val_ann_count} validation annotation JSONs.")
    
    # ─────────────────────────────────────────────────────────────
    # 4. COPY YOLO IMAGES & LABELS
    # ─────────────────────────────────────────────────────────────
    print("\n[4/5] Copying YOLO format images & labels...")
    yolo_dir = root_dir / "data/processed/yolo"
    
    yolo_train_count = 0
    for img in (yolo_dir / "images" / "train").glob("*.jpg"):
        shutil.copy2(img, dirs["yolo_images_train"] / img.name)
        yolo_train_count += 1
    print(f"   -> Copied {yolo_train_count} YOLO train images.")
    
    yolo_val_count = 0
    for img in (yolo_dir / "images" / "val").glob("*.jpg"):
        shutil.copy2(img, dirs["yolo_images_val"] / img.name)
        yolo_val_count += 1
    print(f"   -> Copied {yolo_val_count} YOLO val images.")
    
    label_train_count = 0
    for lbl in (yolo_dir / "labels" / "train").glob("*.txt"):
        shutil.copy2(lbl, dirs["yolo_labels_train"] / lbl.name)
        label_train_count += 1
    print(f"   -> Copied {label_train_count} YOLO train label files.")
    
    label_val_count = 0
    for lbl in (yolo_dir / "labels" / "val").glob("*.txt"):
        shutil.copy2(lbl, dirs["yolo_labels_val"] / lbl.name)
        label_val_count += 1
    print(f"   -> Copied {label_val_count} YOLO val label files.")
    
    # Copy data.yaml
    if (yolo_dir / "data.yaml").exists():
        shutil.copy2(yolo_dir / "data.yaml", export_dir / "yolo_format" / "data.yaml")
    
    # ─────────────────────────────────────────────────────────────
    # 5. WRITE DATASET MANIFEST
    # ─────────────────────────────────────────────────────────────
    print("\n[5/5] Writing dataset manifest...")
    summary = {
        "dataset_name": "VidOR Active Training Subset",
        "export_location": str(export_dir.resolve()),
        "dataset_config": {
            "videos_per_part": per_part,
            "parts_used": ["training_0001", "training_0002", "training_0003", "training_0008"],
            "sampling_fps": fps
        },
        "counts": {
            "train_videos": train_video_count,
            "val_videos": val_video_count,
            "train_annotations": train_ann_count,
            "val_annotations": val_ann_count,
            "yolo_train_images": yolo_train_count,
            "yolo_val_images": yolo_val_count,
            "yolo_train_labels": label_train_count,
            "yolo_val_labels": label_val_count
        }
    }
    with open(export_dir / "dataset_manifest.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✅ Dataset export complete at:\n   {export_dir}")
    print(f"\n   Training Videos  : {train_video_count}")
    print(f"   Validation Videos: {val_video_count}")
    print(f"   Train Annotations: {train_ann_count}")
    print(f"   Val Annotations  : {val_ann_count}")
    print(f"   YOLO Train Images: {yolo_train_count}")
    print(f"   YOLO Val Images  : {yolo_val_count}")

if __name__ == "__main__":
    package_active_dataset()
