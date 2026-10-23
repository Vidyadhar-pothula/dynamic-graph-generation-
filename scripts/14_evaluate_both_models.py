import os
import sys
import shutil
import yaml
import json
from pathlib import Path
from ultralytics import YOLO

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataset.annotation_utils import bbox_to_yolo

def build_pilot_validation_dataset(output_dir, val_images_source_dir, proc_ann_val_dir):
    """
    Creates a temporary 11-class validation split specifically for the pilot model.
    """
    print("\nBuilding custom validation dataset for the Pilot Model (11 classes)...")
    pilot_category_mapping = {
        'adult': 0, 'baby': 1, 'bottle': 2, 'car': 3, 'chair': 4,
        'dog': 5, 'guitar': 6, 'sofa': 7, 'stingray': 8, 'table': 9, 'toy': 10
    }
    
    pilot_yolo_dir = Path(output_dir) / "pilot_yolo_data"
    img_dest = pilot_yolo_dir / "images" / "val"
    lbl_dest = pilot_yolo_dir / "labels" / "val"
    
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    # 1. Find the first 2 validation videos from parsed annotations (same as pilot training)
    proc_ann_val = Path(proc_ann_val_dir)
    val_json_files = []
    for root, dirs, files in os.walk(proc_ann_val):
        for f in files:
            if f.endswith('.json'):
                val_json_files.append(Path(root) / f)
                
    # Sort to ensure deterministic order (first 2 videos)
    val_json_files = sorted(val_json_files)[:2]
    print(f"Using pilot validation videos: {[f.name for f in val_json_files]}")
    
    # 2. Extract and format validation labels using pilot_category_mapping
    for ann_file in val_json_files:
        video_id = ann_file.stem
        
        # Load parsed annotation
        with open(ann_file, 'r') as json_f:
            ann_data = json.load(json_f)
            
        width = ann_data["width"]
        height = ann_data["height"]
        objects = ann_data["objects"]
        trajectories = ann_data["trajectories"]
        
        # Walk through existing images in full yolo split to locate frames belonging to this video
        full_val_images_dir = Path(val_images_source_dir)
        matching_images = sorted(list(full_val_images_dir.glob(f"{video_id}_frame_*.jpg")))
        
        for frame_file in matching_images:
            # Extract frame ID from filename (e.g. "video_id_frame_000120.jpg")
            try:
                frame_id = int(frame_file.stem.split("_frame_")[1])
            except Exception:
                continue
                
            lines = []
            for tid, bbox_list in trajectories.items():
                bbox_for_frame = None
                for item in bbox_list:
                    if item["frame_id"] == frame_id:
                        bbox_for_frame = item["bbox"]
                        break
                        
                if bbox_for_frame is not None:
                    category = objects[tid]
                    # Only map if the class exists in the pilot categories
                    if category in pilot_category_mapping:
                        class_id = pilot_category_mapping[category]
                        yolo_box = bbox_to_yolo(bbox_for_frame, width, height)
                        lines.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                        
            # Save label file if we have predictions
            label_name = f"{video_id}_frame_{frame_id:06d}.txt"
            with open(lbl_dest / label_name, 'w') as lbl_f:
                lbl_f.write("\n".join(lines))
                
            # Copy corresponding image frame
            shutil.copy(frame_file, img_dest / f"{video_id}_frame_{frame_id:06d}.jpg")
            
    # 3. Create pilot_data.yaml
    pilot_data_yaml = pilot_yolo_dir / "pilot_data.yaml"
    pilot_yaml_content = {
        "path": str(pilot_yolo_dir.resolve()),
        "train": "images/val",
        "val": "images/val",
        "names": {i: cat for cat, i in pilot_category_mapping.items()}
    }
    
    with open(pilot_data_yaml, "w") as f:
        yaml.safe_dump(pilot_yaml_content, f, default_flow_style=False)
        
    print(f"Pilot validation split ready at: {pilot_yolo_dir}")
    return str(pilot_data_yaml.resolve())

def main():
    config_path = "configs/config.yaml"
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    full_data_yaml = "data/processed/yolo/data.yaml"
    
    pilot_src = "runs/detect/models/detection_project/yolo_train/weights/best.pt"
    full_src = "models/detection/best.pt"
    
    output_dir = Path("data/validation_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pilot_dest_model = output_dir / "yolov8_pilot_5epochs.pt"
    full_dest_model = output_dir / "yolov8_full_15epochs.pt"
    
    pilot_folder = output_dir / "pilot_5epochs"
    full_folder = output_dir / "full_15epochs"
    
    pilot_folder.mkdir(parents=True, exist_ok=True)
    full_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy model checkpoints to data/validation_output
    if Path(pilot_src).exists():
        print(f"Saving pilot model weights to {pilot_dest_model}...")
        shutil.copy(pilot_src, pilot_dest_model)
    else:
        print(f"WARNING: Pilot checkpoint not found at {pilot_src}")
        
    if Path(full_src).exists():
        print(f"Saving full 15-epoch model weights to {full_dest_model}...")
        shutil.copy(full_src, full_dest_model)
    else:
        print(f"WARNING: Full checkpoint not found at {full_src}")
        
    # 2. Build pilot-specific validation dataset to prevent IndexError
    pilot_data_yaml_path = build_pilot_validation_dataset(
        output_dir=output_dir,
        val_images_source_dir="data/processed/yolo/images/val",
        proc_ann_val_dir="data/processed/annotations/val"
    )
    
    # 3. Evaluate Pilot Model (5 epochs)
    if pilot_dest_model.exists():
        print("\n>>> Evaluating Pilot Model (5 epochs) on its 11-class subset...")
        model_pilot = YOLO(str(pilot_dest_model))
        
        # Run val
        model_pilot.val(
            data=pilot_data_yaml_path,
            device="mps",
            project=str(pilot_folder),
            name="val_run",
            exist_ok=True,
            plots=True,
            verbose=False
        )
        
        # Clean up output plots directory structure
        val_run_dir = pilot_folder / "val_run"
        if val_run_dir.exists():
            for f in val_run_dir.glob("*.*"):
                shutil.copy(f, pilot_folder / f.name)
            shutil.rmtree(val_run_dir)
            
        # Get 20 representative image frames from the pilot val folder to predict on
        pilot_val_images = sorted(list(Path(output_dir / "pilot_yolo_data" / "images" / "val").glob("*.jpg")))[:20]
        pilot_val_image_paths = [str(p) for p in pilot_val_images]
        
        print(f"Generating predictions for {len(pilot_val_image_paths)} validation images using Pilot model...")
        model_pilot.predict(
            source=pilot_val_image_paths,
            save=True,
            project=str(pilot_folder),
            name="predictions",
            device="mps",
            exist_ok=True
        )
        
    # 4. Evaluate Full Model (15 epochs)
    if full_dest_model.exists():
        print("\n>>> Evaluating Full Model (15 epochs) on the full validation split...")
        model_full = YOLO(str(full_dest_model))
        
        # Run val
        model_full.val(
            data=full_data_yaml,
            device="mps",
            project=str(full_folder),
            name="val_run",
            exist_ok=True,
            plots=True,
            verbose=False
        )
        
        # Clean up output plots directory structure
        val_run_dir = full_folder / "val_run"
        if val_run_dir.exists():
            for f in val_run_dir.glob("*.*"):
                shutil.copy(f, full_folder / f.name)
            shutil.rmtree(val_run_dir)
            
        # Save evaluation summary under data/validation_output with 50percent_dataset naming
        full_results = model_full.val(data=full_data_yaml, device="mps", verbose=False)
        results_dict = getattr(full_results, 'results_dict', {})
        
        eval_summary = {
            "dataset_split": "50percent_dataset_1000videos",
            "epochs": 10,
            "training_videos": 1000,
            "validation_videos": 100,
            "precision": round(float(results_dict.get('metrics/precision(B)', 0.0)), 4),
            "recall": round(float(results_dict.get('metrics/recall(B)', 0.0)), 4),
            "mAP50": round(float(results_dict.get('metrics/mAP50(B)', 0.0)), 4),
            "mAP50_95": round(float(results_dict.get('metrics/mAP50-95(B)', 0.0)), 4),
            "eval_output_directory": str(output_dir.resolve())
        }
        
        with open(output_dir / "50percent_dataset_eval_results.json", "w") as f:
            json.dump(eval_summary, f, indent=4)
            
        md_summary = f"""# 50% Dataset Evaluation Results (1,000 Videos, 10 Epochs)

- **Dataset Split**: 1,000 Training Videos (250 per part across 4 ZIP archives)
- **Validation Split**: 100 Validation Videos
- **Epochs**: 10 Epochs

## Detection Metrics
- **Precision**: {eval_summary['precision']} ({eval_summary['precision']*100:.2f}%)
- **Recall**: {eval_summary['recall']} ({eval_summary['recall']*100:.2f}%)
- **mAP50**: {eval_summary['mAP50']} ({eval_summary['mAP50']*100:.2f}%)
- **mAP50-95**: {eval_summary['mAP50_95']} ({eval_summary['mAP50_95']*100:.2f}%)
"""
        with open(output_dir / "50percent_dataset_summary.md", "w") as f:
            f.write(md_summary)
            
        print(f"\nSaved evaluation results to {output_dir / '50percent_dataset_eval_results.json'}")
        
    print("\nValidation runs and comparisons completed successfully!")

if __name__ == "__main__":
    main()

