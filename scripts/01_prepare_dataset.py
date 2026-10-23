import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.extraction import extract_annotations, extract_videos_matching_annotations, extract_balanced_per_part_videos

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset and extract zip files.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset_config = config['dataset']
    root_dir = Path(dataset_config['root'])
    
    # Target directories
    data_dir = Path("data")
    extracted_train_dir = data_dir / "extracted" / "train"
    extracted_val_dir = data_dir / "extracted" / "val"
    ann_train_dir = data_dir / "annotations" / "train"
    ann_val_dir = data_dir / "annotations" / "val"
    
    # Ensure they exist
    extracted_train_dir.mkdir(parents=True, exist_ok=True)
    extracted_val_dir.mkdir(parents=True, exist_ok=True)
    ann_train_dir.mkdir(parents=True, exist_ok=True)
    ann_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Available disk space check
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    
    required_space_gb = 5.0
    
    print("==================================================")
    print("STORAGE CHECK")
    print("==================================================")
    print(f"Required approximate space: {required_space_gb:.2f} GB")
    print(f"Available space: {free_gb:.2f} GB")
    print(f"Estimated remaining after extraction: {free_gb - required_space_gb:.2f} GB")
    
    if free_gb < required_space_gb:
        print("\nERROR: Insufficient storage space on this machine!")
        print(f"We have {free_gb:.2f} GB but need at least {required_space_gb:.2f} GB.")
        print("Please free up some space before proceeding.")
        sys.exit(1)
        
    print("Proceed: YES")
    print("==================================================")
    
    per_part_limit = dataset_config.get("per_part_limit_train", 500)
    limit_val = dataset_config.get("limit_val", 250)
    
    # 1. Extract balanced training videos from each zip & matching annotations
    train_video_zips = [root_dir / p for p in dataset_config['train_parts']]
    train_ann_zip = root_dir / dataset_config['train_annotations'][0]
    extract_balanced_per_part_videos(
        video_zip_paths=train_video_zips,
        ann_zip_path=train_ann_zip,
        dest_video_dir=extracted_train_dir,
        dest_ann_dir=ann_train_dir,
        per_part_limit=per_part_limit
    )
    
    # 2. Extract validation annotations & videos
    val_ann_zip = root_dir / dataset_config['validation_annotations'][0]
    extract_annotations(val_ann_zip, ann_val_dir, limit=limit_val)
    val_video_zips = [root_dir / p for p in dataset_config['validation']]
    extract_videos_matching_annotations(val_video_zips, ann_val_dir, extracted_val_dir, limit=limit_val)
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
