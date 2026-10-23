import os
import sys
import yaml
import json
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.frame_sampling import sample_video_frames

def parse_args():
    parser = argparse.ArgumentParser(description="Extract and sample frames from videos.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def process_videos(source_dir, target_dir, target_fps):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Warning: Video directory does not exist: {source_dir}")
        return {}
        
    metadata = {}
    
    for root, dirs, files in os.walk(source_path):
        for f in files:
            if f.endswith('.mp4'):
                video_file = Path(root) / f
                video_id = video_file.stem
                
                # Maintain relative path structure
                rel_parts = video_file.relative_to(source_path).parts
                if len(rel_parts) >= 2:
                    subfolder = rel_parts[0]
                else:
                    subfolder = ""
                    
                dest_dir = target_path / subfolder / video_id
                
                print(f"Extracting frames from {video_file} at {target_fps} FPS...")
                try:
                    sampled, total, orig_fps = sample_video_frames(video_file, dest_dir, target_fps)
                    metadata[video_id] = {
                        "video_path": str(video_file.resolve()),
                        "frame_dir": str(dest_dir.resolve()),
                        "total_frames": total,
                        "orig_fps": orig_fps,
                        "sampled_frames": sampled
                    }
                    print(f"  Extracted {len(sampled)}/{total} frames.")
                except Exception as e:
                    print(f"Error processing video {video_file}: {str(e)}")
                    
    return metadata

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    target_fps = config['frame_sampling']['fps']
    
    data_dir = Path("data")
    extracted_train = data_dir / "extracted" / "train"
    extracted_val = data_dir / "extracted" / "val"
    
    frames_train = data_dir / "frames" / "train"
    frames_val = data_dir / "frames" / "val"
    
    # Clean/create target directories
    frames_train.mkdir(parents=True, exist_ok=True)
    frames_val.mkdir(parents=True, exist_ok=True)
    
    print("Extracting train video frames...")
    train_meta = process_videos(extracted_train, frames_train, target_fps)
    
    print("Extracting val video frames...")
    val_meta = process_videos(extracted_val, frames_val, target_fps)
    
    # Save metadata
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    all_metadata = {
        "train": train_meta,
        "val": val_meta
    }
    
    with open(proc_dir / "frame_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=4)
        
    print("\nFrame extraction completed!")

if __name__ == "__main__":
    main()
