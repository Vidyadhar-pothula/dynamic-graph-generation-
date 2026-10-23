import os
import sys
import yaml
import json
import pickle
import numpy as np
import cv2
import torch
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.appearance import AppearanceExtractor
from src.features.spatial import extract_spatial_features
from src.features.motion import extract_motion_features

def parse_args():
    parser = argparse.ArgumentParser(description="Extract appearance, spatial, and motion features.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def process_features_for_split(split, tracks_dir, frames_dir, features_dest_dir, frame_metadata, appearance_extractor):
    tracks_path = Path(tracks_dir)
    frames_path = Path(frames_dir)
    dest_path = Path(features_dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if not tracks_path.exists():
        print(f"Warning: Track directory not found: {tracks_dir}")
        return
        
    split_meta = frame_metadata.get(split, {})
    
    # Recurse through track files
    for root, dirs, files in os.walk(tracks_path):
        for f in files:
            if f.endswith('.json'):
                track_file = Path(root) / f
                video_id = track_file.stem
                
                # Get subfolder
                rel_parts = track_file.relative_to(tracks_path).parts
                if len(rel_parts) >= 2:
                    subfolder = rel_parts[0]
                else:
                    subfolder = ""
                    
                target_folder = dest_path / subfolder
                target_folder.mkdir(parents=True, exist_ok=True)
                target_file = target_folder / f"{video_id}.pkl"
                
                # Load tracks
                with open(track_file, 'r') as tr_f:
                    track_data = json.load(tr_f)
                    
                tracks = track_data.get("tracks", [])
                
                # Group tracks by frame_id
                tracks_by_frame = {}
                for t in tracks:
                    fid = int(t["frame_id"])
                    if fid not in tracks_by_frame:
                        tracks_by_frame[fid] = []
                    tracks_by_frame[fid].append(t)
                    
                video_meta = split_meta.get(video_id, {})
                sampled_frames = video_meta.get("sampled_frames", [])
                
                print(f"Extracting features for video {video_id} ({len(sampled_frames)} frames)...")
                
                # We will store: frame_id -> track_id -> features dict
                frame_features = {}
                
                # Keep track of previous frame's spatial features to compute motion
                # track_id -> (prev_spatial_vector, prev_frame_id)
                prev_track_spatial = {}
                
                # Sort sampled frames by frame_id to compute sequential motion correctly
                sorted_sampled = sorted(sampled_frames, key=lambda x: int(x["frame_id"]))
                
                for sf in sorted_sampled:
                    fid = int(sf["frame_id"])
                    frame_file = frames_path / subfolder / video_id / f"frame_{fid:06d}.jpg"
                    
                    if not frame_file.exists():
                        continue
                        
                    # Load frame image
                    frame_cv = cv2.imread(str(frame_file))
                    if frame_cv is None:
                        continue
                        
                    h, w, _ = frame_cv.shape
                    
                    # Get tracked boxes for this frame
                    frame_tracks = tracks_by_frame.get(fid, [])
                    if not frame_tracks:
                        continue
                        
                    # Extract appearance features in batch
                    bboxes = [t["bbox"] for t in frame_tracks]
                    try:
                        app_feats = appearance_extractor.extract_batch_crops(frame_cv, bboxes)
                    except Exception as e:
                        print(f"Error in batch appearance extraction for frame {fid}: {str(e)}")
                        app_feats = [np.zeros(512, dtype=np.float32) for _ in bboxes]
                        
                    frame_features[fid] = {}
                    
                    for idx, t in enumerate(frame_tracks):
                        tid = int(t["track_id"])
                        bbox = t["bbox"]
                        class_id = int(t["class_id"])
                        
                        # Spatial feature
                        spatial = extract_spatial_features(bbox, w, h)
                        
                        # Motion feature
                        prev_info = prev_track_spatial.get(tid)
                        if prev_info is not None:
                            prev_spatial, prev_fid = prev_info
                            dt = fid - prev_fid
                            motion = extract_motion_features(spatial, prev_spatial, dt)
                        else:
                            motion = np.zeros(6, dtype=np.float32)
                            
                        # Update tracker state
                        prev_track_spatial[tid] = (spatial, fid)
                        
                        # Store all
                        frame_features[fid][tid] = {
                            "appearance": app_feats[idx],
                            "spatial": spatial,
                            "motion": motion,
                            "bbox": bbox,
                            "class_id": class_id,
                            "confidence": t["confidence"]
                        }
                        
                # Save to pickle file
                with open(target_file, 'wb') as out_f:
                    pickle.dump({
                        "video_id": video_id,
                        "frame_features": frame_features
                    }, out_f)
                    
                print(f"  Saved features to {target_file}")

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Check device priority: MPS -> CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
        
    # Load frame metadata to get sampled frame IDs
    data_dir = Path("data")
    meta_path = data_dir / "processed" / "frame_metadata.json"
    if not meta_path.exists():
        print(f"ERROR: Frame metadata not found at {meta_path}!")
        print("Please run scripts/03_extract_frames.py first.")
        sys.exit(1)
        
    with open(meta_path, 'r') as f:
        frame_metadata = json.load(f)
        
    # Initialize Visual Extractor
    appearance_extractor = AppearanceExtractor(device=device)
    
    tracks_train = data_dir / "processed" / "tracks" / "train"
    tracks_val = data_dir / "processed" / "tracks" / "val"
    
    frames_train = data_dir / "frames" / "train"
    frames_val = data_dir / "frames" / "val"
    
    features_train = data_dir / "processed" / "features" / "train"
    features_val = data_dir / "processed" / "features" / "val"
    
    print("\nRunning feature extraction on training split...")
    process_features_for_split(
        "train", tracks_train, frames_train, features_train,
        frame_metadata, appearance_extractor
    )
    
    print("\nRunning feature extraction on validation split...")
    process_features_for_split(
        "val", tracks_val, frames_val, features_val,
        frame_metadata, appearance_extractor
    )
    
    print("\nFeature extraction completed successfully!")

if __name__ == "__main__":
    main()
