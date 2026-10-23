import os
import sys
import pickle
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare sequence dataset for Temporal model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def prepare_sequences_for_split(pkl_path, window_size=10, step_size=5):
    if not Path(pkl_path).exists():
        print(f"Warning: Relation dataset not found: {pkl_path}")
        return []
        
    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)
        
    # Group samples by (video_id, subject_track_id, object_track_id)
    groups = {}
    for s in samples:
        key = (s["video_id"], s["subject_track_id"], s["object_track_id"])
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
        
    sequences = []
    
    for key, group in groups.items():
        video_id, sub_tid, obj_tid = key
        
        # Sort group by frame_id to make it sequential
        sorted_group = sorted(group, key=lambda x: x["frame_id"])
        N = len(sorted_group)
        
        if N < window_size:
            # Pad sequence at the beginning
            pad_len = window_size - N
            
            # Pad features with zeros
            features_seq = []
            for _ in range(pad_len):
                features_seq.append(np.zeros(1054, dtype=np.float32))
                
            labels_seq = [0] * pad_len # 0 represents __background__
            mask_seq = [0] * pad_len # 0 means padded/masked out
            
            # Add actual values
            for s in sorted_group:
                # Concatenate subject and object features (527 + 527 = 1054)
                feat = np.concatenate([s["subject_features"], s["object_features"]], axis=-1)
                features_seq.append(feat)
                labels_seq.append(s["label_id"])
                mask_seq.append(1)
                
            sequences.append({
                "video_id": video_id,
                "subject_track_id": sub_tid,
                "object_track_id": obj_tid,
                "features": np.stack(features_seq).astype(np.float32),
                "labels": np.array(labels_seq, dtype=np.int64),
                "mask": np.array(mask_seq, dtype=np.float32)
            })
            
        else:
            # Extract sliding windows
            for start_idx in range(0, N - window_size + 1, step_size):
                end_idx = start_idx + window_size
                window_samples = sorted_group[start_idx:end_idx]
                
                features_seq = []
                labels_seq = []
                mask_seq = []
                
                for s in window_samples:
                    feat = np.concatenate([s["subject_features"], s["object_features"]], axis=-1)
                    features_seq.append(feat)
                    labels_seq.append(s["label_id"])
                    mask_seq.append(1)
                    
                sequences.append({
                    "video_id": video_id,
                    "subject_track_id": sub_tid,
                    "object_track_id": obj_tid,
                    "features": np.stack(features_seq).astype(np.float32),
                    "labels": np.array(labels_seq, dtype=np.int64),
                    "mask": np.array(mask_seq, dtype=np.float32)
                })
                
    return sequences

def main():
    args = parse_args()
    
    data_dir = Path("data/processed")
    train_pkl = data_dir / "relation_train.pkl"
    val_pkl = data_dir / "relation_val.pkl"
    
    print("Preparing temporal sequence samples for training split...")
    train_seqs = prepare_sequences_for_split(train_pkl, window_size=10, step_size=5)
    
    print("Preparing temporal sequence samples for validation split...")
    val_seqs = prepare_sequences_for_split(val_pkl, window_size=10, step_size=5)
    
    print(f"\nCreated {len(train_seqs)} training sequences and {len(val_seqs)} validation sequences.")
    
    # Save sequence datasets
    with open(data_dir / "temporal_train.pkl", "wb") as f:
        pickle.dump(train_seqs, f)
    print(f"Saved training sequences to {data_dir / 'temporal_train.pkl'}")
    
    with open(data_dir / "temporal_val.pkl", "wb") as f:
        pickle.dump(val_seqs, f)
    print(f"Saved validation sequences to {data_dir / 'temporal_val.pkl'}")
    
    print("\nTemporal sequence data preparation completed successfully!")

if __name__ == "__main__":
    main()
