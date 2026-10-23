import os
import sys
import yaml
import json
import pickle
import numpy as np
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.feature_fusion import fuse_features

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare relation dataset from features and annotations.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()

def calculate_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    return intersection / union

def get_unique_predicates(annotations_dir):
    """
    Scans annotation files to find all unique predicates.
    """
    ann_path = Path(annotations_dir)
    if not ann_path.exists():
        return set()
        
    predicates = set()
    for root, dirs, files in os.walk(ann_path):
        for f in files:
            if f.endswith('.json'):
                with open(Path(root) / f, 'r') as json_f:
                    data = json.load(json_f)
                for rel in data.get("relations", []):
                    predicates.add(rel["predicate"])
    return predicates

def build_pairs_for_split(split, features_dir, annotations_dir, predicate_to_id, neg_pos_ratio=3.0):
    features_path = Path(features_dir)
    annotations_path = Path(annotations_dir)
    
    if not features_path.exists():
        print(f"Warning: Features path not found: {features_dir}")
        return []
        
    samples = []
    
    # Process each features file
    for root, dirs, files in os.walk(features_path):
        for f in files:
            if f.endswith('.pkl'):
                feat_file = Path(root) / f
                video_id = feat_file.stem
                
                # Get subfolder
                rel_parts = feat_file.relative_to(features_path).parts
                if len(rel_parts) >= 2:
                    subfolder = rel_parts[0]
                else:
                    subfolder = ""
                    
                ann_file = annotations_path / subfolder / f"{video_id}.json"
                if not ann_file.exists():
                    continue
                    
                # Load features
                with open(feat_file, 'rb') as feat_f:
                    feat_data = pickle.load(feat_f)
                frame_features = feat_data.get("frame_features", {})
                
                # Load annotations
                with open(ann_file, 'r') as ann_f:
                    ann_data = json.load(ann_f)
                relations = ann_data.get("relations", [])
                trajectories = ann_data.get("trajectories", {})
                
                # Group ground truth bboxes by frame_id
                # gt_tid -> frame_id -> bbox
                gt_by_frame = {}
                for gt_tid, traj in trajectories.items():
                    gt_tid_int = int(gt_tid)
                    for point in traj:
                        fid = int(point["frame_id"])
                        if fid not in gt_by_frame:
                            gt_by_frame[fid] = {}
                        gt_by_frame[fid][gt_tid_int] = point["bbox"]
                        
                video_positives = 0
                video_negatives = []
                
                # Iterate through frames
                for fid, tracks_dict in frame_features.items():
                    # Match predicted track_ids to ground truth tids using IoU
                    track_to_gt = {}
                    gt_bboxes = gt_by_frame.get(fid, {})
                    
                    for tid, track_info in tracks_dict.items():
                        bbox = track_info["bbox"]
                        best_gt_id = None
                        best_iou = 0.5 # Minimum overlap threshold
                        
                        for gt_tid_int, gt_bbox in gt_bboxes.items():
                            iou = calculate_iou(bbox, gt_bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_id = gt_tid_int
                                
                        if best_gt_id is not None:
                            track_to_gt[tid] = best_gt_id
                            
                    # Pair up all tracks in this frame
                    track_ids = list(tracks_dict.keys())
                    for i in range(len(track_ids)):
                        for j in range(len(track_ids)):
                            if i == j:
                                continue
                                
                            sub_tid = track_ids[i]
                            obj_tid = track_ids[j]
                            
                            sub_info = tracks_dict[sub_tid]
                            obj_info = tracks_dict[obj_tid]
                            
                            # Fuse features
                            sub_fused = fuse_features(
                                sub_info["appearance"], sub_info["spatial"], sub_info["motion"]
                            )
                            obj_fused = fuse_features(
                                obj_info["appearance"], obj_info["spatial"], obj_info["motion"]
                            )
                            
                            # Check relationship label
                            label = "__background__"
                            sub_gt = track_to_gt.get(sub_tid)
                            obj_gt = track_to_gt.get(obj_tid)
                            
                            if sub_gt is not None and obj_gt is not None:
                                # Look for relationship in relations
                                for r in relations:
                                    if r["subject_id"] == sub_gt and r["object_id"] == obj_gt:
                                        if r["begin_fid"] <= fid <= r["end_fid"]:
                                            label = r["predicate"]
                                            break
                                            
                            label_id = predicate_to_id[label]
                            
                            sample = {
                                "video_id": video_id,
                                "frame_id": fid,
                                "subject_track_id": sub_tid,
                                "object_track_id": obj_tid,
                                "subject_features": sub_fused.astype(np.float32),
                                "object_features": obj_fused.astype(np.float32),
                                "subject_class_id": sub_info["class_id"],
                                "object_class_id": obj_info["class_id"],
                                "label_id": label_id,
                                "label_name": label
                            }
                            
                            if label != "__background__":
                                samples.append(sample)
                                video_positives += 1
                            else:
                                video_negatives.append(sample)
                                
                # Downsample backgrounds for this video
                if video_positives > 0 and neg_pos_ratio > 0:
                    max_negs = int(video_positives * neg_pos_ratio)
                    if len(video_negatives) > max_negs:
                        # Shuffle and sample
                        indices = np.random.choice(len(video_negatives), max_negs, replace=False)
                        video_negatives = [video_negatives[idx] for idx in indices]
                        
                samples.extend(video_negatives)
                
    return samples

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    neg_pos_ratio = config.get('relation', {}).get('neg_pos_ratio', 3.0)
    
    data_dir = Path("data")
    proc_ann_dir = data_dir / "processed" / "annotations"
    
    # 1. Build predicate vocabulary
    print("Building predicate vocabulary...")
    train_preds = get_unique_predicates(proc_ann_dir / "train")
    val_preds = get_unique_predicates(proc_ann_dir / "val")
    all_preds = sorted(list(train_preds.union(val_preds)))
    
    predicate_to_id = {"__background__": 0}
    for idx, pred in enumerate(all_preds):
        predicate_to_id[pred] = idx + 1
        
    print(f"Discovered {len(predicate_to_id)} relationship classes (including __background__).")
    print(f"Predicate Mapping: {predicate_to_id}")
    
    # Save vocabulary
    vocab_dir = data_dir / "processed"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    with open(vocab_dir / "relation_vocabulary.json", "w") as f:
        json.dump(predicate_to_id, f, indent=4)
        
    features_train = data_dir / "processed" / "features" / "train"
    features_val = data_dir / "processed" / "features" / "val"
    
    print("\nPreparing training pairs...")
    train_samples = build_pairs_for_split(
        "train", features_train, proc_ann_dir / "train", predicate_to_id, neg_pos_ratio
    )
    
    print("\nPreparing validation pairs...")
    val_samples = build_pairs_for_split(
        "val", features_val, proc_ann_dir / "val", predicate_to_id, neg_pos_ratio
    )
    
    # Save datasets
    print(f"\nSaving {len(train_samples)} training samples to relation_train.pkl...")
    with open(vocab_dir / "relation_train.pkl", "wb") as f:
        pickle.dump(train_samples, f)
        
    print(f"Saving {len(val_samples)} validation samples to relation_val.pkl...")
    with open(vocab_dir / "relation_val.pkl", "wb") as f:
        pickle.dump(val_samples, f)
        
    print("\nRelation data preparation completed successfully!")

if __name__ == "__main__":
    main()
