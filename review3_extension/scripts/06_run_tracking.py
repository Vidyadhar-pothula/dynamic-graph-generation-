import os
import sys
import yaml
import json
import time
import argparse
from pathlib import Path
import torch

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.tracking.bytetrack_tracker import ByteTrackTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Run ByteTrack tracking on videos.")
    parser.add_argument("--stride", type=int, default=1, help="Track every Nth video only")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def is_valid_track_file(json_path, expected_video_id):
    """
    Returns True if the JSON file exists, parses correctly, and contains
    the expected video_id + a tracks list.
    Returns False if missing, corrupt, or structurally invalid.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data.get("video_id") != expected_video_id:
            return False
        if "tracks" not in data or not isinstance(data["tracks"], list):
            return False
        return True
    except Exception:
        return False


STRIDE = 1


def process_tracking_for_split(split, videos_dir, tracks_dest_dir, frame_metadata,
                                tracker, conf_thresh, iou_thresh):
    videos_path = Path(videos_dir)
    dest_path = Path(tracks_dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    if not videos_path.exists():
        print(f"ERROR: Directory not found: {videos_dir}")
        return {"skipped": 0, "processed": 0, "failed": [], "corrupted": []}

    split_meta = frame_metadata.get(split, {})

    skipped = 0
    processed = 0
    failed = []
    corrupted = []

    # Collect all mp4 files
    all_videos = []
    for root, dirs, files in os.walk(videos_path):
        for f in files:
            if f.endswith('.mp4'):
                all_videos.append(Path(root) / f)

    all_videos = sorted(all_videos)[::STRIDE]
    total = len(all_videos)
    print(f"  Total {split} videos found: {total}")

    for idx, video_file in enumerate(all_videos, 1):
        video_id = video_file.stem

        # Determine output path
        rel_parts = video_file.relative_to(videos_path).parts
        subfolder = rel_parts[0] if len(rel_parts) >= 2 else ""
        target_folder = dest_path / subfolder
        target_folder.mkdir(parents=True, exist_ok=True)
        target_file = target_folder / f"{video_id}.json"

        # ── RESUME/SKIP LOGIC ─────────────────────────────────────────────────
        if target_file.exists():
            if is_valid_track_file(target_file, video_id):
                skipped += 1
                print(f"  [{idx}/{total}] SKIP  {video_id}  (already tracked, {target_file.stat().st_size // 1024}KB)")
                continue
            else:
                corrupted.append(str(video_id))
                print(f"  [{idx}/{total}] CORRUPT {video_id}  → reprocessing...")
        # ──────────────────────────────────────────────────────────────────────

        # Retrieve sampled frame IDs for this video
        video_meta = split_meta.get(video_id, {})
        sampled_frames = video_meta.get("sampled_frames", [])
        sampled_ids = {int(item["frame_id"]) for item in sampled_frames}

        print(f"  [{idx}/{total}] TRACK {video_id}  ({len(sampled_ids)} sampled frames)...", end="", flush=True)
        t0 = time.time()

        try:
            all_tracks = tracker.track_video(video_file, conf=conf_thresh, iou=iou_thresh)

            # Filter to only sampled frame IDs
            filtered_tracks = [t for t in all_tracks if t["frame_id"] in sampled_ids]

            track_data = {
                "video_id": video_id,
                "tracks": filtered_tracks
            }

            with open(target_file, 'w') as out_f:
                json.dump(track_data, out_f, indent=4)

            elapsed = time.time() - t0
            print(f" → {len(filtered_tracks)} detections saved  ({elapsed:.1f}s)")
            processed += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f" → FAILED: {str(e)}  ({elapsed:.1f}s)")
            failed.append({"video_id": video_id, "error": str(e)})

    return {
        "skipped": skipped,
        "processed": processed,
        "failed": failed,
        "corrupted": corrupted
    }


def main():
    global STRIDE
    args = parse_args()
    STRIDE = args.stride
    overall_start = time.time()

    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    detection_cfg = config['detection']

    # ── Verify frame_metadata.json ─────────────────────────────────────────────
    data_dir = Path("data")
    meta_path = data_dir / "processed" / "frame_metadata.json"
    if not meta_path.exists():
        print(f"ERROR: Frame metadata not found at {meta_path}")
        print("Please run scripts/03_extract_frames.py first.")
        sys.exit(1)

    with open(meta_path, 'r') as f:
        frame_metadata = json.load(f)

    # ── Verify best.pt ─────────────────────────────────────────────────────────
    model_path = Path("models/detection/best.pt")
    if not model_path.exists():
        print(f"ERROR: Trained model checkpoint not found at {model_path}")
        print("Please run scripts/05_train_yolo.py first.")
        sys.exit(1)

    # ── Device selection: MPS → CUDA → CPU ────────────────────────────────────
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\nDevice: {device}")
    print(f"Model : {model_path}")

    tracker = ByteTrackTracker(model_path=model_path, device=device)

    extracted_train = data_dir / "extracted" / "train"
    extracted_val   = data_dir / "extracted" / "val"
    tracks_train    = data_dir / "processed" / "tracks" / "train"
    tracks_val      = data_dir / "processed" / "tracks" / "val"

    # ── Verify extracted directories ───────────────────────────────────────────
    for p in [extracted_train, extracted_val]:
        if not p.exists():
            print(f"ERROR: Required directory missing: {p}")
            print("Do NOT regenerate — check your data/extracted/ folder.")
            sys.exit(1)

    # ── TRAIN ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRACKING: TRAIN SPLIT")
    print("="*60)
    train_result = process_tracking_for_split(
        "train", extracted_train, tracks_train, frame_metadata,
        tracker, detection_cfg['confidence'], detection_cfg['iou']
    )

    # ── VAL ────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRACKING: VALIDATION SPLIT")
    print("="*60)
    val_result = process_tracking_for_split(
        "val", extracted_val, tracks_val, frame_metadata,
        tracker, detection_cfg['confidence'], detection_cfg['iou']
    )

    # ── Final counts ───────────────────────────────────────────────────────────
    total_train_json = len(list(tracks_train.rglob("*.json")))
    total_val_json   = len(list(tracks_val.rglob("*.json")))
    total_runtime    = time.time() - overall_start

    print("\n" + "="*60)
    print("TRACKING COMPLETE — SUMMARY")
    print("="*60)
    print(f"TRAIN  skipped  : {train_result['skipped']}")
    print(f"TRAIN  processed: {train_result['processed']}")
    print(f"TRAIN  failed   : {len(train_result['failed'])}")
    print(f"TRAIN  corrupted: {len(train_result['corrupted'])}")
    print(f"TRAIN  JSONs    : {total_train_json} / 1000")
    print()
    print(f"VAL    skipped  : {val_result['skipped']}")
    print(f"VAL    processed: {val_result['processed']}")
    print(f"VAL    failed   : {len(val_result['failed'])}")
    print(f"VAL    corrupted: {len(val_result['corrupted'])}")
    print(f"VAL    JSONs    : {total_val_json} / 100")
    print()
    print(f"Total runtime   : {total_runtime/60:.1f} minutes")

    if train_result['failed'] or val_result['failed']:
        print("\nFAILED VIDEOS:")
        for item in train_result['failed'] + val_result['failed']:
            print(f"  {item['video_id']}: {item['error']}")

    if train_result['corrupted'] or val_result['corrupted']:
        print("\nCORRUPTED (reprocessed):")
        for vid in train_result['corrupted'] + val_result['corrupted']:
            print(f"  {vid}")


if __name__ == "__main__":
    main()
