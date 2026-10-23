import cv2
from pathlib import Path

def sample_video_frames(video_path, dest_dir, target_fps=2.0):
    """
    Samples frames from a video file and saves them to the destination directory.
    Uses cv2.VideoCapture and computes the step size to achieve target_fps.
    Saves frames as frame_{frame_id:06d}.jpg preserving original index.
    """
    video_path = Path(video_path)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if orig_fps <= 0:
        orig_fps = 30.0 # Default fallback
        
    # Calculate step size
    step = max(1, int(round(orig_fps / target_fps)))
    
    sampled_metadata = []
    frame_id = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_id % step == 0:
            filename = f"frame_{frame_id:06d}.jpg"
            out_file = dest_path / filename
            
            # Save frame to disk
            if not out_file.exists():
                cv2.imwrite(str(out_file), frame)
                
            sampled_metadata.append({
                "frame_id": frame_id,
                "filename": filename,
                "timestamp": frame_id / orig_fps
            })
            saved_count += 1
            
        frame_id += 1
        
    cap.release()
    return sampled_metadata, total_frames, orig_fps
