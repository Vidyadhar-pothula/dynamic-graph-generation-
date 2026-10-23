import os
from pathlib import Path
from ultralytics import YOLO

class ByteTrackTracker:
    def __init__(self, model_path="models/detection/best.pt", device="mps"):
        self.device = device
        print(f"Loading YOLO tracking model from {model_path} on {device}...")
        self.model = YOLO(model_path)
        
    def track_video(self, video_path, conf=0.25, iou=0.50, tracker_type="bytetrack.yaml"):
        """
        Runs tracking on a video file frame-by-frame (streaming) and returns a list of tracks.
        Each track is a dict with: frame_id, track_id, class_id, class_name, bbox, confidence.
        """
        # Run tracking using stream=True to prevent loading all frames into RAM
        results = self.model.track(
            source=str(video_path),
            tracker=tracker_type,
            persist=True,
            stream=True,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        tracks = []
        for frame_idx, r in enumerate(results):
            # If no boxes or IDs are detected, skip
            if r.boxes is None or r.boxes.id is None:
                continue
                
            xyxy = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            
            for i in range(len(track_ids)):
                tracks.append({
                    "frame_id": frame_idx,
                    "track_id": int(track_ids[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": r.names[class_ids[i]],
                    "bbox": [int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3])],
                    "confidence": float(confidences[i])
                })
                
        return tracks
