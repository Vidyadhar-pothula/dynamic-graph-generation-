import torch
from ultralytics import YOLO
from pathlib import Path

class YoloDetector:
    def __init__(self, model_path="yolov8s.pt", device=None):
        """
        Initializes the YOLOv8 detector.
        If device is None, auto-selects MPS or CPU.
        """
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Initializing YOLOv8 from {model_path} on device: {self.device}")
        self.model = YOLO(model_path)
        
    def detect(self, img, conf=0.25, iou=0.50):
        """
        Runs object detection on a single image.
        Returns:
            list of dicts containing:
                'bbox': [xmin, ymin, xmax, ymax]
                'confidence': float
                'class_id': int
                'class_name': str
        """
        results = self.model(img, conf=conf, iou=iou, device=self.device, verbose=False)
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for box in boxes:
                # Get bounding box in absolute [xmin, ymin, xmax, ymax]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf_score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                detections.append({
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf_score,
                    "class_id": cls_id,
                    "class_name": cls_name
                })
                
        return detections
