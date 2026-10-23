import json
from pathlib import Path

class VidorParser:
    def __init__(self):
        pass
        
    def parse_file(self, file_path):
        """
        Parses a single VidOR raw JSON annotation file and returns a normalized dictionary.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        video_id = data.get("video_id")
        frame_count = data.get("frame_count")
        fps = data.get("fps")
        width = data.get("width")
        height = data.get("height")
        
        # Get objects mapping: tid (int) -> category (str)
        objects = {}
        for obj in data.get("subject/objects", []):
            objects[str(obj["tid"])] = obj["category"]
            
        # Get trajectories: tid (str) -> list of dicts with {"frame_id": int, "bbox": [xmin, ymin, xmax, ymax]}
        trajectories = {tid: [] for tid in objects.keys()}
        for frame_idx, frame_data in enumerate(data.get("trajectories", [])):
            for obj_box in frame_data:
                tid = str(obj_box["tid"])
                bbox = obj_box["bbox"]
                # Convert to [xmin, ymin, xmax, ymax] list
                bbox_list = [
                    int(bbox["xmin"]),
                    int(bbox["ymin"]),
                    int(bbox["xmax"]),
                    int(bbox["ymax"])
                ]
                trajectories[tid].append({
                    "frame_id": frame_idx,
                    "bbox": bbox_list
                })
                
        # Parse relation instances
        relations = []
        for rel in data.get("relation_instances", []):
            relations.append({
                "subject_id": int(rel["subject_tid"]),
                "object_id": int(rel["object_tid"]),
                "predicate": rel["predicate"],
                "begin_fid": int(rel["begin_fid"]),
                "end_fid": int(rel["end_fid"])
            })
            
        normalized = {
            "video_id": video_id,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "objects": objects,
            "trajectories": trajectories,
            "relations": relations
        }
        
        return normalized
