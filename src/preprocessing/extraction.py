import os
import zipfile
from pathlib import Path

def extract_annotations(zip_path, dest_dir, limit=None):
    """
    Extracts annotation json files from raw zip archive.
    If limit is specified, extracts only the first N json files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting annotations from {zip_path} to {dest_dir}...")
    
    count = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Sort files to ensure deterministic subset selection if limited
        files = sorted([f for f in zf.namelist() if f.endswith('.json')])
        
        for file in files:
            if limit is not None and count >= limit:
                break
                
            # Extract content
            member = zf.getinfo(file)
            # The folder structure in zip is training/<subfolder>/<id>.json
            # We want to extract it so that we map the inner parts.
            # e.g., training/0000/xxx.json -> we can just extract it and then reorganize
            # or extract individual files directly to dest_dir / subfolder / filename.
            parts = Path(file).parts # ('training', '0000', 'xxx.json')
            if len(parts) >= 3:
                subfolder = parts[1]
                filename = parts[2]
            else:
                subfolder = ""
                filename = parts[-1]
                
            target_dir = dest_path / subfolder
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / filename
            
            # Avoid extracting again if it already exists
            if not target_file.exists():
                with zf.open(file) as source, open(target_file, "wb") as target:
                    target.write(source.read())
            
            count += 1
            if count % 1000 == 0:
                print(f"Extracted {count} annotations...")
                
    print(f"Extraction complete. Total extracted: {count}")
    return count

def extract_balanced_per_part_videos(video_zip_paths, ann_zip_path, dest_video_dir, dest_ann_dir, per_part_limit=500):
    """
    Extracts up to `per_part_limit` MP4 videos from EACH specified video ZIP archive,
    and extracts their matching annotation JSONs from `ann_zip_path`.
    """
    dest_video_path = Path(dest_video_dir)
    dest_ann_path = Path(dest_ann_dir)
    dest_video_path.mkdir(parents=True, exist_ok=True)
    dest_ann_path.mkdir(parents=True, exist_ok=True)
    
    extracted_video_ids = set()
    video_subfolder_map = {} # video_id -> subfolder
    
    # 1. Extract up to per_part_limit videos per ZIP
    for zip_path in video_zip_paths:
        zp = Path(zip_path)
        if not zp.exists():
            print(f"Warning: Zip file not found: {zip_path}")
            continue
            
        print(f"\nScanning and extracting up to {per_part_limit} videos from {zp.name}...")
        count_from_this_zip = 0
        
        with zipfile.ZipFile(zp, 'r') as zf:
            mp4_members = sorted([m for m in zf.namelist() if m.endswith('.mp4')])
            
            for member in mp4_members:
                if count_from_this_zip >= per_part_limit:
                    break
                    
                parts = Path(member).parts
                if len(parts) >= 3:
                    subfolder = parts[1]
                    video_id = Path(parts[2]).stem
                else:
                    subfolder = ""
                    video_id = Path(parts[-1]).stem
                    
                target_dir = dest_video_path / subfolder
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / f"{video_id}.mp4"
                
                if not target_file.exists():
                    with zf.open(member) as source, open(target_file, "wb") as target:
                        target.write(source.read())
                        
                extracted_video_ids.add(video_id)
                video_subfolder_map[video_id] = subfolder
                count_from_this_zip += 1
                
        print(f"Extracted {count_from_this_zip} videos from {zp.name}")
        
    print(f"\nTotal distinct training videos extracted: {len(extracted_video_ids)}")
    
    # 2. Extract matching annotations from training-annotation.zip
    print(f"Extracting matching annotations for {len(extracted_video_ids)} videos from {ann_zip_path}...")
    ann_count = 0
    with zipfile.ZipFile(ann_zip_path, 'r') as zf:
        # Build map of video_id -> zip_member
        ann_members = {Path(m).stem: m for m in zf.namelist() if m.endswith('.json')}
        
        for vid in extracted_video_ids:
            if vid in ann_members:
                member = ann_members[vid]
                subfolder = video_subfolder_map.get(vid, "")
                target_dir = dest_ann_path / subfolder
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / f"{vid}.json"
                
                if not target_file.exists():
                    with zf.open(member) as source, open(target_file, "wb") as target:
                        target.write(source.read())
                ann_count += 1
                
    print(f"Successfully extracted {ann_count} matching annotation files.")
    return len(extracted_video_ids)

def extract_videos_matching_annotations(zip_paths, annotations_dir, dest_dir, limit=None):
    """
    Scans the annotations folder to see which video IDs we have,
    and extracts only those videos from the list of video zip files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all expected video files (e.g. "0000/2401075277.mp4")
    # from the directory structure under annotations_dir.
    expected_videos = []
    annotations_path = Path(annotations_dir)
    for root, dirs, files in os.walk(annotations_path):
        for f in files:
            if f.endswith('.json'):
                video_id = Path(f).stem
                subfolder = Path(root).name
                # Note: if root is annotations_dir itself, subfolder is empty
                if subfolder == annotations_path.name:
                    subfolder = ""
                expected_videos.append((subfolder, video_id))
                
    if limit is not None:
        expected_videos = expected_videos[:limit]
        
    print(f"Searching for and extracting {len(expected_videos)} videos from ZIPs...")
    
    extracted_count = 0
    extracted_ids = set()
    
    for zip_path in zip_paths:
        if not os.path.exists(zip_path):
            print(f"Warning: Zip file not found: {zip_path}")
            continue
            
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            # Map filenames in zip for fast lookups
            # Format in zip: "video/0000/2401075277.mp4"
            name_map = {}
            for name in namelist:
                if name.endswith('.mp4'):
                    parts = Path(name).parts
                    if len(parts) >= 3:
                        name_map[(parts[1], Path(parts[2]).stem)] = name
                    else:
                        name_map[("", Path(parts[-1]).stem)] = name
                        
            # Check which expected videos are in this ZIP
            for subfolder, video_id in expected_videos:
                if (subfolder, video_id) in name_map:
                    zip_name = name_map[(subfolder, video_id)]
                    target_dir = dest_path / subfolder
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_file = target_dir / f"{video_id}.mp4"
                    
                    if not target_file.exists():
                        with zf.open(zip_name) as source, open(target_file, "wb") as target:
                            target.write(source.read())
                            
                    extracted_count += 1
                    extracted_ids.add(video_id)
                    
    print(f"Extraction complete. Total videos extracted/verified: {extracted_count}")
    return extracted_count

