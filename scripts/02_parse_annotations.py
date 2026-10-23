import os
import sys
import json
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset.vidor_parser import VidorParser

def parse_args():
    parser = argparse.ArgumentParser(description="Parse raw VidOR annotations and save normalized ones.")
    return parser.parse_args()

def process_directory(source_dir, target_dir):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Warning: Source directory does not exist: {source_dir}")
        return
        
    parser = VidorParser()
    count = 0
    
    # Recurse through raw annotations
    for root, dirs, files in os.walk(source_path):
        for f in files:
            if f.endswith('.json'):
                source_file = Path(root) / f
                
                # Get the relative structure from source_path
                rel_parts = source_file.relative_to(source_path).parts
                if len(rel_parts) >= 2:
                    subfolder = rel_parts[0]
                else:
                    subfolder = ""
                    
                dest_folder = target_path / subfolder
                dest_folder.mkdir(parents=True, exist_ok=True)
                dest_file = dest_folder / f
                
                # Parse and save
                try:
                    normalized = parser.parse_file(source_file)
                    with open(dest_file, 'w') as out_f:
                        json.dump(normalized, out_f, indent=4)
                    count += 1
                except Exception as e:
                    print(f"Error parsing {source_file}: {str(e)}")
                    
    print(f"Finished. Normalized {count} files under {target_dir}")

def main():
    args = parse_args()
    
    data_dir = Path("data")
    ann_train = data_dir / "annotations" / "train"
    ann_val = data_dir / "annotations" / "val"
    
    proc_ann_train = data_dir / "processed" / "annotations" / "train"
    proc_ann_val = data_dir / "processed" / "annotations" / "val"
    
    print("Normalizing training annotations...")
    process_directory(ann_train, proc_ann_train)
    
    print("Normalizing validation annotations...")
    process_directory(ann_val, proc_ann_val)

if __name__ == "__main__":
    main()
