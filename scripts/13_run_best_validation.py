import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    config_path = "configs/config.yaml"
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_path = "models/detection/best.pt"
    data_yaml = "data/processed/yolo/data.yaml"
    
    if not Path(model_path).exists():
        print(f"ERROR: Best model checkpoint not found at {model_path}!")
        sys.exit(1)
        
    print(f"Loading best YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    output_project = "data"
    output_name = "validation_output"
    
    print(f"Running validation and saving plots to {output_project}/{output_name}/...")
    metrics = model.val(
        data=data_yaml,
        device="mps",
        project=output_project,
        name=output_name,
        exist_ok=True,
        plots=True,
        verbose=True
    )
    
    print("\n==================================================")
    print("BEST MODEL VALIDATION METRICS SUMMARY")
    print("==================================================")
    print(f"Mean Precision: {metrics.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Mean Recall: {metrics.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print(f"mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print("==================================================\n")
    print("All validation plots and curve graphs saved successfully!")

if __name__ == "__main__":
    main()
