import os
import sys
import yaml
import json
import torch
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.temporal.train_temporal import train_temporal_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Temporal sequence model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"], help="Temporal model type")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cpu or mps)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run 1-epoch temporal smoke test")
    group.add_argument("--pilot", action="store_true", help="Run 5-epoch temporal pilot test")
    group.add_argument("--full", action="store_true", help="Run full 15-epoch temporal training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    temporal_cfg = config['temporal']
    
    # Check dataset splits
    data_dir = Path("data/processed")
    train_pkl = data_dir / "temporal_train.pkl"
    val_pkl = data_dir / "temporal_val.pkl"
    vocab_json = data_dir / "relation_vocabulary.json"
    
    if not train_pkl.exists() or not val_pkl.exists() or not vocab_json.exists():
        print("ERROR: Temporal sequence data pickle or vocabulary not found!")
        print("Please run scripts/10_prepare_temporal_data.py first.")
        sys.exit(1)
        
    # Determine epochs
    if args.smoke:
        epochs = temporal_cfg['epochs']['smoke_test']
        print(f"--- RUNNING TEMPORAL {args.model.upper()} SMOKE TEST ---")
    elif args.pilot:
        epochs = temporal_cfg['epochs']['pilot']
        print(f"--- RUNNING TEMPORAL {args.model.upper()} PILOT ---")
    else:
        epochs = temporal_cfg['epochs']['full']
        print(f"--- RUNNING TEMPORAL {args.model.upper()} FULL TRAINING ---")
        
    # Load vocabulary to get number of classes
    with open(vocab_json, 'r') as f:
        vocab = json.load(f)
    num_classes = len(vocab)
    
    # Set device priority (defaults to cpu)
    device = args.device if args.device else "cpu"
        
    # Run training
    train_temporal_model(
        train_pkl=str(train_pkl),
        val_pkl=str(val_pkl),
        num_classes=num_classes,
        model_type=args.model,
        epochs=epochs,
        batch_size=temporal_cfg['batch_size'],
        lr=temporal_cfg['learning_rate'],
        patience=temporal_cfg['patience'],
        device=device
    )

if __name__ == "__main__":
    main()
