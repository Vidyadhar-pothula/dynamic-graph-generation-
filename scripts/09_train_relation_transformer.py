import os
import sys
import yaml
import json
import torch
import argparse
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.relations.train_relation import train_relation_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Relation Transformer.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cpu or mps)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Run 1-epoch relation smoke test")
    group.add_argument("--pilot", action="store_true", help="Run 5-epoch relation pilot test")
    group.add_argument("--full", action="store_true", help="Run full 15-epoch relation training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    relation_cfg = config['relation']
    
    # Check dataset splits
    data_dir = Path("data/processed")
    train_pkl = data_dir / "relation_train.pkl"
    val_pkl = data_dir / "relation_val.pkl"
    vocab_json = data_dir / "relation_vocabulary.json"
    
    if not train_pkl.exists() or not val_pkl.exists() or not vocab_json.exists():
        print("ERROR: Relation data pickle or vocabulary not found!")
        print("Please run scripts/08_prepare_relation_data.py first.")
        sys.exit(1)
        
    # Determine epochs
    if args.smoke:
        epochs = relation_cfg['epochs']['smoke_test']
        print("--- RUNNING RELATION SMOKE TEST ---")
    elif args.pilot:
        epochs = relation_cfg['epochs']['pilot']
        print("--- RUNNING RELATION PILOT ---")
    else:
        epochs = relation_cfg['epochs']['full']
        print("--- RUNNING RELATION FULL TRAINING ---")
        
    # Load vocabulary to get number of relation classes
    with open(vocab_json, 'r') as f:
        vocab = json.load(f)
    num_relation_classes = len(vocab)
    
    # Set device priority (defaults to cpu)
    device = args.device if args.device else "cpu"
        
    num_object_classes = 28 # Constant mapped from yolo prep
    
    # Run training
    train_relation_model(
        train_pkl=str(train_pkl),
        val_pkl=str(val_pkl),
        num_object_classes=num_object_classes,
        num_relation_classes=num_relation_classes,
        epochs=epochs,
        batch_size=relation_cfg['batch_size'],
        lr=relation_cfg['learning_rate'],
        weight_decay=relation_cfg['weight_decay'],
        patience=relation_cfg['patience'],
        device=device
    )

if __name__ == "__main__":
    main()
