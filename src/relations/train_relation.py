import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

def train_relation_model(train_pkl, val_pkl, num_object_classes, num_relation_classes, epochs, batch_size, lr, weight_decay, patience, device="mps"):
    from src.relations.relation_dataset import RelationDataset
    from src.relations.relation_transformer import RelationTransformer
    
    # 1. Datasets and Loaders
    print("Loading datasets...")
    train_dataset = RelationDataset(train_pkl)
    val_dataset = RelationDataset(val_pkl)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.")
    
    # 2. Model, Loss, Optimizer
    model = RelationTransformer(
        num_object_classes=num_object_classes,
        num_relation_classes=num_relation_classes,
        d_model=256,
        nhead=4,
        num_layers=2
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    models_dir = Path("models/relation")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_relation_model.pt"
    
    print(f"Starting Relation Transformer training for {epochs} epochs on device: {device}...")
    train_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Training loop
        model.train()
        train_loss = 0.0
        for sub_feat, obj_feat, sub_class, obj_class, label in train_loader:
            sub_feat = sub_feat.to(device)
            obj_feat = obj_feat.to(device)
            sub_class = sub_class.to(device)
            obj_class = obj_class.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            logits = model(sub_feat, obj_feat, sub_class, obj_class)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sub_feat.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sub_feat, obj_feat, sub_class, obj_class, label in val_loader:
                sub_feat = sub_feat.to(device)
                obj_feat = obj_feat.to(device)
                sub_class = sub_class.to(device)
                obj_class = obj_class.to(device)
                label = label.to(device)
                
                logits = model(sub_feat, obj_feat, sub_class, obj_class)
                loss = criterion(logits, label)
                
                val_loss += loss.item() * sub_feat.size(0)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy())
                
        val_loss /= len(val_dataset)
        
        # Calculate validation metrics (accuracy, precision, recall)
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        accuracy = correct / len(val_dataset) if len(val_dataset) > 0 else 0.0
        
        # Compute macro precision/recall/f1 (excluding background if desired, or overall)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - train_start_time
        avg_epoch_time = total_elapsed / epoch
        est_remaining = avg_epoch_time * (epochs - epoch)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f} | Macro Precision: {precision:.4f} | Macro Recall: {recall:.4f}")
        print(f"Epoch Time: {format_time(epoch_time)} | Elapsed: {format_time(total_elapsed)} | Est. Remaining: {format_time(est_remaining)}")
        
        # Check early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model checkpoint to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No validation loss improvement for {patience} epochs.")
                break
                
    print("\nRelation Transformer training completed!")
    return best_model_path
