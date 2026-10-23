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

def train_temporal_model(train_pkl, val_pkl, num_classes, model_type, epochs, batch_size, lr, patience, device="mps"):
    from src.temporal.temporal_dataset import TemporalDataset
    
    # Select model class
    if model_type.lower() == "lstm":
        from src.temporal.temporal_lstm import TemporalLSTM
        model_class = TemporalLSTM
    elif model_type.lower() == "gru":
        from src.temporal.temporal_gru import TemporalGRU
        model_class = TemporalGRU
    else:
        raise ValueError(f"Unknown temporal model type: {model_type}")
        
    # 1. Dataset and Loader
    print(f"Loading sequence datasets for {model_type.upper()} model...")
    train_dataset = TemporalDataset(train_pkl)
    val_dataset = TemporalDataset(val_pkl)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Loaded {len(train_dataset)} training sequences and {len(val_dataset)} validation sequences.")
    
    # 2. Model, Loss, Optimizer
    model = model_class(input_dim=1054, hidden_dim=256, num_layers=2, num_classes=num_classes)
    model.to(device)
    
    # CrossEntropyLoss with reduction='none' so we can apply the padding mask
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    models_dir = Path("models/temporal")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"best_temporal_{model_type.lower()}_model.pt"
    
    print(f"Starting {model_type.upper()} training for {epochs} epochs on device: {device}...")
    train_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Training loop
        model.train()
        train_loss = 0.0
        total_valid_steps_train = 0
        
        for features, labels, mask in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            logits = model(features)  # (batch_size, seq_len, num_classes)
            
            # Reshape for cross entropy
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)
            
            loss_unmasked = criterion(logits_flat, labels_flat)
            loss = (loss_unmasked * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * mask_flat.sum().item()
            total_valid_steps_train += mask_flat.sum().item()
            
        train_loss = train_loss / total_valid_steps_train if total_valid_steps_train > 0 else 0.0
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        total_valid_steps_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                
                logits = model(features)
                
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                mask_flat = mask.view(-1)
                
                loss_unmasked = criterion(logits_flat, labels_flat)
                loss = (loss_unmasked * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                val_loss += loss.item() * mask_flat.sum().item()
                total_valid_steps_val += mask_flat.sum().item()
                
                # Get predictions and ground truth only for valid (non-masked) steps
                preds = torch.argmax(logits_flat, dim=1).cpu().numpy()
                lbls = labels_flat.cpu().numpy()
                msk = mask_flat.cpu().numpy()
                
                for idx in range(len(msk)):
                    if msk[idx] > 0.5:
                        all_preds.append(preds[idx])
                        all_labels.append(lbls[idx])
                        
        val_loss = val_loss / total_valid_steps_val if total_valid_steps_val > 0 else 0.0
        
        # Metrics
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0
        
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
        
        # Check early stopping
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
                
    print(f"\n{model_type.upper()} temporal model training completed!")
    return best_model_path
