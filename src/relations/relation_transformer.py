import torch
import torch.nn as nn

class RelationTransformer(nn.Module):
    def __init__(self, num_object_classes=28, num_relation_classes=51, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Projection layers for 527-dim fused features
        self.feat_proj = nn.Linear(527, d_model)
        
        # Class embeddings
        self.class_embed = nn.Embedding(num_object_classes, d_model)
        
        # Layer Normalization and Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification head (combining subject and object outputs)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_relation_classes)
        )
        
    def forward(self, sub_feat, obj_feat, sub_class, obj_class):
        """
        sub_feat: (batch_size, 527)
        obj_feat: (batch_size, 527)
        sub_class: (batch_size,) - object class ID for subject
        obj_class: (batch_size,) - object class ID for object
        """
        # Project features to d_model
        sub_proj = self.feat_proj(sub_feat)
        obj_proj = self.feat_proj(obj_feat)
        
        # Add class embeddings
        sub_embed = sub_proj + self.class_embed(sub_class)
        obj_embed = obj_proj + self.class_embed(obj_class)
        
        # Apply normalization and dropout
        sub_embed = self.dropout(self.norm(sub_embed))
        obj_embed = self.dropout(self.norm(obj_embed))
        
        # Pack into sequence of shape (batch_size, seq_len=2, d_model)
        seq = torch.stack([sub_embed, obj_embed], dim=1)
        
        # Pass through Transformer encoder
        encoded = self.transformer(seq)  # (batch_size, 2, d_model)
        
        # Flatten sequence dimension to feed to classifier
        pooled = encoded.reshape(encoded.size(0), -1)  # (batch_size, d_model * 2)
        
        # Predict relationship logits
        logits = self.classifier(pooled)
        return logits
