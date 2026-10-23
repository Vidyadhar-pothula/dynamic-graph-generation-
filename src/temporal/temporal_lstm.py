import torch
import torch.nn as nn

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=1054, hidden_dim=256, num_layers=2, num_classes=51, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classification head for each sequence step
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len=10, input_dim=1054)
        Returns:
            logits: (batch_size, seq_len=10, num_classes)
        """
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_dim)
        logits = self.classifier(out)  # logits: (batch_size, seq_len, num_classes)
        return logits
