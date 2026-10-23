import pickle
import torch
from torch.utils.data import Dataset

class TemporalDataset(Dataset):
    def __init__(self, pickle_path):
        """
        PyTorch Dataset for relationship sequence modeling.
        """
        with open(pickle_path, 'rb') as f:
            self.sequences = pickle.load(f)
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        features = torch.tensor(seq["features"], dtype=torch.float32)
        labels = torch.tensor(seq["labels"], dtype=torch.long)
        mask = torch.tensor(seq["mask"], dtype=torch.float32)
        
        return features, labels, mask
