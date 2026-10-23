import pickle
import torch
from torch.utils.data import Dataset

class RelationDataset(Dataset):
    def __init__(self, pickle_path):
        """
        PyTorch Dataset for relationship prediction.
        Loads paired subject and object features from a pickle file.
        """
        with open(pickle_path, 'rb') as f:
            self.samples = pickle.load(f)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Features: float32
        sub_feat = torch.tensor(sample["subject_features"], dtype=torch.float32)
        obj_feat = torch.tensor(sample["object_features"], dtype=torch.float32)
        
        # Object Category IDs: long
        sub_class = torch.tensor(sample["subject_class_id"], dtype=torch.long)
        obj_class = torch.tensor(sample["object_class_id"], dtype=torch.long)
        
        # Relationship Label ID: long
        label_id = torch.tensor(sample["label_id"], dtype=torch.long)
        
        return sub_feat, obj_feat, sub_class, obj_class, label_id
