# Generated from: Dataset.ipynb
# Converted at: 2026-10-19T01:37:27.769Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    """PyTorch Dataset for MNIST CSV files."""
    
    def __init__(self, dataframe, transform=None, is_test=False):
        if is_test:
            self.images = dataframe.values.reshape(-1, 28, 28).astype(np.float32) / 255.0
            self.labels = None
        else:
            self.labels = dataframe.iloc[:, 0].values
            self.images = dataframe.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] if self.labels is not None else -1
        
        image = torch.tensor(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, torch.tensor(label, dtype=torch.long)
        return image