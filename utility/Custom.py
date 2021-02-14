import torch
import torch.utils.data as data

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class CustomDataset(Dataset):
    def __init__(self, data_a, label):
        self.len = data_a.shape[0]
        self.x_data = data_a
        self.y_data = label
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len