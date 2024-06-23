import torch
from torch.utils.data import Dataset

class TK100Dataset(Dataset):
    def __init__(self, data_info_file_path, transform=None):
        super().__init__()

        self.data_info_file_path = data_info_file_path
        self.transform = transform

        

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()