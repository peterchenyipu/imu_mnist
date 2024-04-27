import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

class EdgeImpulseDataset(Dataset):
    def __init__(self, directory, split='training', stratify=True, split_ratio=0.9, random_state=42):
        self.directory = directory
        self.split = split
        self.files = [f for f in os.listdir(directory) if f.endswith('.json') and not f.startswith('info')]
        self.files.sort()  # Ensuring a consistent order
        
        if stratify:
            # Gather files by labels
            files_by_label = {}
            for f in self.files:
                label = int(f[0])  # Assuming the first character of the filename represents the label
                if label not in files_by_label:
                    files_by_label[label] = []
                files_by_label[label].append(f)
            
            train_files = []
            val_files = []
            # Split each label group
            for label, files in files_by_label.items():
                train_f, val_f = train_test_split(files, test_size=1.0 - split_ratio, random_state=random_state)
                train_files.extend(train_f)
                val_files.extend(val_f)
            
            # Shuffle the splits
            np.random.shuffle(train_files)
            np.random.shuffle(val_files)
            
            if split == 'training':
                self.files = train_files
            elif split == 'validation':
                self.files = val_files
        else:
            # Non-stratified split
            split_thresh = int(split_ratio * len(self.files))
            if split == 'training':
                self.files = self.files[:split_thresh]
            elif split == 'validation':
                self.files = self.files[split_thresh:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        values = data['payload']['values']
        values_flat = np.array(values).flatten()
        
        if len(values_flat) != 1800:
            values_flat = np.pad(values_flat, (0, 1800-len(values_flat)), 'constant')
        if len(values_flat) > 1800:
            values_flat = values_flat[:1800]
        
        values_tensor = torch.tensor(values_flat, dtype=torch.float)
        label = int(self.files[idx][0])
        
        return values_tensor, label
    
class EdgeImpulseAccelDataset(Dataset):
    def __init__(self, directory, split='training', stratify=True, split_ratio=0.9, random_state=42):
        self.directory = directory
        self.split = split
        self.files = [f for f in os.listdir(directory) if f.endswith('.json') and not f.startswith('info')]
        self.files.sort()  # Ensuring a consistent order
        
        if stratify:
            # Gather files by labels
            files_by_label = {}
            for f in self.files:
                label = int(f[0])  # Assuming the first character of the filename represents the label
                if label not in files_by_label:
                    files_by_label[label] = []
                files_by_label[label].append(f)
            
            train_files = []
            val_files = []
            # Split each label group
            for label, files in files_by_label.items():
                train_f, val_f = train_test_split(files, test_size=1.0 - split_ratio, random_state=random_state)
                train_files.extend(train_f)
                val_files.extend(val_f)
            
            # Shuffle the splits
            np.random.shuffle(train_files)
            np.random.shuffle(val_files)
            
            if split == 'training':
                self.files = train_files
            elif split == 'validation':
                self.files = val_files
        else:
            # Non-stratified split
            split_thresh = int(split_ratio * len(self.files))
            if split == 'training':
                self.files = self.files[:split_thresh]
            elif split == 'validation':
                self.files = self.files[split_thresh:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        values = data['payload']['values']
        # accel data is the first 3 values of each row
        values_flat = np.array(values)[:, :3].flatten()
        
        if len(values_flat) != 900:
            values_flat = np.pad(values_flat, (0, 900-len(values_flat)), 'constant')
        if len(values_flat) > 900:
            values_flat = values_flat[:900]
        
        values_tensor = torch.tensor(values_flat, dtype=torch.float)
        label = int(self.files[idx][0])
        
        return values_tensor, label
    
    
if __name__ == '__main__':
    validation_dataset = EdgeImpulseDataset('data/EI_dataset/testing')
    for i in range(len(validation_dataset)):
        data, label = validation_dataset[i]
        assert data.shape[0] == 1800, 'data shape should be 1800'
        assert label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'label should be in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
    
    # test the accel dataset
    validation_dataset = EdgeImpulseAccelDataset('data/EI_dataset/testing')
    for i in range(len(validation_dataset)):
        data, label = validation_dataset[i]
        assert data.shape[0] == 900, 'data shape should be 900'
        assert label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'label should be in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
    
