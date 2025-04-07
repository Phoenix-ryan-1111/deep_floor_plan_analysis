import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class FloorPlanDataset(Dataset):
    def __init__(self, file_list_path, transform=None):
        """
        Args:
            file_list_path (string): Path to the text file with list of data samples
            transform (callable, optional): Optional transform to be applied
        """
        self.transform = transform
        
        # Read the file list
        with open(file_list_path, 'r') as f:
            lines = f.readlines()
        
        # Parse the lines to get all the file paths
        self.samples = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:  # At least input and wall label
                sample = {
                    'input': parts[0],
                    'wall': parts[1],
                    'close': parts[2] if len(parts) > 2 else None,
                    'rooms': parts[3] if len(parts) > 3 else None,
                    'close_wall': parts[4] if len(parts) > 4 else None
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load input image
        image = Image.open(sample['input']).convert('RGB')
        
        # Load labels - we'll use wall and rooms labels
        wall_label = Image.open(sample['wall'])
        rooms_label = Image.open(sample['rooms']) if sample['rooms'] else None

        if self.transform:
            image = self.transform(image)
            wall_label = self.transform(wall_label)
            if rooms_label:
                rooms_label = self.transform(rooms_label)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        wall_label = torch.from_numpy(np.array(wall_label)).long()
        rooms_label = torch.from_numpy(np.array(rooms_label)).long() if rooms_label else None

        return image, (rooms_label, wall_label)


def get_data_loaders(train_file, test_file, batch_size=8, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create full training dataset
    train_dataset = FloorPlanDataset(train_file, transform=transform)
    
    # Split into train and validation
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    # Create test dataset
    test_dataset = FloorPlanDataset(test_file, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


# Example usage:
# train_loader, val_loader, test_loader = get_data_loaders(
#     'r2v_train.txt', 
#     'r2v_test.txt',
#     batch_size=8
# )