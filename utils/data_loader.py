import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np


class FloorPlanDataset(Dataset):

    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(self.root_dir) if f.endswith('_input.jpg')
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        base_name = self.image_files[idx].replace('_input.jpg', '')

        # Load input image
        img_path = os.path.join(self.root_dir, f"{base_name}_input.jpg")
        image = Image.open(img_path).convert('RGB')

        # Load labels - assuming we have room and boundary labels
        room_label_path = os.path.join(self.root_dir, f"{base_name}_rooms.png")
        boundary_label_path = os.path.join(self.root_dir,
                                           f"{base_name}_wall.png")

        room_label = Image.open(room_label_path)
        boundary_label = Image.open(boundary_label_path)

        if self.transform:
            image = self.transform(image)
            room_label = self.transform(room_label)
            boundary_label = self.transform(boundary_label)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        room_label = torch.from_numpy(np.array(room_label)).long()
        boundary_label = torch.from_numpy(np.array(boundary_label)).long()

        return image, (room_label, boundary_label)


def get_data_loaders(root_dir, batch_size=8, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = FloorPlanDataset(root_dir, transform=transform)

    # Split into train and validation
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

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

    return train_loader, val_loader
