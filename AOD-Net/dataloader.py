import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class DehazingDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.foggy_path = os.path.join(root_dir, 'foggy_images', subset)
        self.original_path = os.path.join(root_dir, 'original_images', subset)
        self.image_files = [f for f in os.listdir(self.foggy_path) if os.path.isfile(os.path.join(self.foggy_path, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        foggy_image_path = os.path.join(self.foggy_path, self.image_files[idx])
        original_image_path = os.path.join(self.original_path, self.image_files[idx])

        foggy_image = Image.open(foggy_image_path).convert('RGB')
        original_image = Image.open(original_image_path).convert('RGB')

        if self.transform:
            foggy_image = self.transform(foggy_image)
            original_image = self.transform(original_image)

        return {'foggy': foggy_image, 'original': original_image}

