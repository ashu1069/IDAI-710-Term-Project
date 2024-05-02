import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from unet import UNet  # Ensure this is your actual model's file and class name
from utils import ssim
import numpy as np
from torchvision.utils import save_image

# Directory Setup
root_dir = '/home/stu12/s11/ak1825/idai710/Project'
save_dir = os.path.join(root_dir, 'Unet_dehazed')
os.makedirs(save_dir, exist_ok=True)

# Transform Setup
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Dataset and DataLoader
test_dataset = DehazingDataset(root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_channels = 3  # Update this if your input images are not RGB
n_classes = 3 

model = UNet(n_channels, n_classes)  # Ensure the model class name matches exactly what is defined in your unet.py file
model_path = os.path.join(root_dir, 'Dehazed-Detection/U-Net_Dehaze/dcdpn_path.pth')
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()


# Loss Function
loss_fn = nn.MSELoss()
running_loss = 0.0
total_ssim = 0.0
num_samples = 0

with torch.no_grad():
    for data in test_loader:
        original_images = data['original_image'].to(device)
        foggy_images = data['foggy_image'].to(device)
        filenames = data['filename']

        outputs = model(foggy_images)
        loss = loss_fn(outputs, original_images)

        outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
        original_images_np = original_images.permute(0, 2, 3, 1).cpu().numpy()

        batch_ssim = np.mean([ssim(out, orig) for out, orig in zip(outputs_np, original_images_np)])

        running_loss += loss.item() * original_images.size(0)
        total_ssim += batch_ssim * original_images.size(0)
        num_samples += original_images.size(0)

        # Save dehazed images
        for output, filename in zip(outputs, filenames):
            save_path = os.path.join(save_dir, filename)
            save_image(output, save_path)

avg_loss = running_loss / num_samples
avg_ssim = total_ssim / num_samples
print(f"Average Loss: {avg_loss:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
