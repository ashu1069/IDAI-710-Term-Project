import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from dehazenet import DehazeNet
from utils import ssim
from torchvision.utils import save_image
import numpy as np

# Directory and Device Setup
root_dir = '/home/stu12/s11/ak1825/idai710/Project'
save_dir = os.path.join(root_dir, 'Dehazenet_dehazed')
os.makedirs(save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Setup
model = DehazeNet().to(device)
model_path = os.path.join(root_dir, 'Dehazed-Detection/Dehaze-Net/dehazenet.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# DataLoader Setup
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
test_dataset = DehazingDataset(root_dir=root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Evaluation
loss_fn = torch.nn.MSELoss()
running_loss = 0.0
total_ssim = 0.0
num_samples = 0

with torch.no_grad():
    for data in test_loader:
        original_images = data['original_image'].to(device)
        foggy_images = data['foggy_image'].to(device)
        filenames = data['filename']

        outputs = model(foggy_images)
        replicated_outputs = outputs.repeat(1, 3, 1, 1)  # Replicate the single channel across the RGB channels
        loss = loss_fn(replicated_outputs, original_images)

        outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()  # Single channel
        original_images_np = original_images.permute(0, 2, 3, 1).cpu().numpy()
        original_images_np = np.dot(original_images_np, [0.2989, 0.5870, 0.1140])[:, :, :, None]  # Convert to grayscale

        # Calculate SSIM for each image in the batch
        batch_ssim = np.mean([
            ssim(out, orig)
            for out, orig in zip(outputs_np, original_images_np)
            if out.shape == orig.shape  # Ensure dimensions match
        ])

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