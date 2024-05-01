import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from aodnet import DehazeNetAttention
from utils import ssim
import numpy as np
from torchvision.utils import save_image

def collate_fn(batch):
    batch_mod = {key: [d[key] for d in batch] for key in batch[0]}
    batch_mod['original_image'] = torch.stack(batch_mod['original_image'])
    batch_mod['foggy_image'] = torch.stack(batch_mod['foggy_image'])
    return batch_mod

original_dir = '/home/stu12/s11/ak1825/idai710/Project/original_images'
foggy_dir = '/home/stu12/s11/ak1825/idai710/Project/foggyX_images'
bb_dir = '/home/stu12/s11/ak1825/idai710/Project/BB'

save_dir = '/home/stu12/s11/ak1825/idai710/Project/Dehazed_Images'
os.makedirs(save_dir, exist_ok=True) 

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

test_dataset = DehazingDataset(original_dir, foggy_dir, bb_dir, 'test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = torch.load('/home/stu12/s11/ak1825/idai710/Project/Dehazed-Detection/AOD-NetX/aodnetX2.pt').to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

loss_fn = nn.MSELoss()
running_loss = 0.0
total_ssim = 0.0
num_samples = 0

with torch.no_grad():
    for data in test_loader:
        original_images = data['original_image'].to(device)
        foggy_images = data['foggy_image'].to(device)
        filenames = data['filename']  # Extract filenames from the dataset
        bounding_boxes = data['bboxes']

        outputs = model(foggy_images, bounding_boxes)  # Assuming the model does not need bounding boxes
        loss = loss_fn(outputs, original_images)

        outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
        original_images_np = original_images.permute(0, 2, 3, 1).cpu().numpy()

        batch_ssim = np.mean([ssim(out, orig) for out, orig in zip(outputs_np, original_images_np)])

        running_loss += loss.item() * original_images.size(0)
        total_ssim += batch_ssim * original_images.size(0)
        num_samples += original_images.size(0)

        # Save each output image using the original filename
        for output, filename in zip(outputs, filenames):
            save_path = os.path.join(save_dir, filename)  # Use the original filename
            save_image(output, save_path)

avg_loss = running_loss / num_samples
avg_ssim = total_ssim / num_samples
print(f"Average Loss: {avg_loss:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")