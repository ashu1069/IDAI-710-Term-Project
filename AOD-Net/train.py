import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from nnet import dehaze_net
from PIL import Image

root_dir = '/home/stu12/s11/ak1825/idai710/Project/'

transform = transforms.Compose([
    transforms.Resize((480, 640),interpolation=Image.Resampling.LANCZOS),
    transforms.ToTensor()
])

train_dataset = DehazingDataset(root_dir=root_dir, subset='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = DehazingDataset(root_dir=root_dir, subset='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = dehaze_net().to(device)

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# number of epochs
num_epochs = 15 

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['foggy'], data['original']

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # zero the parameter gradients

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels) 

            # backward + optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()  # set model to evaluate mode
        with torch.no_grad():
            val_running_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data['foggy'], data['original']

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels) 

                val_running_loss += loss.item() * inputs.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}')


# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

torch.save(model, '/home/stu12/s11/ak1825/idai710/Project/Dehazed-Detection/AOD-Net/aodnet.pth')