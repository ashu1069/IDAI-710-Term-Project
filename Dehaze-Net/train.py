import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DehazingDataset
from dehazenet import DehazeNet

root_dir = '/home/stu12/s11/ak1825/idai710/Project/'

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to a larger square size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters typical for pretrained models
])

train_dataset = DehazingDataset(root_dir=root_dir, subset='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

val_dataset = DehazingDataset(root_dir=root_dir, subset='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# Define the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss criterion, and optimizer
model = DehazeNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # you can adjust the learning rate

# Define the number of epochs
num_epochs = 10  # you can adjust the number of epochs

# Initialize lists to store losses
train_losses = []
val_losses = []

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
    train_losses.append(epoch_loss)  # Store the epoch training loss
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
            loss = criterion(outputs, labels)  # assuming target is the input itself

            val_running_loss += loss.item() * inputs.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)  # Store the epoch validation loss
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}')

torch.save(model, 'dehazenet.pth')