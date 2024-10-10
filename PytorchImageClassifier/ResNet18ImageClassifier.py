import torch 
import torch.nn as nn
import torchvision.models as models

# Load in the resnet18 classifier
resnet18 = models.resnet18(pretrained = True)

# Replace the last fully_connected layer
num_classes = 3
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_classes)

# Optional: Fine-tuning
for param in resnet18.parameters():
    param.requires_grad = True

# Defin loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)


# Prepare data
trainingImagesDataset = torchvision.datasets.CIFAR10(root='../Images/Training', train=True,
                                        download=True, transform=transform)
trainingDataLoader = torch.utils.data.DataLoader(trainingImagesDataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
                    
# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

