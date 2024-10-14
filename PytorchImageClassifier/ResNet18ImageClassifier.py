import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

# Used to normalise the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
# Prepare data
batch_size = 64
trainingImagesDataset = datasets.CIFAR10(root='../Images/Training', train=True,
                                        download=True, transform=ToTensor())
trainingDataLoader = DataLoader(trainingImagesDataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

validationImagesDataset = datasets.CIFAR10(root='../Images/Validation', train=False,
                                       download=True, transform=ToTensor())
validationDataLoader = DataLoader(validationImagesDataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
         
## Train the network ##
epoch_number = 10
for epoch in range(epoch_number):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainingDataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

## CHECK ACCURACY ##
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validationDataLoader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = resnet18(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# Training loop
#for epoch in range(num_epochs):
#    for inputs, labels in train_loader:
#        optimizer.zero_grad()
#        outputs = resnet18(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()

