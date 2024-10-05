# Using: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## Hyperparameterization (parameters that must be chosen)
# - batch size
# - for conv2d(3, 6, 5) <This layer will learn 6 5x5 filters, which will be convolved over the input image to detect different features>:
#    - (6) number of output challenges/how many features we're looking at)
#    - (5) the grid we chose that might work best
# For nn.MaxPool2d(2,2): reducing feature maps (what should we have as kernel size and stride)
# For nn.Linear(16 * 5 * 5, 120)),  input size is chosen from above (we have to choose 120) - number of neurons in this layer - we have to do this for each layer
# Learn rate
# Moneum - maybe doesn't seem like its required
# Optimiser type
# Actviation function - reLu <- the one that's used in the example

# Questions to consider
# - R we doing greyscale?
# Repo to make GrayScale - https://github.com/dolphin-acoustics-vip/Generating-Datasets

import torch
import torchvision
import torchvision.transforms as transforms

# Used to normalise the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# Download in training set+loader and test set+loader
trainingImagesDataset = torchvision.datasets.CIFAR10(root='../Images/Training', train=True,
                                        download=True, transform=transform)
trainingDataLoader = torch.utils.data.DataLoader(trainingImagesDataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

validationImagesDataset = torchvision.datasets.CIFAR10(root='../Images/Validation', train=False,
                                       download=True, transform=transform)
validationDataLoader = torch.utils.data.DataLoader(validationImagesDataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('bottlenose', 'common', 'melon-headed')


# Define the neural network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define a loss function and optimizer
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)


## Train the network ##
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainingDataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
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
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')