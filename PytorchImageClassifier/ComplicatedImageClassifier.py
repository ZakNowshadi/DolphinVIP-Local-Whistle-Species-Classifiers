# Using: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import optim
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
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64

# Getting the training images from the Images/Training folder

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     # Greyscale images
     transforms.Normalize((0.5,), (0.5,))])

trainingImagesDataset = torchvision.datasets.ImageFolder(root='Images/Training', transform=transform)
trainingDataLoader = DataLoader(trainingImagesDataset, batch_size=BATCH_SIZE, shuffle=True)

# Getting the test images from the Images/Validation folder
validationImagesDataset = torchvision.datasets.ImageFolder(root='Images/Validation', transform=transform)
validationDataLoader = DataLoader(validationImagesDataset, batch_size=BATCH_SIZE, shuffle=True)

# Defining the model

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Using - https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#:~:text=%23%20Get%20cpu%2C%20gpu,(model)
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             # TODO: Figure out these numbers
#             nn.Linear(83426, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
#
#
# model = NeuralNetwork().to(device)
# print(model)

# Model parameters

# lossFunction = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)




# Training the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Testing the model function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
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


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

numberOfEpochs = 5
for epoch in range(numberOfEpochs):
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


# for t in range(numberOfEpochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(trainingDataLoader, model, lossFunction, optimizer)
#     test(validationDataLoader, model, lossFunction)
# print("Done!")

