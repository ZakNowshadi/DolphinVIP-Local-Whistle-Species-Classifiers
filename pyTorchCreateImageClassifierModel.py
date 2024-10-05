# Using: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

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

BATCH_SIZE = 64

# Getting the training images from the Images/Training folder

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     # Greyscale images
     transforms.Normalize((0.5,), (0.5,))])

trainingImagesDataset = torchvision.datasets.ImageFolder(root='Images/Training', transform=transform)
trainingDataLoader = DataLoader(trainingImagesDataset, batch_size=BATCH_SIZE, shuffle=True)

# Getting the test images from the Images/Test folder
testImagesDataset = torchvision.datasets.ImageFolder(root='Images/Test', transform=transform)
testDataLoader = DataLoader(testImagesDataset, batch_size=BATCH_SIZE, shuffle=True)


# Defining the model
# Using - https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#:~:text=%23%20Get%20cpu%2C%20gpu,(model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # TODO: Figure out these numbers
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
print(model)

# Model parameters

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
