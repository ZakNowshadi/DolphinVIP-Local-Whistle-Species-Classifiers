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

import torch
import torchvision
import torchvision.transforms as transforms

