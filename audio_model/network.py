from torch import nn
# Using the tutorial: https://www.youtube.com/watch?v=SQ1iIKs190Q

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        # Four convolutional layers for the neural network, with a ReLu activation function 
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        # flatten the dimensions
        self.flatten = nn.Flatten()

        # linear transformation into single layer
        self.linear = nn.Linear(128 * 5 * 4, 10)

        # rescales so that the elements of the n-dimensional output are between 0 and 1
        self.softmax = nn.Softmax(dim=1)

        # apply sigmoid function
       #  self.output = nn.Sigmoid()


    # One forward pass in then neural network
    def forward(self, nn_input):
        i = self.layer1(nn_input)
        i = self.layer2(i)
        i = self.layer3(i)
        i = self.layer4(i)
        i = self.flatten(i)
        nn_output = self.linear(i)
        prediction = self.softmax(nn_output)

        return prediction