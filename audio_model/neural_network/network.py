from torch import nn

class network(nn.Module):

    def __init__(self):
        super().__init__()
        # Four convolutional layers for the neural network, with a ReLu activation function 
        self.layer1 = nn.sequential(nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,stride=2,padding=1), nn.ReLU, nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.sequential(nn.Conv2d(in_channels=20,out_channels=40,kernel_size=3,stride=2,padding=1), nn.ReLU, nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.sequential(nn.Conv2d(in_channels=40,out_channels=80,kernel_size=3,stride=2,padding=1), nn.ReLU, nn.MaxPool2d(kernel_size=2))

        # flatten the dimensions
        self.flatten = nn.Flatten()

        # linear transformation into single layer
        self.linear = nn.Linear(80 * 5 * 3, 10)

        # rescales so that the elements of the n-dimensional output are between 0 and 1
        self.softmax = nn.SoftMax(dim=1)

        # apply sigmoid function
       #  self.output = nn.Sigmoid()


    # One forward pass in then neural network
    def forward(self, nn_input):
        i = self.layer1(nn_input)
        i = self.layer2(i)
        i = self.layer3(i)
        i = self.flatten(i)
        nn_output = self.linear(i)
        prediction = self.softmax(nn_output)

        return prediction