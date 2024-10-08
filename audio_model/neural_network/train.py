import torch
import torchaudio
from torch import nn
from network import network
from torch import optim

def train_epoch(neural_network, data, loss_function, optimiser, device):
    for input, target in data:
        input, target = input.to(device), target.to(device)

        prediction = neural_network(input)
        loss = loss_function(prediction, target)

        # zero the gradient buffers of all the parameters
        optimiser.zero_grad()
        # back propogate 
        loss.backward()
        optimiser.step()



def train_model(neural_network, data, loss_function, optimiser, device, num_epochs):
    for i in range(num_epochs):
        train_epoch(neural_network, data, loss_function, optimiser, device)
    print("finished")

# create the neural network
neural_network = network().to("cpu")

# initialise loss function
loss_function = nn.CrossEntropyLoss()

# initialise optimiser 
optimiser = torch.optim.Adam(neural_network.parameters(), lr = 0.001)

# train the model
train_model(neural_network, data, loss_function, optimiser, "cpu", 10)


