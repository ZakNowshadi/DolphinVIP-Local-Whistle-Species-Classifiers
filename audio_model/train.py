import torch
import torchaudio
from torch import nn
from network import Network
from torch import optim
from dolphinwhistledataset import DolphinWhistleDataset
from torch.utils.data import DataLoader

# using tutorial: https://www.youtube.com/watch?v=MMkeLjcBTcI


# parameters that can be changed and should be decided
batch_size = 128
epochs = 10
learning_rate = 0.001
sample_rate = 22050
number_samples = 22050
device_used = "cpu"

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


    

def data_loader(training_data, batch_size):
    data_loader = DataLoader(training_data, batch_size = batch_size)
    return data_loader

ANNOTATIONS_FILE = "../Audio/labels.csv"
AUDIO_DIR = "../Audio/"

# load the data
data = DolphinWhistleDataset(ANNOTATIONS_FILE, AUDIO_DIR)
load_data = data_loader(data, batch_size)

# create the neural network
neural_network = Network().to(device_used)

# initialise loss function
loss_function = nn.CrossEntropyLoss()

# initialise optimiser 
optimiser = torch.optim.Adam(neural_network.parameters(), lr = learning_rate)

# train the model
train_model(neural_network, load_data, loss_function, optimiser, device_used, epochs)


