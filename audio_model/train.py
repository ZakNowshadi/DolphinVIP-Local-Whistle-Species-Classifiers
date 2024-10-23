import torch
import torchaudio
from torch import nn, t
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


def train_epoch(neural_network, data, loss_function, optimiser, device):
   
    for input, target in data:
        input = input.to(device)

        target = target.to(device)

        prediction = neural_network(input)
        loss = loss_function(prediction, target)

        # zero the gradient buffers of all the parameters
        optimiser.zero_grad()
        # back propogate 
        loss.backward()
        # dprint(f"training an epoch")
        optimiser.step()
    # print(f"loss: {loss.item()}")



def train_model(neural_network, data, loss_function, optimiser, device, num_epochs):
    print(data)
    for i in range(num_epochs):
        train_epoch(neural_network, data, loss_function, optimiser, device)
    print("finished")


  

def data_loader(training_data, batch_size):
    data_loader = DataLoader(dataset=training_data, batch_size = batch_size, shuffle=True,num_workers=2)
    return data_loader

ANNOTATIONS_FILE = "../Audio/labels.csv"
AUDIO_DIR = "../Audio/"

if torch.cuda.is_available():
    device = "cuda"
else:
     device = "cpu"
print(f"Using {device}")


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
)


# load the data
data = DolphinWhistleDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,sample_rate, number_samples, device)
load_data = data_loader(data, batch_size)
print(data)
# create the neural network
neural_network = Network().to(device)
print(neural_network)


# initialise loss function
loss_function = nn.CrossEntropyLoss()

# initialise optimiser 
optimiser = torch.optim.Adam(neural_network.parameters(), lr = learning_rate)

# train the model
train_model(neural_network, load_data, loss_function, optimiser, device, epochs)
torch.save(neural_network.state_dict(), "neural_network.pth")

