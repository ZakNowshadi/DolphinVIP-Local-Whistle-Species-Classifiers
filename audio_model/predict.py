import torch
import torchaudio
from network import Network
from dolphinwhistledataset import DolphinWhistleDataset
from torch.utils.data import DataLoader

from train import AUDIO_DIR, ANNOTATIONS_FILE, sample_rate, number_samples


class_model = [
        'bottlenose/',
        'common/',
        'melon-headed/'
]

def predict(model, input, target, class_model):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        print(predictions)
        print(torch.argmax(predictions[0]))
        predicted_index = torch.argmax(predictions[0])
        predicted = class_model[predicted_index]
        expected = class_model[target]
    return predicted, expected


neural_network = Network()
state_dict = torch.load("neural_network.pth", weights_only=True)
neural_network.load_state_dict(state_dict)

    # load urban sound dataset dataset
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

data = DolphinWhistleDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,sample_rate, number_samples, "cpu")
for x in range(0,11):

    input, target = data[x][0], data[x][1]
    input.unsqueeze_(0)

    predicted, expected = predict(neural_network, input, target,
                                    class_model)
    print(f"Predicted: '{predicted}', expected: '{expected}'")