from torch.utils.data import Dataset, DataLoader
import torchaudio

import pandas as pd
import os


class DolphinWhistleDataset(Dataset):
    
    # Constructor
    # annotations_file - FILE containing all annotations for the audio files
    # audio_dir - DIRECTORY containing all the audio files 
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        # no duplicates of audio files 
        name_set=set()
        for file in os.listdir(audio_dir):
            if file.endswith('wav'):
                name_set.add(file)
        name_set=list(name_set)
        self.datalist=name_set

    # Method to define how to use length syntax 
    # len(dolphin dataset) = ?
    # number of sample?
    def __len__(self):
        return len(self.datalist)

    # a_list[1] -> a_list.__getitem__(1) 
    # how to get items from dataset
    def __getitem__(self, index):
        # return path to audio file from database
        audio_sample_path = self._get_audio_sample_path(index)
        # label associate with this sample path
        label = self._get_audio_sample_label(index)

        # Load the audio file
        signal, sr = torchaudio.load(audio_sample_path)

        # MORE STUFF 
        # signal = signal.to(self.device)
        # signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)
        # signal = self.transformation(signal)
        return signal, label


    # TODO implement this based off database structure we use
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 1]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path
    
    # TODO implement this based off database structure we use
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 0]
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = ""
    AUDIO_DIR = ""
    SAMPLE_RATE = 0

    # TODO look more into the different settings
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64 )