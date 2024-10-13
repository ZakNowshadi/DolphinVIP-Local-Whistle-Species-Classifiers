import torch
from torch import nn
from network import Network
from torch import optim
from dolphinwhistledataset import DolphinWhistleDataset
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
# using https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py


# parameters that can be changed 
batch_size = None
epochs = None
learning_rate = None
data_file = ""
audio_file = ""
sample_rate = None
number_samples = None
device_used = "cpu"


def data_loader(training_data, batch_size):
    data_loader = DataLoader(training_data, batch_size = batch_size)
    return data_loader


def objective(trial):
    neural_network = Network().to(device_used)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(neural_network.parameters(), lr=learning_rate)

    data = DolphinWhistleDataset(data_file, audio_file)
    train_loader = data_loader(data, batch_size)
    valid_loader = data_loader(data, batch_size)
    
    for epoch in range(epochs):
        neural_network.train()
        for batch_idex, (data, target) in enumerate(train_loader):
            if batch_idex * batch_size >= number_samples:
                break

            data, target = data.view(data.size(0), -1).to(device_used), target.to(device_used)

            optimizer.zero_grad()
            output = neural_network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            neural_network.eval()
            correct = 0
            with torch.no_grad():
                for batch_index, (data, target) in enumerate(valid_loader):
                    if batch_index * batch_size >= number_samples:
                        break
                    data, target = data.view(data.size(0), -1).to(device_used), target.to(device_used)
                    output = neural_network(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / min(len(valid_loader.dataset),number_samples)

            trial.report(accuracy, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    print("finished")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Num of finished trials: ", len(study.trials))
print("Num of pruned trials: ", len(pruned_trials))
print("Numb of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("Value:", trial.value)

print("Params:")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

