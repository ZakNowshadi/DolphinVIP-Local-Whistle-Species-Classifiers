import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb
wandb.init(project="dolphin-classifier", name="resnet18-run2")

## GET TOTAL NUMBER OF IMAGES FOR EACH CLASS ##
import os

def count_images_in_folder(folder_path):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    # Count files with image extensions
    image_count = sum(1 for file in os.listdir(folder_path) 
                      if os.path.splitext(file)[1].lower() in image_extensions)
    
    return image_count

class_names = ['bottlenose', 'common', 'melon-headed']
for class_name in class_names:
    folder_path = f'../Images/Training/{class_name}'  # Replace with your folder path
    total_images = count_images_in_folder(folder_path)
    print(f"Total number of images in the training folder for {class_name}: {total_images}")

    folder_path = f'../Images/Validation/{class_name}'  # Replace with your folder path
    total_images = count_images_in_folder(folder_path)
    print(f"Total number of images in the validation folder for {class_name}: {total_images}")


# Load in the resnet18 classifier
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the last fully_connected layer
num_classes = 3
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_classes)

# # Optional: Fine-tuning
# for param in resnet18.parameters():
#     param.requires_grad = True

# Defin loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.005, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

BATCH_SIZE = 64

# Normalise the dataset
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=num_classes),
     transforms.ToTensor(),
     # Greyscale images
     transforms.Normalize((0.5,), (0.5,))])

trainingImagesDataset = torchvision.datasets.ImageFolder(root='../Images/Training', transform=transform)
trainingDataLoader = DataLoader(trainingImagesDataset, batch_size=BATCH_SIZE, shuffle=True)

# Getting the test images from the Images/Validation folder
validationImagesDataset = torchvision.datasets.ImageFolder(root='../Images/Validation', transform=transform)
validationDataLoader = DataLoader(validationImagesDataset, batch_size=BATCH_SIZE, shuffle=True)


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Move model to the device
resnet18.to(device)

# use pytoch dataset
# wandb
# tensor([0, 1, 5, 8, 1, 6, 7, 0, 0, 7]) <- why are there 9 different labels
## Train the network ##
epoch_number = 5

# wandb config
wandb.config.update({
    "learning_rate": 0.001,
    "epochs": epoch_number,
    "batch_size": BATCH_SIZE,
    "model": "ResNet18",
    "optimizer": "Adam"
})

for epoch in range(epoch_number):  # loop over the dataset multiple times
    resnet18.train()
    running_loss = 0.0
    for i, data in enumerate(trainingDataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # Move inputs and labels to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            wandb.log({"epoch": epoch, "batch": i, "training_loss": running_loss / 100})
            running_loss = 0.0

    resnet18.eval()

    correct = 0
    total = 0
    val_loss = 0.0
    resnet18.eval()  # Set model to evaluation mode
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validationDataLoader:
            images, labels = data
            # Move images and labels to the GPU
            images, labels = images.to(device), labels.to(device)
            
            outputs = resnet18(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    wandb.log({
        "epoch": epoch,
        "validation_accuracy": accuracy,
        "validation_loss": val_loss / len(validationDataLoader)
    })

    print(f'Epoch {epoch+1}/{epoch_number}, '
          f'Train Loss: {running_loss/100:.3f}, '
          f'Val Loss: {val_loss/len(validationDataLoader):.3f}, '
          f'Val Accuracy: {accuracy:.2f}%')

print('Finished Training')

wandb.finish()


# ## CHECK ACCURACY ##
# correct = 0
# total = 0
# resnet18.eval()  # Set model to evaluation mode
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in validationDataLoader:
#         images, labels = data
#         # Move images and labels to the GPU
#         images, labels = images.to(device), labels.to(device)
        
#         # calculate outputs by running images through the network
#         outputs = resnet18(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# wandb.log({"validation_accuracy": accuracy})

# wandb.finish()
# ## CALCULATE ACCURACY FOR EACH CLASS ##
# import torch
# from collections import defaultdict

# # Initialize dictionaries to keep track of correct predictions and total samples for each class
# class_correct = defaultdict(int)
# class_total = defaultdict(int)

# with torch.no_grad():
#     for data in validationDataLoader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
        
#         outputs = resnet18(images)
#         _, predicted = torch.max(outputs.data, 1)
        
#         # Compare predictions with true labels
#         c = (predicted == labels).squeeze()
        
#         # Update correct predictions and total samples for each class
#         for i in range(len(labels)):
#             label = labels[i].item()
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# # Calculate and print accuracy for each class
# classes = ['bottlenose', 'common', 'melon-headed']
# for i, class_name in enumerate(classes):
#     if class_total[i] > 0:
#         accuracy = 100 * class_correct[i] / class_total[i]
#         print(f'Accuracy of {class_name}: {accuracy:.2f}%')
#     else:
#         print(f'No samples for class {class_name}')

# # Calculate and print overall accuracy
# total_correct = sum(class_correct.values())
# total_samples = sum(class_total.values())
# overall_accuracy = 100 * total_correct / total_samples
# print(f'Overall accuracy: {overall_accuracy:.2f}%')



# ## COMPARISON WITH ACTUAL IMAGES ##
# classes = [
#     "bottlenose",
#     "common",
#     "melon-headed",
# ]

# # Total number of images in the training folder for bottlenose: 2246
# # Total number of images in the validation folder for bottlenose: 962
# # Total number of images in the training folder for common: 2134
# # Total number of images in the validation folder for common: 914
# # Total number of images in the training folder for melon-headed: 1590
# # Total number of images in the validation folder for melon-headed: 681

# for i in range(1000):
#     x, y = validationImagesDataset[i][0], validationImagesDataset[i][1]
#     with torch.no_grad():
#         x = x.unsqueeze(0).to(device)  # Add batch dimension and move to device
#         pred = resnet18(x)
#         predicted_idx = pred[0].argmax(0).item()  # Get the index of the max log-probability
#         predicted, actual = classes[predicted_idx], classes[y]
#         print(f'Image {i+1} - Predicted: "{predicted}", Actual: "{actual}"')
