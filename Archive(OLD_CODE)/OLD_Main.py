# Using: https://www.tensorflow.org/tutorials/images/classification

import os
import pathlib


# Counting the number of images in the training dataset
# Need to recursively count the number of images in all subdirectories
def countNumberOfFilesInDirectoryAndSubDirectories(directory):
    numberOfImages = 0
    for item in directory.iterdir():
        if item.is_dir():
            numberOfImages += countNumberOfFilesInDirectoryAndSubDirectories(item)
        else:
            numberOfImages += 1
    return numberOfImages


# Load the training dataset
trainingDataDirectory = pathlib.Path("../Images/Training")


def getTrainingDataDirectory():
    return trainingDataDirectory


validationDataDirectory = pathlib.Path("../Images/Validation")


def getValidationDataDirectory():
    return validationDataDirectory


totalNumberOfTrainingImages = countNumberOfFilesInDirectoryAndSubDirectories(trainingDataDirectory)


class SpeciesImages:
    # Constructor containing the species name and the number of images
    def __init__(self, speciesName):
        self.speciesName = speciesName
        self.numberOfTrainingImages = countNumberOfFilesInDirectoryAndSubDirectories(
            trainingDataDirectory / speciesName)
        self.numberOfValidationImages = countNumberOfFilesInDirectoryAndSubDirectories(
            validationDataDirectory / speciesName)
        speciesSpecificTrainingDataDirectory = trainingDataDirectory / speciesName
        self.listOfTrainingImages = list(speciesSpecificTrainingDataDirectory.glob("*"))
        speciesSpecificValidationDataDirectory = validationDataDirectory / speciesName
        self.listOfValidationImages = list(speciesSpecificValidationDataDirectory.glob("*"))

        # Ensuring the number of training and validation images are correct
        assert self.numberOfTrainingImages + self.numberOfValidationImages == len(self.listOfTrainingImages) + len(
            self.listOfValidationImages)


# Making a new object for each species found in the training dataset
arrayOfAllSpeciesObjects = []
for item in trainingDataDirectory.iterdir():
    if item.is_dir():
        arrayOfAllSpeciesObjects.append(SpeciesImages(item.name))


# Making a getter for the species array such that it can be accessed from other files
def getArrayOfAllSpeciesObjects():
    return arrayOfAllSpeciesObjects


# Display the number of images in both the training and validation dataset
def printSpeciesInformation(_arrayOfAllSpeciesObjects):
    for species in arrayOfAllSpeciesObjects:
        print("Species: ", species.speciesName)
        print("Number of training images: ", species.numberOfTrainingImages)
        print("Number of validation images: ", species.numberOfValidationImages)


if __name__ == '__main__':
    printSpeciesInformation(arrayOfAllSpeciesObjects)
