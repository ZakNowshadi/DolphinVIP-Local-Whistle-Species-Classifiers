### Whilstle Classifier

Required File Structure:

Immediately down from the root directory the following directories are not version controlled and are required for the program to run:
* `Images/` - Contains the images to be classified
  * `Training/` - Contains the training images
    * `<insert species name>/` - Contains all the training images of the specific species, will need to be created for each species
  * `Validation/` - Contains the testing images
    * `<insert species name>/` - Contains all the validation images of the specific species, will need to be created for each species
    * (The name of the species directory here should be the exact same as the name of the complementary folder in the training folder)
* `Models/` - Contains the trained models
  * `PyTorch/` - Contains the PyTorch models
  * `TensorFlow/` - Contains the TensorFlow models

Set up instructions:

1. Clone the repository
2. Make a venv using `python3 -m venv venv`
3. Activate the venv using `source venv/bin/activate`
4. Install the requirements using `pip install -r requirements.txt`
5. Run the main program by running `python Main.py`