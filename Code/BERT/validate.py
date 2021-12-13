# Dependencies
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Getting the Models:
# Unfortunately, GitHub enforces file size limits to be less than 100 MB.
# Consequently, I am hosting the models and variables on Google Drive as a public folder.
# Include the folder in the same parent directory that houses this file, and it should run without errors.
# You can find the files here: https://drive.google.com/drive/folders/1HDwiN1V25_uOkTUApsER5mJCQ7ivOOFe?usp=sharing


# Load model of Choice
electra_base = hub.load('imdb_models\\electra_base')

# Sample Data
joke = ["This movie was good, Nic Cage was awesome.",
        "That movie should not have existed"]

# Result
result = tf.sigmoid(electra_base(joke))

# See Results
for entry in result:
    if entry < 0.5:
        print("negative")
    else:
        print("positive")
