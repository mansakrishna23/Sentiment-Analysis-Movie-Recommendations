# Dependencies
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

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
