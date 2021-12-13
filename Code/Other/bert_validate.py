import tensorflow_hub as hub
import tensorflow_text

bert_model = hub.load('imdb_electra')

joke = ["This movie was good, Nic Cage was awesome."]
result = bert_model(joke)
print(result)