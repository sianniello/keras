import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import plot_model

from utils.model_utils import json_to_model


def sentiment_encoding(val):
    if val >= 0.5:
        return 'Good'
    else:
        return 'Bad'


TOP_WORDS = 5000
MAX_REVIEW_LENGTH = 500
EMBEDDING_VECTOR_LENGTH = 32

# load the dataset and truncate and pad input sequences
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=TOP_WORDS
)

word_index = imdb.get_word_index()

sentence = 'this is excellent sentence and i like it very much. Nice work too.'
words = text_to_word_sequence(sentence)

to_predict = [[word_index[word] for word in words if word in word_index]]

print(to_predict)

to_predict = pad_sequences(
    sequences=to_predict,
    maxlen=MAX_REVIEW_LENGTH
)

to_predict = np.array(to_predict.flatten())
print(to_predict)

model = json_to_model('models/sequence_classification-1513327582/sequence_classification.json')

plot_model(model, to_file='model.png')

predicted = model.predict(to_predict.reshape((1, to_predict.shape[0])))

print(sentiment_encoding(predicted))