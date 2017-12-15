# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

import numpy as np
from keras.datasets import imdb
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing import sequence

from utils.model_utils import model_to_json

TOP_WORDS = 5000
MAX_REVIEW_LENGTH = 500
EMBEDDING_VECTOR_LENGTH = 32

np.random.seed(7)

# load the dataset and truncate and pad input sequences
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=TOP_WORDS)

X_train = sequence.pad_sequences(
    sequences=X_train,
    maxlen=MAX_REVIEW_LENGTH)
X_test = sequence.pad_sequences(
    sequences=X_test,
    maxlen=MAX_REVIEW_LENGTH
)

print(X_test[0])
print(y_test[0])

model = Sequential()

model.add(Embedding(
    input_dim=TOP_WORDS,
    output_dim=EMBEDDING_VECTOR_LENGTH,
    input_length=MAX_REVIEW_LENGTH
))

model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=64
)

model_to_json(model, 'sequence_classification')

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
