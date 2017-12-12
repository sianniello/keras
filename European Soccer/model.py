from data_preprocessing import get_train_test
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

from utils.model_utils import model_to_json


def baseline_model():
    model = Sequential()

    model.add(Dense(
        units=19,
        input_dim=X.shape[1],
        kernel_initializer='normal',
        activation='relu'
    ))

    model.add(Dense(
        units=1,
        kernel_initializer='normal',
        activation='sigmoid'
    ))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


X, Y = get_train_test()

print(X[1], Y[1])

model = baseline_model()

seed = np.random.seed(7)

# evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# model_to_json(model, 'european_soccer')