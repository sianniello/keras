from keras import Sequential
from data_preprocessing import get_train_test
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np


def baseline_model():
    model = Sequential()

    model.add(Dense(
        units=X.shape[1],
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

print(X.shape)

model = baseline_model()

seed = np.random.seed(7)

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=100, batch_size=32, verbose=2)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))