import sqlite3
import pandas as pd
from keras import Sequential
from keras.layers import Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

con = sqlite3.connect('dataset/matches.sqlite')
match = pd.read_sql('SELECT * FROM match WHERE league_id==10257', con)
con.close()
match = match.iloc[:, :11]

kernel = match.iloc[:, [7, 8, 9, 10]]
values = kernel.values

encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])
values[:, 1] = encoder.fit_transform(values[:, 1])

X = values[:, :2]
Y = values[:, 2:]

print(X[0:10])


def baseline_model():
    model = Sequential()

    model.add(Dense(
        units=5,
        input_dim=2,
        kernel_initializer='normal',
        activation='relu'
    ))

    model.add(Dense(units=2, kernel_initializer='normal'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    return model


def model_evaluation():
    seed = 7
    np.random.seed(seed)

    estimator = KerasRegressor(
        build_fn=baseline_model,
        nb_epoch=100,
        batch_size=5,
        verbose=0
    )

    kfold = KFold(n_splits=10, random_state=seed)

    results = cross_val_score(
        estimator=estimator,
        X=X,
        y=Y,
        cv=kfold
    )

    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# model = baseline_model()

# model.fit(
#     x=X,
#     y=Y,
#     epochs=100,
#     verbose=2,
#     batch_size=8,
#     validation_split=0.2
# )
