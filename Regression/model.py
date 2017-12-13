from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.data_preprocessing import fetch_dataset

from pandas import read_csv
from numpy.random import seed

SEED = 7
EPOCHS = 50
BATCH_SIZE = 5

# Data preparation
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
filename = 'housing.csv'
fetch_dataset(url, filename)

dataframe = read_csv(
    filepath_or_buffer='dataset/' + filename,
    delim_whitespace=True,
    header=None
)

dataset = dataframe.values

X = dataset[:, :-1]
Y = dataset[:, -1]


# Model definition
def baseline_model():
    model = Sequential()
    model.add(Dense(
        units=13,
        input_dim=13,
        kernel_initializer='normal',
        activation='relu',
    ))
    model.add(Dense(
        units=1,
        kernel_initializer='normal'
    ))
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    return model


def larger_model():
    model = Sequential()
    model.add(Dense(
        units=13,
        input_dim=13,
        kernel_initializer='normal',
        activation='relu'
    ))
    model.add(Dense(
        units=6,
        kernel_initializer='normal',
        activation='relu'
    ))
    model.add(Dense(
        units=1,
        kernel_initializer='normal'
    ))
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    return model


def wider_model():
    model = Sequential()
    model.add(Dense(
        units=20,
        input_dim=13,
        kernel_initializer='normal',
        activation='relu'
    ))
    model.add(Dense(
        units=1,
        kernel_initializer='normal'
    ))
    model.compile(
        loss='mean_squared_error',
        optimizer='adam')
    return model


# Model evaluation
seed(SEED)
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

print("Baseline results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# Standardized dataset
seed(SEED)
kfold = KFold(n_splits=10, random_state=seed)
estimators = []
estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(
    build_fn=baseline_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)))

pipeline = Pipeline(estimators)

results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized dataset results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# Larger model
seed(SEED)
estimators = []
estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(
    build_fn=larger_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)))

pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger model results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# Wider model
seed(SEED)
estimators = []
estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(
    build_fn=wider_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)))

pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider model results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
