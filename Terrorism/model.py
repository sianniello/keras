from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from utils.data_preprocessing import date_parsing, series_to_supervised
from utils.model_utils import model_to_json, json_to_model

import numpy as np


def predict(model, val):
    s = MinMaxScaler(feature_range=(0, 1))
    values_scaled = s.fit_transform(val)
    values_scaled = values_scaled.reshape((values_scaled.shape[0], 1, values_scaled.shape[1]))
    y = model.predict(values_scaled)
    p = np.concatenate((y, values_scaled.reshape((1, 2))), axis=1)
    print(p)
    print(scaler.inverse_transform(p))


dataset = read_csv(
    filepath_or_buffer='dataset/globalterrorismdb_0617dist.csv',
    sep=',',
    index_col=0,
    encoding='ISO-8859-1',
    usecols=['iyear', 'imonth', 'iday', 'latitude', 'longitude', 'nkill'],
    parse_dates=[['iyear', 'imonth', 'iday']],
    date_parser=date_parsing
)

dataset.index.name = 'date'
dataset['nkill'].fillna(0, inplace=True)

dataset.dropna(inplace=True)

print(dataset.head())

values = dataset.values

# pyplot.figure()
# pyplot.plot(values[:, 2])
# pyplot.show()

values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# reframed = series_to_supervised(scaled)

train_split = 0.75
train_split = round(train_split * scaled.shape[0])

train = scaled[:train_split, :]
test = scaled[train_split:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()

# model.add(LSTM(
#     units=50,
#     input_shape=(train_X.shape[1], train_X.shape[2]),
# ))

# model.add(Dense(units=1))
# model.compile(loss='mae', optimizer='adam')

# history = model.fit(x=train_X,
#                     y=train_y,
#                     epochs=50,
#                     batch_size=72,
#                     validation_data=(test_X, test_y),
#                     verbose=2,
#                     shuffle=False
#                     )

# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# model_to_json(model, 'global_terrorism')

model = json_to_model('models/global_terrorism-1512835160/global_terrorism.json')

pre = np.array([[37.229468, 15.221124]])

predict(model, pre)