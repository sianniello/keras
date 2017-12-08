from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from utils.data_preprocessing import date_parsing, series_to_supervised

dataset = read_csv(
    filepath_or_buffer='dataset/globalterrorismdb_0617dist.csv',
    sep=',',
    index_col=0,
    encoding='ISO-8859-1',
    usecols=['iyear', 'imonth', 'iday', 'country', 'country_txt', 'latitude', 'longitude', 'nkill'],
    parse_dates=[['iyear', 'imonth', 'iday']],
    date_parser=date_parsing
)

dataset.index.name = 'date'
dataset['nkill'].fillna(0, inplace=True)

encoded_country = dict(zip(dataset['country'], dataset['country_txt']))
decoded_country = dict(zip(dataset['country_txt'], dataset['country']))

dataset.drop('country_txt', inplace=True, axis=1)
dataset.dropna(inplace=True)

values = dataset.values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled)

train_split = 0.75
train_split = round(train_split * reframed.shape[0])

values = reframed.values

train = values[:train_split, :]
test = values[train_split:, :]

train_X, train_y = train[:, :4], train[:, 4:]
test_X, test_y = test[:, :4], test[:, 4:]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()

model.add(LSTM(
    units=50,
    input_shape=(train_X.shape[1], train_X.shape[2]),
))

model.add(Dense(units=4))
model.compile(loss='mae', optimizer='adam')

history = model.fit(x=train_X,
                    y=train_y,
                    epochs=50,
                    batch_size=72,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False
                    )

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

