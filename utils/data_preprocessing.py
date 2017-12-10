from numpy import loadtxt
from os import path, makedirs
from urllib import request
from datetime import datetime
from pandas import DataFrame, concat


def load_dataset(filename, n_features, n_target):
    """
    Load dataset and split it into input and output variables

    @param filename: string
    @param n_target: number
    @param n_features: number
    @return: array, array
    """
    try:
        dataset = loadtxt('dataset/' + filename, delimiter=',')
        x = dataset[:, 0:n_features]
        y = dataset[:, n_features:n_target]
    except FileNotFoundError:
        print("Error")
        return
    return x, y


def fetch_dataset(url, filename):
    """
    Fetch dataset from remote url

    @param url:         string
    @param filename:    string
    @return:
    """
    if not path.exists('dataset'):
        makedirs('dataset')
    if not path.exists('dataset/' + filename):
        filename = 'dataset/' + filename
        print("Fetching dataset from " + url)
        request.urlretrieve(url=url, filename=filename)
        print("Dataset saved in " + filename)


def date_parsing(x, y, z):
    """

    :param z:
    :param y:
    :param x: array
    :return:
    """
    if z == '0':
        z = '1'
    if y == '0':
        y = '1'
    x = ' '.join([x, y, z])
    return datetime.strptime(x, '%Y %m %d')


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.

    :param data: Sequence of observations as a list or NumPy array.
    :param n_in: Number of lag observations as input (X).
    :param n_out: Number of observations as output (y).
    :param dropnan: Boolean whether or not to drop rows with NaN values.
    :return: Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
