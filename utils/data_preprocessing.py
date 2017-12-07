from numpy import loadtxt
from os import path, makedirs
from urllib import request


def load_dataset(filename):
    """
    Load dataset and split it into input and output variables

    @param filename: string
    @return: array, array
    """
    dataset = loadtxt('dataset/' + filename, delimiter=',')
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    return X, Y


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




