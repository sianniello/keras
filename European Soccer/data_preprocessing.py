import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def get_train_test():
    con = sqlite3.connect('dataset/player_attributes.sqlite')
    df = pd.read_sql('SELECT * FROM player_attributes', con)
    con.close()

    df = df.iloc[:, 4:]
    print(df.head())
    df.dropna(inplace=True)

    dataset = df.values

    encoder = LabelEncoder()
    for i in range(2, 5):
        dataset[:, i] = encoder.fit_transform(dataset[:, i])

    dataset = dataset.astype('float')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    Y = dataset[:, 2]
    X = np.delete(dataset, 2, axis=1)
    print(X)
    print(Y)

    X_train, X_test = train_test_split(X, test_size=0.2)
    y_train, y_test = train_test_split(Y, test_size=0.2)

    return X, Y
