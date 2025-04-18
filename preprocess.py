import pickle 
import numpy as np 
import pandas as pd 
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import torch

training_data_filepath = "data/train.csv"
testing_data_filepath = "data/test.csv"

# put the files in chronological order
def combine_csv(files : list[str], output_filepath):
    if not os.path.isfile(output_filepath):
        res = []
        headers =  pd.read_csv(files[0], nrows=0)
        res.append(headers)

        for filepath in files:
            df = pd.read_csv(filepath, skiprows = 0)
            rev = df.iloc[::-1].reset_index(drop = True)
            res.append(rev)
        concatenated = pd.concat(res, ignore_index=True)
        concatenated.to_csv(output_filepath, index = False)

# crashes computers
def create_batches(data, batch_size=3600 * 24):
    X, y = [], []
    for i in range(len(data) - batch_size):
        X.append(data[i:i+batch_size])
        y.append(data[i+batch_size][3])  # close price
    return np.array(X), np.array(y)

def get_data(testing=False):
    combine_csv(["data/BTC-2017min.csv", "data/BTC-2018min.csv", "data/BTC-2019min.csv"], training_data_filepath)
    combine_csv(["data/BTC-2021min.csv"], testing_data_filepath)

    train = pd.read_csv(training_data_filepath)
    test = pd.read_csv(testing_data_filepath)

    indices = [0,3,4,5,6,7,8]
    labels = np.array(test.columns)[indices]

    training_data = train[["unix", "open", "high", "low", "close", "Volume BTC", "Volume USD"]]
    testing_data = test[["unix", "open", "high", "low", "close", "Volume BTC", "Volume USD"]]

    scaler = MinMaxScaler()
    scaled_training_data = np.array(scaler.fit_transform(training_data))
    scaled_testing_data = np.array(scaler.fit_transform(testing_data))
    if testing:
        inputs = tf.reshape(scaled_testing_data, [-1, 365, 1440])
    else:
        inputs = tf.reshape(scaled_training_data, [-1, 365, 4320])
    return inputs, labels

# print(get_data(testing=False))