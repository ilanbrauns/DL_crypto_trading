import numpy as np 
import pandas as pd 
import os
from sklearn.preprocessing import MinMaxScaler

# Filepaths for the training and testing datasets after combining CSVs
training_data_filepath = "data/train.csv"
testing_data_filepath = "data/test.csv"

def combine_csv(files: list[str], output_filepath):
    """
    Combines multiple CSV files into a single file and saves the result to `output_filepath` if it doesn't already exist.

    Args:
        files (list[str]): List of filepaths to CSVs.
        output_filepath: Path to save the concatenated output.
    """
    if not os.path.isfile(output_filepath):
        res = []
        for filepath in files:
            df = pd.read_csv(filepath)
            rev = df.iloc[::-1].reset_index(drop=True)
            res.append(rev)
        concatenated = pd.concat(res, ignore_index=True)
        concatenated.to_csv(output_filepath, index=False)

def create_batches(data, batch_size=144):
    """
    Creates batches of sequential data for time series prediction - training and testing labels are generated separately in get_data().

    Args:
        data: The full dataset of features (after scaling).
        batch_size: Length of each input sequence.

    Returns:
        Array: Input sequences X
    """
    X = []
    for i in range(0, len(data) - batch_size + 1, batch_size):
        X.append(data[i:i + batch_size])
    return np.array(X)

def get_data(testing=False, prediction_distance=1440):
    """
    Preprocesses and returns the training or testing dataset

    Args:
        testing: Whether we are training or testing
        downsample_factor: Factor to shrink dataset size

    Returns:
        Tuple:
            - X: Input sequences
            - y: Target values
            - feature_scaler: Scaler used to normalize feature inputs, returned to unscale for data presentation in main.py
            - start_close: Last closing price in each input sequence, returned for metrics in main.py
    """
    # Combine CSVs into one training and one testing file 
    combine_csv(["data/BTC-2017min.csv", "data/BTC-2018min.csv", "data/BTC-2019min.csv"], training_data_filepath)
    combine_csv(["data/BTC-2021min.csv"], testing_data_filepath)

    # Select appropriate file to preprocess depending on training or testing
    filepath = testing_data_filepath if testing else training_data_filepath
    df = pd.read_csv(filepath)

    # Select relevant features for model input from dataset
    features = ["open", "high", "low", "close", "Volume BTC", "Volume USD"]
    df = df[features].dropna().reset_index(drop=True)

    batch_size = 360

    # Normalize all input features to linear scale for input
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df)

    # Create input sequences from the scaled feature data
    X = create_batches(scaled_features, batch_size=batch_size)

    # Reshape X to match model input expectations: (samples, batch_size, number of features)
    X = X.reshape((X.shape[0], batch_size, len(features)))

    # Save closing prices for statistics presentation
    close_prices = df["close"].values

    # Get labels from price vector by matchign spacing to input dataset batch creation
    y_indices = [(i + 1) * batch_size - 1 + prediction_distance for i in range(len(X))]
    valid_y_indices = [i for i in y_indices if i < len(close_prices)]

    # Remove excess input data
    X = X[:len(valid_y_indices)]

    # Generate labels, which are future prices
    y_raw = close_prices[valid_y_indices]

    # Fit a separate scaler just for targets, returned to inverse transform to present final prices
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

    # Get the last close price in each input sequence to use for statistics
    start_close = [close_prices[(i + 1) * batch_size - 1] for i in range(len(valid_y_indices))]

    return X, y_scaled.astype(np.float32), target_scaler, start_close