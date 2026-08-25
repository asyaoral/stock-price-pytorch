import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, lookback):
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])

    return np.array(X), np.array(y)


def prepare_data(close_data, train_ratio=0.70, val_ratio=0.15, lookback=20):
    train_end = int(len(close_data) * train_ratio)
    val_end = int(len(close_data) * (train_ratio + val_ratio))

    train_data = close_data.iloc[:train_end]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_data)

    X_train, y_train = create_sequences(train_scaled, lookback)

    val_input = close_data.iloc[train_end - lookback:val_end]
    val_scaled = scaler.transform(val_input)
    X_val, y_val = create_sequences(val_scaled, lookback)

    test_input = close_data.iloc[val_end - lookback:]
    test_scaled = scaler.transform(test_input)
    X_test, y_test = create_sequences(test_scaled, lookback)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32),
        "scaler": scaler,
    }