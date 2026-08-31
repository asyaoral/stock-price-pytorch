import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(csv_path="data/amzn_stock.csv"):
    """
    Loads stock data, computes daily log returns, and cleans NaNs.
    """
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    close = df["Close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    return df, close, log_ret


def create_sequences(data_array, lookback=20):
    """
    Constructs sliding window input sequences and next-step targets.
    """
    X, y = [], []
    for i in range(len(data_array) - lookback):
        X.append(data_array[i : i + lookback])
        y.append(data_array[i + lookback])
    return np.array(X), np.array(y)


def prepare_log_return_data(
    csv_path="data/amzn_stock.csv",
    train_ratio=0.70,
    val_ratio=0.15,
    lookback=20,
    batch_size=32,
    seed=42,
):
    """
    Splits log return series chronologically (70% train, 15% val, 15% test)
    and prepares PyTorch tensors and DataLoaders without future leakage.
    """
    df, close, log_ret = load_data(csv_path)

    N_total = len(log_ret)
    train_end = int(N_total * train_ratio)
    val_end = int(N_total * (train_ratio + val_ratio))

    train_data = log_ret.iloc[:train_end]
    val_data = log_ret.iloc[train_end:val_end]
    test_data = log_ret.iloc[val_end:]

    # Deep learning sequence generation with proper historical window overlaps
    X_train_dl, y_train_dl = create_sequences(train_data.values, lookback)
    val_input = log_ret.iloc[train_end - lookback : val_end].values
    X_val_dl, y_val_dl = create_sequences(val_input, lookback)
    test_input = log_ret.iloc[val_end - lookback :].values
    X_test_dl, y_test_dl = create_sequences(test_input, lookback)

    # Tensor conversion
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = TensorDataset(
        torch.tensor(X_train_dl, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train_dl, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_X_t = torch.tensor(X_val_dl, dtype=torch.float32).unsqueeze(-1)
    val_y_t = torch.tensor(y_val_dl, dtype=torch.float32)
    test_X_t = torch.tensor(X_test_dl, dtype=torch.float32).unsqueeze(-1)
    test_y_t = torch.tensor(y_test_dl, dtype=torch.float32)

    return {
        "df": df,
        "close": close,
        "log_ret": log_ret,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_loader": train_loader,
        "val_X": val_X_t,
        "val_y": val_y_t,
        "test_X": test_X_t,
        "test_y": test_y_t,
        "train_end": train_end,
        "val_end": val_end,
        "lookback": lookback,
    }