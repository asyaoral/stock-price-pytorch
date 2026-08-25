import pandas as pd
import torch

from data_utils import prepare_data
from models import GRUModel
from train import train_model
from evaluate import evaluate_model


# 1. Veriyi yükle
df = pd.read_csv(
    "data/amzn_stock.csv",
    index_col="Date",
    parse_dates=True
)

close_data = df[["Close"]]


# 2. Final eğitim verisini hazırla
lookback = 20
val_end = int(len(close_data) * 0.85)

train_val_data = close_data.iloc[:val_end]

from sklearn.preprocessing import MinMaxScaler
from data_utils import create_sequences

scaler = MinMaxScaler(feature_range=(-1, 1))
train_val_scaled = scaler.fit_transform(train_val_data)

X_train, y_train = create_sequences(
    train_val_scaled,
    lookback
)

test_input = close_data.iloc[val_end - lookback:]
test_scaled = scaler.transform(test_input)

X_test, y_test = create_sequences(
    test_scaled,
    lookback
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 3. Model
torch.manual_seed(42)

model = GRUModel(
    hidden_size=64,
    num_layers=2
)


# 4. Eğit
model, losses = train_model(
    model,
    X_train,
    y_train,
    epochs=100,
    lr=0.001
)


# 5. Test et
rmse, actual, predicted = evaluate_model(
    model,
    X_test,
    y_test,
    scaler
)

print(f"Final GRU Test RMSE: ${rmse:.2f}")