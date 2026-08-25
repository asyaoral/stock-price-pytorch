import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def evaluate_model(model, X_test, y_test, scaler):
    model.eval()

    with torch.no_grad():
        predictions = model(X_test).numpy()

    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.numpy())

    rmse = np.sqrt(
        mean_squared_error(actual_prices, predicted_prices)
    )

    return rmse, actual_prices, predicted_prices