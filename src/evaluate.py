import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(actual, pred):
    """
    Computes RMSE, MAE, and Directional Accuracy for return predictions.
    """
    actual_arr = np.asarray(actual)
    pred_arr = np.asarray(pred)

    rmse = np.sqrt(mean_squared_error(actual_arr, pred_arr))
    mae = mean_absolute_error(actual_arr, pred_arr)
    da = np.mean(np.sign(actual_arr) == np.sign(pred_arr))

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Directional_Accuracy": float(da),
    }


def reconstruct_price(prev_actual_close, pred_log_return):
    """
    Reconstructs one-step-ahead price:
    Predicted_Close_t = Actual_Close_(t-1) * exp(Predicted_LogReturn_t)
    """
    prev_close = np.asarray(prev_actual_close)
    pred_ret = np.asarray(pred_log_return)
    return prev_close * np.exp(pred_ret)


def calculate_price_rmse(actual_close, predicted_close):
    """
    Computes price-space RMSE.
    """
    actual_arr = np.asarray(actual_close)
    pred_arr = np.asarray(predicted_close)
    return float(np.sqrt(mean_squared_error(actual_arr, pred_arr)))