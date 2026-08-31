import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from data_utils import prepare_log_return_data
from evaluate import calculate_metrics, reconstruct_price, calculate_price_rmse
from models import LSTMModel, GRUModel
from train import train_model


def run_pipeline():
    print("=" * 80)
    print("STOCK PRICE FORECASTING PIPELINE (LOG RETURN EXPERIMENT)")
    print("=" * 80)

    # 1. Prepare data
    data_dict = prepare_log_return_data(
        csv_path="data/amzn_stock.csv",
        train_ratio=0.70,
        val_ratio=0.15,
        lookback=20,
        batch_size=32,
        seed=42,
    )

    close = data_dict["close"]
    log_ret = data_dict["log_ret"]
    train_data = data_dict["train_data"]
    val_data = data_dict["val_data"]
    test_data = data_dict["test_data"]

    train_loader = data_dict["train_loader"]
    val_X = data_dict["val_X"]
    val_y = data_dict["val_y"]
    test_X = data_dict["test_X"]
    test_y = data_dict["test_y"]

    train_end = data_dict["train_end"]
    val_end = data_dict["val_end"]

    N_test = len(test_data)
    N_val = len(val_data)

    print(f"Total samples: {len(log_ret)} | Train: {len(train_data)} | Val: {N_val} | Test (N): {N_test}")

    # 2. Baselines
    # Baseline 1: Zero Return
    zero_val_pred = np.zeros(N_val)
    zero_test_pred = np.zeros(N_test)

    # Baseline 2: Previous Return (r_t = r_{t-1})
    prev_val_pred = log_ret.shift(1).iloc[train_end:val_end].values
    prev_test_pred = log_ret.shift(1).iloc[val_end:].values

    # Baseline 3: Linear Regression (Lags 1, 2, 3, 5, 10)
    lags = [1, 2, 3, 5, 10]
    df_lags = pd.DataFrame(index=log_ret.index)
    for l in lags:
        df_lags[f"lag_{l}"] = log_ret.shift(l)

    X_train_lr = df_lags.iloc[max(lags) : train_end]
    y_train_lr = log_ret.iloc[max(lags) : train_end]
    X_val_lr = df_lags.iloc[train_end:val_end]
    y_val_lr = log_ret.iloc[train_end:val_end]
    X_test_lr = df_lags.iloc[val_end:]
    y_test_lr = log_ret.iloc[val_end:]

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)
    lr_val_pred = lr_model.predict(X_val_lr)
    lr_test_pred = lr_model.predict(X_test_lr)

    # 3. Train Deep Learning Models
    print("\nTraining LSTM Model...")
    torch.manual_seed(42)
    lstm_model = LSTMModel(input_size=1, hidden_size=32, num_layers=2)
    lstm_model, lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, train_loader, val_X, val_y, epochs=100, lr=0.001, patience=5
    )

    print("Training GRU Model...")
    torch.manual_seed(42)
    gru_model = GRUModel(input_size=1, hidden_size=32, num_layers=2)
    gru_model, gru_train_losses, gru_val_losses = train_model(
        gru_model, train_loader, val_X, val_y, epochs=100, lr=0.001, patience=5
    )

    # 4. Predictions
    with torch.no_grad():
        lstm_val_pred = lstm_model(val_X).numpy()
        lstm_test_pred = lstm_model(test_X).numpy()
        gru_val_pred = gru_model(val_X).numpy()
        gru_test_pred = gru_model(test_X).numpy()

    # 5. Evaluate Log Return Metrics
    models = {
        "Zero Return": (zero_val_pred, zero_test_pred),
        "Previous Return": (prev_val_pred, prev_test_pred),
        "Linear Regression": (lr_val_pred, lr_test_pred),
        "LSTM": (lstm_val_pred, lstm_test_pred),
        "GRU": (gru_val_pred, gru_test_pred),
    }

    y_val_actual = val_data.values
    y_test_actual = test_data.values

    comparison_records = []
    for name, (vp, tp) in models.items():
        v_metrics = calculate_metrics(y_val_actual, vp)
        t_metrics = calculate_metrics(y_test_actual, tp)
        comparison_records.append({
            "Model": name,
            "N": N_test,
            "Validation_RMSE": v_metrics["RMSE"],
            "Test_RMSE": t_metrics["RMSE"],
            "Test_MAE": t_metrics["MAE"],
            "Directional_Accuracy": (
                t_metrics["Directional_Accuracy"] if name != "Zero Return" else np.nan
            ),
        })

    results_df = pd.DataFrame(comparison_records)

    # 6. Price-Space Evaluation
    test_actual_close_prev = close.shift(1).loc[test_data.index].values
    actual_test_close = close.loc[test_data.index].values

    lstm_price_pred = reconstruct_price(test_actual_close_prev, lstm_test_pred)
    gru_price_pred = reconstruct_price(test_actual_close_prev, gru_test_pred)
    naive_price_pred = test_actual_close_prev

    lstm_price_rmse = calculate_price_rmse(actual_test_close, lstm_price_pred)
    gru_price_rmse = calculate_price_rmse(actual_test_close, gru_price_pred)
    naive_price_rmse = calculate_price_rmse(actual_test_close, naive_price_pred)

    price_records = [
        {"Model": "Naive", "Price_RMSE": naive_price_rmse},
        {"Model": "LSTM", "Price_RMSE": lstm_price_rmse},
        {"Model": "GRU", "Price_RMSE": gru_price_rmse},
    ]
    price_df = pd.DataFrame(price_records)

    # 7. Print Outputs
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON (LOG RETURN SPACE)")
    print("=" * 80)
    formatted_df = results_df.copy()
    formatted_df["Validation_RMSE"] = formatted_df["Validation_RMSE"].map(lambda x: f"{x:.5f}")
    formatted_df["Test_RMSE"] = formatted_df["Test_RMSE"].map(lambda x: f"{x:.5f}")
    formatted_df["Test_MAE"] = formatted_df["Test_MAE"].map(lambda x: f"{x:.5f}")
    formatted_df["Directional_Accuracy"] = formatted_df["Directional_Accuracy"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
    )
    print(formatted_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("FINAL PRICE-SPACE COMPARISON (ONE-STEP-AHEAD PRICE RMSE)")
    print("=" * 80)
    formatted_price_df = price_df.copy()
    formatted_price_df["Price_RMSE"] = formatted_price_df["Price_RMSE"].map(lambda x: f"{x:.6f}")
    print(formatted_price_df.to_string(index=False))

    # 8. Save artifacts to results/
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/final_model_comparison.csv", index=False)
    price_df.to_csv("results/final_price_comparison.csv", index=False)

    # Plot 1: Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lstm_train_losses, label="Train Loss", color="royalblue")
    axes[0].plot(lstm_val_losses, label="Validation Loss", color="orange")
    axes[0].set_title("LSTM Learning Curve (Log Return)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gru_train_losses, label="Train Loss", color="royalblue")
    axes[1].plot(gru_val_losses, label="Validation Loss", color="orange")
    axes[1].set_title("GRU Learning Curve (Log Return)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=300)
    plt.close()

    # Plot 2: Prediction Comparison (Test Price series)
    test_dates = test_data.index
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, actual_test_close, label="Actual Price", color="black", linewidth=1.5)
    plt.plot(test_dates, naive_price_pred, label=f"Naive Baseline (RMSE: ${naive_price_rmse:.4f})", color="gray", linestyle="--", alpha=0.7)
    plt.plot(test_dates, lstm_price_pred, label=f"LSTM Prediction (RMSE: ${lstm_price_rmse:.4f})", color="blue", alpha=0.7)
    plt.plot(test_dates, gru_price_pred, label=f"GRU Prediction (RMSE: ${gru_price_rmse:.4f})", color="green", alpha=0.7)
    plt.title("One-Step-Ahead Price Predictions vs Actual Price (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/final_prediction_comparison.png", dpi=300)
    plt.close()

    print("\nResults and figures saved to results/ directory successfully.")


if __name__ == "__main__":
    run_pipeline()