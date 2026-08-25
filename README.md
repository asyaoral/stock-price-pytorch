# Stock Price Prediction with PyTorch

Amazon (AMZN) hisse kapanış fiyatlarını LSTM ve GRU modelleriyle tahmin eden zaman serisi projesi.

## Project Objective

- Historical AMZN stock data analysis
- Time-series preprocessing with sliding windows
- LSTM and GRU model development with PyTorch
- Hyperparameter comparison using validation data
- Final model evaluation on unseen test data

## Dataset

- Stock: Amazon (AMZN)
- Period: 2010–2025
- Target: Closing Price
- Source: Yahoo Finance via `yfinance`

## Data Split

- Train: 70%
- Validation: 15%
- Test: 15%

Time order is preserved; the dataset is not shuffled.

## Models

- LSTM
- GRU

Parameters tested included:
- Lookback window: 10, 20
- Hidden size: 32, 64

## Best Model

GRU:

- Lookback: 20
- Hidden size: 64
- Layers: 2
- Epochs: 100
- Final Test RMSE: **$10.09**

## Project Structure

stock-price-pytorch/
- data/
- notebooks/
- results/
- src/
- README.md
- requirements.txt

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

---------

Notebooks
01_data_exploration.ipynb — data preparation and initial LSTM/GRU comparison
02_model_experiments.ipynb — validation experiments and final GRU model
Note
This project is an educational machine learning experiment and is not intended for financial advice or real-world trading decisions.