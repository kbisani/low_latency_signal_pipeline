import pandas as pd
import joblib
import numpy as np

FEATURE_ORDER = [
    "mean_price", "price_std", "buy_sell_ratio", "trades_per_second", "price_momentum",
    "price_range", "price_skewness", "price_kurtosis", "mean_quantity", "std_quantity",
    "price_zscore", "volume_per_second", "order_flow_imbalance"
]

# Load model once at import time
model = joblib.load("models/signal_model.pkl")

# Prediction wrapper
def generate_signal(features):
    df = pd.DataFrame([features], columns=FEATURE_ORDER)
    raw_pred = model.predict(df)[0]

    # Map back to original labels
    inverse_label_map = {0: -1, 1: 0, 2: 1}
    return inverse_label_map[int(raw_pred)]