import pandas as pd
import joblib
import numpy as np

FEATURE_ORDER = [
    "mean_price", "price_std", "buy_sell_ratio", "trades_per_second", "price_momentum",
    "price_range", "price_skewness", "price_kurtosis", "mean_quantity", "std_quantity",
    "price_zscore", "volume_per_second", "order_flow_imbalance"
]

# Load the Random Forest model once (on module import)
model = joblib.load("models/signal_model.pkl")

# Remap RandomForest prediction class -> original label
inverse_label_map = {0: -1, 1: 0, 2: 1}

def generate_signal(features: dict) -> int:
    """ Given a feature dict, predict the trading signal. """
    x = np.array([[features[f] for f in FEATURE_ORDER]])
    raw_pred = model.predict(x)[0]
    return inverse_label_map[int(raw_pred)]