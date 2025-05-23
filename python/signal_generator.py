import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("models", "signal_model.pkl")
model = joblib.load(MODEL_PATH)

# Ensure strict column order
FEATURE_ORDER = ["mean_price", "price_std", "buy_sell_ratio", "trades_per_second", "price_momentum"]

def generate_signal(feature_dict):
    row = [[feature_dict.get(col, 0.0) for col in FEATURE_ORDER]]
    df = pd.DataFrame(row, columns=FEATURE_ORDER)
    return int(model.predict(df)[0])