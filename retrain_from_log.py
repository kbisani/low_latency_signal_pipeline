import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

LOG_PATH = "labeled_data.jsonl"
MODEL_PATH = "models/signal_model.pkl"

# Load log file
def load_log_data(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                features = entry["features"]
                features["label"] = entry["label"]
                records.append(features)
            except Exception as e:
                print(f"⚠️ Skipping bad line: {e}")
    return pd.DataFrame(records)

# Load data
print("📦 Loading labeled data from log...")
data = load_log_data(LOG_PATH)
print(f"✅ Loaded {len(data)} samples")

# Define features
FEATURE_ORDER = [
    "mean_price", "price_std", "buy_sell_ratio", "trades_per_second", "price_momentum",
    "price_range", "price_skewness", "price_kurtosis", "mean_quantity", "std_quantity",
    "price_zscore", "volume_per_second", "order_flow_imbalance"
]
X = data[FEATURE_ORDER]

# Remap labels: -1 -> 0, 0 -> 1, 1 -> 2
label_map = {-1: 0, 0: 1, 1: 2}
inverse_label_map = {0: -1, 1: 0, 2: 1}
y = data["label"].map(label_map)

# Split data
train_size = int(len(data) * 0.8)
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# Train Random Forest
print("🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("\n📊 Evaluation on held-out test set:")
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred).map(inverse_label_map)
y_test_mapped = y_test.map(inverse_label_map)
print(classification_report(y_test_mapped, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Random Forest model saved to {MODEL_PATH}")

# Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(FEATURE_ORDER, importances)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()