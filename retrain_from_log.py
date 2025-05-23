import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

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

# Prepare features and remap labels for XGBoost
FEATURE_ORDER = ["mean_price", "price_std", "buy_sell_ratio", "trades_per_second", "price_momentum"]
X = data[FEATURE_ORDER]

# Remap labels: -1 -> 0, 0 -> 1, 1 -> 2
label_map = {-1: 0, 0: 1, 1: 2}
inverse_label_map = {0: -1, 1: 0, 2: 1}
y = data["label"].map(label_map)

train_size = int(len(data) * 0.8)
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# Retrain model using XGBoost
print("🔁 Retraining XGBoost model...")
model = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", use_label_encoder=False)
model.fit(X_train, y_train)

# Evaluate
print("\n📊 Evaluation on held-out test set:")
y_pred = model.predict(X_test)
y_pred = pd.Series(y_pred).map(inverse_label_map)
y_test_mapped = y_test.map(inverse_label_map)
print(classification_report(y_test_mapped, y_pred))

# Save updated model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"✅ Updated XGBoost model saved to {MODEL_PATH}")