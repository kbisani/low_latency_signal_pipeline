import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Simulate training data (you can replace with real labeled_data.jsonl later)
def generate_dummy_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        "mean_price": np.random.normal(68000, 100, n),
        "price_std": np.random.uniform(1, 10, n),
        "buy_sell_ratio": np.random.uniform(0.5, 2.0, n),
        "trades_per_second": np.random.uniform(1.0, 3.0, n),
        "price_momentum": np.random.uniform(-0.005, 0.005, n),
        "price_range": np.random.uniform(5, 50, n),
        "price_skewness": np.random.normal(0, 1, n),
        "price_kurtosis": np.random.normal(3, 1, n),
        "mean_quantity": np.random.uniform(0.001, 0.01, n),
        "std_quantity": np.random.uniform(0.0001, 0.002, n),
        "price_zscore": np.random.normal(0, 1, n),
        "volume_per_second": np.random.uniform(0.1, 1.0, n),
        "order_flow_imbalance": np.random.uniform(-1.0, 1.0, n)
    })

    # Dummy signal rule based on price_momentum and buy_sell_ratio
    conditions = [
        (data.buy_sell_ratio > 1.5) & (data.price_momentum > 0.001),
        (data.buy_sell_ratio < 0.8) & (data.price_momentum < -0.001),
    ]
    choices = [1, -1]
    data["signal"] = np.select(conditions, choices, default=0)
    return data

# Generate and split data
data = generate_dummy_data()
X = data.drop("signal", axis=1)
y = data["signal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(multi_class="multinomial", max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/signal_model.pkl")
print("✅ Model saved to models/signal_model.pkl")