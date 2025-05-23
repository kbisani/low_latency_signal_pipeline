import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

LOG_PATH = "labeled_data_with_preds.jsonl"

# Load records
records = []
with open(LOG_PATH, "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            records.append(entry)
        except Exception as e:
            print(f"⚠️ Skipping bad line: {e}")

df = pd.DataFrame(records)
print(f"✅ Loaded {len(df)} labeled predictions")

# Calculate prediction correctness
df["correct"] = df["label"] == df["predicted_signal"]
df["rolling_accuracy"] = df["correct"].rolling(window=200).mean()

# --- PLOT 1: Rolling Accuracy ---
plt.figure(figsize=(10, 4))
plt.plot(df["rolling_accuracy"], label="Rolling Accuracy (window=200)")
plt.ylim(0, 1)
plt.xlabel("Sample Index")
plt.ylabel("Accuracy")
plt.title("📈 Rolling Prediction Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- PLOT 2: Confusion Matrix ---
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(df["label"], df["predicted_signal"], labels=[-1, 0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SELL (-1)", "HOLD (0)", "BUY (1)"])
disp.plot(cmap="Blues")
plt.title("🔢 Confusion Matrix")
plt.tight_layout()
plt.show()

# --- METRICS ---
print("\n📊 Classification Report:")
print(classification_report(df["label"], df["predicted_signal"], digits=3))

print(f"\n🎯 Overall Accuracy: {accuracy_score(df['label'], df['predicted_signal']):.4f}")