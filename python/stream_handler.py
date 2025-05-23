import asyncio
import websockets
import json
from datetime import datetime
from collections import deque
import sys
import time
import numpy as np

# C++ feature extractor and signal generator
sys.path.append("cpp/build")
from feature_extractor_cpp import Trade, compute_features
from signal_generator import generate_signal

# Constants
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
WINDOW_SIZE = 30
LABEL_HORIZON_MS = 4000
PRICE_HISTORY_WINDOW_MS = 5000

# State
trade_window = deque(maxlen=WINDOW_SIZE)
price_log = deque()  # (timestamp_ms, price)
label_counts = {1: 0, 0: 0, -1: 0}

async def stream_trades():
    async with websockets.connect(BINANCE_WS_URL) as websocket:
        print(f"[{datetime.now()}] ✅ Connected to Binance Trade WebSocket")

        while True:
            try:
                message = await websocket.recv()
                trade = json.loads(message)

                trade_time = datetime.fromtimestamp(trade["T"] / 1000)
                price = float(trade["p"])
                quantity = float(trade["q"])
                side = "BUY" if not trade["m"] else "SELL"
                now = int(time.time() * 1000)

                print(f"[{trade_time}] {side}: {quantity} BTC @ ${price}")

                # Rolling trade + price log
                trade_obj = Trade(price, quantity, side, trade["T"])
                trade_window.append(trade_obj)
                price_log.append((now, price))
                while price_log and now - price_log[0][0] > PRICE_HISTORY_WINDOW_MS:
                    price_log.popleft()

                # Compute features + signal
                if len(trade_window) >= 2:
                    features = compute_features(list(trade_window))

                    if any(np.isnan(v) or np.isinf(v) for v in features.values()):
                        print("⚠️ Skipping due to invalid feature values\n")
                        continue

                    print(f"📊 Features: {features}")
                    signal = generate_signal(features)
                    signal_label = {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}
                    print(f"{signal_label.get(signal)} (model)\n")

                    
                    # Label training data from future price change (dynamic threshold)
                    label = None
                    recent_changes = []

                    # Compute price changes for volatility estimation
                    for i in range(len(price_log) - 1):
                        pct_change = (price_log[i+1][1] - price_log[i][1]) / price_log[i][1]
                        recent_changes.append(pct_change)

                    volatility = np.std(recent_changes) if len(recent_changes) >= 2 else 0.0002  # fallback std dev

                    for past_ts, past_price in price_log:
                        if now - past_ts >= LABEL_HORIZON_MS:
                            change = (price - past_price) / past_price
                            if change > volatility:
                                label = 1
                            elif change < -volatility:
                                label = -1
                            else:
                                label = 0
                            break

                    if label is not None:
                        record = {
                            "timestamp": int(now),
                            "features": features,
                            "predicted_signal": int(signal),
                            "label": int(label)
                        }
                        with open("labeled_data_with_preds.jsonl", "a") as f:
                            f.write(json.dumps(record) + "\n")

                        label_counts[label] += 1
                        print(f"📝 Logged labeled sample: {record}")
                        print(f"📊 Label counts: {label_counts}\n")

            except Exception as e:
                print(f"[{datetime.now()}] ❌ Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(stream_trades())






# python extractor
# import asyncio
# import websockets
# import json
# from datetime import datetime
# from collections import deque
# import sys
# import time

# # Python-based feature extractor
# from feature_extractor import FeatureExtractor
# from signal_generator import generate_signal

# # Constants
# BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
# WINDOW_SIZE = 30
# LABEL_HORIZON_MS = 3000  # 5 seconds
# PRICE_HISTORY_WINDOW_MS = 5000  # 5 seconds

# # State
# extractor = FeatureExtractor(window_size=WINDOW_SIZE)
# price_log = deque()  # (timestamp_ms, price)
# log_file = open("labeled_data.jsonl", "a")
# label_counts = {1: 0, 0: 0, -1: 0}

# async def stream_trades():
#     async with websockets.connect(BINANCE_WS_URL) as websocket:
#         print(f"[{datetime.now()}] ✅ Connected to Binance Trade WebSocket")

#         while True:
#             try:
#                 message = await websocket.recv()
#                 trade = json.loads(message)

#                 trade_time = datetime.fromtimestamp(trade["T"] / 1000)
#                 price = float(trade["p"])
#                 quantity = float(trade["q"])
#                 side = "BUY" if trade["m"] is False else "SELL"
#                 now = int(time.time() * 1000)

#                 print(f"[{trade_time}] {side}: {quantity} BTC @ ${price}")

#                 # Rolling trade + price log
#                 trade_obj = {
#                     "price": price,
#                     "quantity": quantity,
#                     "side": side,
#                     "timestamp": trade["T"]
#                 }
#                 extractor.update(trade_obj)
#                 price_log.append((now, price))
#                 while price_log and now - price_log[0][0] > PRICE_HISTORY_WINDOW_MS:
#                     price_log.popleft()

#                 # Compute features + signal
#                 features = extractor.compute_features()
#                 if features:
#                     print(f"📊 Features: {features}")

#                     signal = generate_signal(features)
#                     signal_label = {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}
#                     print(f"{signal_label.get(signal)} (model)\n")

#                     # Label training data from future price change
#                     label = None
#                     for past_ts, past_price in price_log:
#                         if now - past_ts >= LABEL_HORIZON_MS:
#                             change = (price - past_price) / past_price
#                             if change > 0.00015:
#                                 label = 1
#                             elif change < -0.00015:
#                                 label = -1
#                             else:
#                                 label = 0
#                             break

#                     # Log training data
#                     if label is not None:
#                         record = {
#                             "timestamp": int(now),
#                             "features": features,
#                             "predicted_signal": int(signal),  # what the model just predicted
#                             "label": int(label)
#                         }
#                         with open("labeled_data_with_preds.jsonl", "a") as log_file:
#                             log_file.write(json.dumps(record) + "\n")

#                         label_counts[label] += 1
#                         print(f"📝 Logged labeled sample: {record}")
#                         print(f"📊 Label counts: {label_counts}\n")

#             except Exception as e:
#                 print(f"[{datetime.now()}] ❌ Error: {e}")
#                 break

# if __name__ == "__main__":
#     asyncio.run(stream_trades())