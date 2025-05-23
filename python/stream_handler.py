import asyncio
import websockets
import json
from datetime import datetime
from collections import deque
import sys
import time

# C++ and signal model
sys.path.append("cpp/build")
from feature_extractor_cpp import Trade, compute_features
from signal_generator import generate_signal

# Constants
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
WINDOW_SIZE = 30
LABEL_HORIZON_MS = 2000  # 2 seconds
PRICE_HISTORY_WINDOW_MS = 5000  # 5 seconds

# State
trade_window = deque(maxlen=WINDOW_SIZE)
price_log = deque()  # (timestamp_ms, price)
log_file = open("labeled_data.jsonl", "a")
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
                side = "BUY" if trade["m"] is False else "SELL"
                now = int(time.time() * 1000)

                print(f"[{trade_time}] {side}: {quantity} BTC @ ${price}")

                # Rolling trade + price log
                trade_window.append(Trade(price, quantity, side, trade["T"]))
                price_log.append((now, price))
                while price_log and now - price_log[0][0] > PRICE_HISTORY_WINDOW_MS:
                    price_log.popleft()

                # Compute features + signal
                if len(trade_window) >= 2:
                    features = compute_features(list(trade_window))
                    print(f"📊 Features: {features}")

                    signal = generate_signal(features)
                    signal_label = {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}
                    print(f"{signal_label.get(signal)} (model)\n")

                    # Label training data from future price change
                    label = None
                    for past_ts, past_price in price_log:
                        if now - past_ts >= LABEL_HORIZON_MS:
                            change = (price - past_price) / past_price
                            if change > 0.0003:
                                label = 1
                            elif change < -0.0003:
                                label = -1
                            else:
                                label = 0
                            break

                    # Log training data
                    if label is not None:
                        record = {
                            "timestamp": now,
                            "features": features,
                            "label": label
                        }
                        log_file.write(json.dumps(record) + "\n")
                        log_file.flush()

                        label_counts[label] += 1
                        print(f"📝 Logged labeled sample: {record}")
                        print(f"📊 Label counts: {label_counts}\n")

            except Exception as e:
                print(f"[{datetime.now()}] ❌ Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(stream_trades())