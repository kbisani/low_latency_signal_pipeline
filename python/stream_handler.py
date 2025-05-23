import asyncio
import websockets
import json
from datetime import datetime
from collections import deque
import sys

# Add path to C++ module
sys.path.append("cpp/build")
from feature_extractor_cpp import Trade, compute_features

# Binance WebSocket
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
WINDOW_SIZE = 30
trade_window = deque(maxlen=WINDOW_SIZE)

async def stream_trades():
    async with websockets.connect(BINANCE_WS_URL) as websocket:
        print(f"[{datetime.now()}] ✅ Connected to Binance Trade WebSocket")

        while True:
            try:
                message = await websocket.recv()
                trade = json.loads(message)

                trade_time = datetime.fromtimestamp(trade['T'] / 1000)
                price = float(trade['p'])
                quantity = float(trade['q'])
                side = "BUY" if trade['m'] is False else "SELL"

                print(f"[{trade_time}] {side}: {quantity} BTC @ ${price}")

                # Add new trade to window
                trade_obj = Trade(price, quantity, side, trade['T'])
                trade_window.append(trade_obj)

                # Compute features once window is full
                if len(trade_window) >= 2:
                    features = compute_features(list(trade_window))
                    print(f"📊 Features (C++): {features}")

            except Exception as e:
                print(f"[{datetime.now()}] ❌ Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(stream_trades())