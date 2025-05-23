import asyncio
import websockets
import json
from datetime import datetime
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor(window_size=30)

# Binance trade stream for BTC/USDT
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

async def stream_trades():
    async with websockets.connect(BINANCE_WS_URL) as websocket:
        print(f"[{datetime.now()}] ✅ Connected to Binance Trade WebSocket")

        while True:
            try:
                message = await websocket.recv()
                trade = json.loads(message)

                trade_time = datetime.fromtimestamp(trade['T'] / 1000)
                price = trade['p']
                quantity = trade['q']
                side = "BUY" if trade['m'] is False else "SELL"

                print(f"[{trade_time}] {side}: {quantity} BTC @ ${price}")

                trade_obj = {
                    "price": price,
                    "quantity": quantity,
                    "side": side,
                    "timestamp": trade['T']
                }
                extractor.update(trade_obj)
                
                features = extractor.compute_features()
                if features:
                    print(f"📊 Features: {features}")

            except Exception as e:
                print(f"[{datetime.now()}] ❌ Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(stream_trades())