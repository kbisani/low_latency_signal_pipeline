# python/stream_handler.py

import asyncio
import websockets
import json
import datetime

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth"

async def stream_order_book():
    async with websockets.connect(BINANCE_WS_URL) as ws:
        print(f"[{datetime.datetime.now()}] Connected to Binance WebSocket...")
        while True:
            try:
                data = await ws.recv()
                parsed = json.loads(data)

                # Extract top of book
                bids = parsed.get("bids", [])
                asks = parsed.get("asks", [])
                if bids and asks:
                    best_bid = bids[0]
                    best_ask = asks[0]
                    print(f"Bid: {best_bid[0]} x {best_bid[1]}, Ask: {best_ask[0]} x {best_ask[1]}")

            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(stream_order_book())