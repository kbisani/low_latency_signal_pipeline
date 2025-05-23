# python/feature_extractor.py

from collections import deque
from datetime import datetime
import numpy as np

class FeatureExtractor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.trades = deque(maxlen=window_size)

    def update(self, trade):
        """
        Add a new trade to the rolling window.
        trade: dict with keys 'price', 'quantity', 'side', 'timestamp'
        """
        self.trades.append(trade)

    def compute_features(self):
        if not self.trades:
            return None

        prices = np.array([float(t['price']) for t in self.trades])
        quantities = np.array([float(t['quantity']) for t in self.trades])
        buy_vol = sum(float(t['quantity']) for t in self.trades if t['side'] == "BUY")
        sell_vol = sum(float(t['quantity']) for t in self.trades if t['side'] == "SELL")
        time_deltas = np.diff([t['timestamp'] for t in self.trades])

        features = {
            "mean_price": np.mean(prices),
            "price_std": np.std(prices),
            "buy_sell_ratio": buy_vol / (sell_vol + 1e-6),
            "trades_per_second": len(self.trades) / (sum(time_deltas) / 1000 + 1e-6) if len(time_deltas) > 0 else 0,
        }

        return features