# This is a Python prototype of the improved feature extractor to mirror in C++ later
import numpy as np
from collections import deque

class FeatureExtractor:
    def __init__(self, window_size=30):
        self.trades = deque(maxlen=window_size)

    def update(self, trade):
        self.trades.append(trade)

    def compute_features(self):
        if len(self.trades) < 2:
            return None

        prices = np.array([float(t['price']) for t in self.trades])
        quantities = np.array([float(t['quantity']) for t in self.trades])
        sides = [t['side'] for t in self.trades]
        timestamps = np.array([int(t['timestamp']) for t in self.trades])

        buy_count = sum(1 for s in sides if s == 'BUY')
        sell_count = sum(1 for s in sides if s == 'SELL')
        buy_sell_ratio = buy_count / (sell_count + 1e-5)

        mean_price = np.mean(prices)
        price_std = np.std(prices)

        price_delta = prices[-1] - prices[0]
        price_momentum = price_delta / prices[0] if prices[0] != 0 else 0

        duration = (timestamps[-1] - timestamps[0]) / 1000  # milliseconds to seconds
        trades_per_second = len(prices) / duration if duration > 0 else 0

        return {
            "mean_price": mean_price,
            "price_std": price_std,
            "buy_sell_ratio": buy_sell_ratio,
            "trades_per_second": trades_per_second,
            "price_momentum": price_momentum
        }