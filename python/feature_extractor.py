# This is a Python prototype of the improved feature extractor to mirror in C++ later
import numpy as np
from collections import deque
from scipy.stats import skew, kurtosis, zscore

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
        price_range = np.max(prices) - np.min(prices)
        price_skewness = skew(prices)
        price_kurt = kurtosis(prices)

        mean_quantity = np.mean(quantities)
        std_quantity = np.std(quantities)

        price_z = zscore(prices)[-1] if len(prices) > 2 else 0

        price_delta = prices[-1] - prices[0]
        price_momentum = price_delta / prices[0] if prices[0] != 0 else 0

        duration = (timestamps[-1] - timestamps[0]) / 1000  # ms to seconds
        trades_per_second = len(prices) / duration if duration > 0 else 0
        volume_per_second = np.sum(quantities) / duration if duration > 0 else 0

        order_flow_imbalance = (buy_count - sell_count) / (buy_count + sell_count + 1e-5)

        return {
            "mean_price": mean_price,
            "price_std": price_std,
            "buy_sell_ratio": buy_sell_ratio,
            "trades_per_second": trades_per_second,
            "price_momentum": price_momentum,
            "price_range": price_range,
            "price_skewness": price_skewness,
            "price_kurtosis": price_kurt,
            "mean_quantity": mean_quantity,
            "std_quantity": std_quantity,
            "price_zscore": price_z,
            "volume_per_second": volume_per_second,
            "order_flow_imbalance": order_flow_imbalance
        }