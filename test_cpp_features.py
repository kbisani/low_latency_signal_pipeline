import sys
sys.path.append("cpp/build")  # Add path to compiled .so

from feature_extractor_cpp import Trade, compute_features

trades = [
    Trade(68000, 0.01, "BUY", 1720000000000),
    Trade(68010, 0.02, "SELL", 1720000001000),
]

features = compute_features(trades)
print("✅ Features from C++:", features)