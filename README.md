# ⚡ Low-Latency Crypto Signal Pipeline

A real-time crypto trading signal engine processing live Binance trades, built with a C++ feature extractor (via Pybind11) and Python-based ML models. Supports continuous learning, self-supervised labeling, and benchmarking of model performance.

## 🧠 Features
- Streams 2,000+ Binance trades/min via WebSocket
- C++ (Pybind11) module for fast feature computation
- 13+ engineered features (e.g., momentum, skewness, imbalance)
- XGBoost + Random Forest classification models
- Real-time self-labeling and prediction logging
- Benchmarking tools: rolling accuracy, confusion matrix, and classification report

## ⚙️ Setup Instructions

1. **Clone repo and install submodules**  
```bash
git clone https://github.com/yourusername/low_latency_signal_pipeline.git
cd low_latency_signal_pipeline
git submodule update --init --recursive
```

2. **Install Python packages**  
```bash
pip install -r requirements.txt
```

3. **Build the C++ feature extractor**
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

4. **Run the live stream handler**  
```bash
python python/stream_handler.py
```

5. **Benchmark model accuracy (optional)**  
```bash
python python/benchmark_accuracy.py
```

## 🧪 Model Training & Testing

- Train a dummy model:  
```bash
python python/train_signal_model.py
```

- Retrain using collected labeled data:  
```bash
python python/retrain_from_log.py
```

## 📈 Sample Results

- ✅ Live benchmark accuracy: ~73%
- 🔍 Features include: `price_momentum`, `volume_per_second`, `order_flow_imbalance`, `price_skewness`, etc.
- 🧠 Transformer & reinforcement learning models under exploration for adaptive trading

## 📄 Requirements
```
websockets==11.0.3  
pandas==2.2.2  
numpy==1.26.4  
scikit-learn==1.4.2  
xgboost==2.0.3  
pybind11==2.12.0  
matplotlib==3.8.4  
```

## 🛡️ License

MIT License

Inspired by real-time HFT systems, this project combines market data engineering with low-latency ML signal generation for continuous experimentation.
