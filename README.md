# Low-Latency Trading Signal Pipeline

This project implements a real-time trading signal engine that simulates high-frequency trading (HFT) speed constraints. It integrates a C++ module for low-latency data parsing and a Python-based ML model to generate trading signals from live market data.

## 📁 Project Structure

```
low_latency_signal_pipeline/
├── cpp/
│   └── feature_extractor.cpp      # C++ code for fast feature extraction
├── data/
│   └── raw_data.jsonl             # Saved Binance WebSocket data (optional)
├── models/
│   └── signal_model.pkl           # Trained Python ML model
├── python/
│   ├── stream_handler.py          # WebSocket connection and data streamer
│   ├── signal_generator.py        # ML signal generation logic
│   └── pipeline.py                # Main execution loop
├── tests/
│   └── benchmark_latency.py       # Measure latency and performance
├── .gitignore                     # Ignore virtual env and other artifacts
├── README.md
└── requirements.txt               # Python dependencies
```

## 🚀 Features
- Real-time order book streaming from Binance via WebSocket API
- C++ module (with pybind11) for fast feature extraction
- Python ML model (Logistic Regression/XGBoost) for generating trading signals
- Performance benchmarking: latency, throughput, signal accuracy

## ⚙️ Setup Instructions
1. Clone the repo and navigate to the directory:
```bash
git clone https://github.com/your-username/low_latency_signal_pipeline.git
cd low_latency_signal_pipeline
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the C++ extension:
```bash
cd cpp
mkdir build && cd build
cmake ..
make
```

4. Run the main pipeline:
```bash
python python/pipeline.py
```

## 📊 TODO
- [x] Build WebSocket streamer
- [ ] Implement C++ feature extractor
- [ ] Train signal model and serialize it
- [ ] Integrate C++ + Python pipeline
- [ ] Add latency benchmarking

## 📜 License
MIT

---

## 📦 requirements.txt
```
websockets==11.0.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
pybind11==2.12.0
```

## 📁 .gitignore
```
# Ignore virtual environments
llsh/
.venv/
venv/

# Python cache files
__pycache__/
*.py[cod]

# IDE settings
.vscode/
```
