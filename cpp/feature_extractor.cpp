// cpp/feature_extractor.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>

namespace py = pybind11;

struct Trade {
    double price;
    double quantity;
    std::string side;
    long timestamp;
};

std::map<std::string, double> compute_features(const std::vector<Trade>& trades) {
    std::vector<double> prices, buy_volumes, sell_volumes;
    std::vector<long> timestamps;

    for (const auto& trade : trades) {
        prices.push_back(trade.price);
        timestamps.push_back(trade.timestamp);
        if (trade.side == "BUY") buy_volumes.push_back(trade.quantity);
        else if (trade.side == "SELL") sell_volumes.push_back(trade.quantity);
    }

    double mean_price = 0.0, std_dev = 0.0, buy_total = 0.0, sell_total = 0.0;
    if (!prices.empty()) {
        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        mean_price = sum / prices.size();

        double sq_sum = std::accumulate(prices.begin(), prices.end(), 0.0,
                            [mean_price](double acc, double val) {
                                return acc + (val - mean_price) * (val - mean_price);
                            });
        std_dev = std::sqrt(sq_sum / prices.size());
    }

    buy_total = std::accumulate(buy_volumes.begin(), buy_volumes.end(), 0.0);
    sell_total = std::accumulate(sell_volumes.begin(), sell_volumes.end(), 0.0);

    double time_span = (timestamps.size() > 1) ? (timestamps.back() - timestamps.front()) / 1000.0 : 1.0;

    return {
        {"mean_price", mean_price},
        {"price_std", std_dev},
        {"buy_sell_ratio", buy_total / (sell_total + 1e-6)},
        {"trades_per_second", trades.size() / (time_span + 1e-6)}
    };
}

PYBIND11_MODULE(feature_extractor_cpp, m) {
    py::class_<Trade>(m, "Trade")
    .def(py::init<double, double, std::string, long>())
    .def_readwrite("price", &Trade::price)
    .def_readwrite("quantity", &Trade::quantity)
    .def_readwrite("side", &Trade::side)
    .def_readwrite("timestamp", &Trade::timestamp);

    m.def("compute_features", &compute_features, "Compute rolling features from trades");
}