// cpp/feature_extractor.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <map>
#include <algorithm>

namespace py = pybind11;

struct Trade {
    double price;
    double quantity;
    std::string side;
    long timestamp;
};

std::map<std::string, double> compute_features(const std::vector<Trade>& trades) {
    std::vector<double> prices, quantities;
    std::vector<long> timestamps;
    int buy_count = 0, sell_count = 0;

    for (const auto& trade : trades) {
        prices.push_back(trade.price);
        quantities.push_back(trade.quantity);
        timestamps.push_back(trade.timestamp);
        if (trade.side == "BUY") buy_count++;
        else if (trade.side == "SELL") sell_count++;
    }

    double mean_price = 0.0, price_std = 0.0, price_range = 0.0, price_momentum = 0.0;
    double mean_quantity = 0.0, std_quantity = 0.0, volume_per_second = 0.0, price_zscore = 0.0;
    double skewness = 0.0, kurt = 0.0;
    double buy_sell_ratio = (double)buy_count / (sell_count + 1e-5);
    double time_span = (timestamps.back() - timestamps.front()) / 1000.0;
    double trades_per_second = trades.size() / (time_span + 1e-5);
    double order_flow_imbalance = (buy_count - sell_count) / (double)(buy_count + sell_count + 1e-5);

    // Price stats
    if (!prices.empty()) {
        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        mean_price = sum / prices.size();

        double sq_sum = std::accumulate(prices.begin(), prices.end(), 0.0, [mean_price](double acc, double val) {
            return acc + std::pow(val - mean_price, 2);
        });
        price_std = std::sqrt(sq_sum / prices.size());

        price_range = *std::max_element(prices.begin(), prices.end()) - *std::min_element(prices.begin(), prices.end());

        if (prices.size() >= 2 && prices.front() != 0.0) {
            price_momentum = (prices.back() - prices.front()) / prices.front();
        }

        // Z-score
        if (price_std > 1e-8) {
            price_zscore = (prices.back() - mean_price) / price_std;
        }

        // Skewness
        double third_moment = std::accumulate(prices.begin(), prices.end(), 0.0, [mean_price](double acc, double val) {
            return acc + std::pow(val - mean_price, 3);
        }) / prices.size();
        skewness = (price_std > 1e-8) ? third_moment / std::pow(price_std, 3) : 0.0;

        // Kurtosis
        double fourth_moment = std::accumulate(prices.begin(), prices.end(), 0.0, [mean_price](double acc, double val) {
            return acc + std::pow(val - mean_price, 4);
        }) / prices.size();
        kurt = (price_std > 1e-8) ? (fourth_moment / std::pow(price_std, 4)) - 3.0 : 0.0;
    }

    // Quantity stats
    if (!quantities.empty()) {
        double q_sum = std::accumulate(quantities.begin(), quantities.end(), 0.0);
        mean_quantity = q_sum / quantities.size();

        double q_sq_sum = std::accumulate(quantities.begin(), quantities.end(), 0.0, [mean_quantity](double acc, double val) {
            return acc + std::pow(val - mean_quantity, 2);
        });
        std_quantity = std::sqrt(q_sq_sum / quantities.size());

        volume_per_second = q_sum / (time_span + 1e-5);
    }

    return {
        {"mean_price", mean_price},
        {"price_std", price_std},
        {"buy_sell_ratio", buy_sell_ratio},
        {"trades_per_second", trades_per_second},
        {"price_momentum", price_momentum},
        {"price_range", price_range},
        {"price_skewness", skewness},
        {"price_kurtosis", kurt},
        {"mean_quantity", mean_quantity},
        {"std_quantity", std_quantity},
        {"price_zscore", price_zscore},
        {"volume_per_second", volume_per_second},
        {"order_flow_imbalance", order_flow_imbalance}
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