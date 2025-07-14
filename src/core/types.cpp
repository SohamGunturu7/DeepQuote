#include "core/types.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
using namespace std;

namespace deepquote {

// String conversions
template<>
string enumToString(Side side) {
    switch (side) {
        case Side::BUY: return "BUY";
        case Side::SELL: return "SELL";
        default: return "UNKNOWN";
    }
}

template<>
Side stringToEnum(const string& str) {
    if (str == "BUY" || str == "buy") return Side::BUY;
    if (str == "SELL" || str == "sell") return Side::SELL;
    throw invalid_argument("Invalid side: " + str);
}

template<>
string enumToString(OrderType type) {
    switch (type) {
        case OrderType::MARKET: return "MARKET";
        case OrderType::LIMIT: return "LIMIT";
        case OrderType::CANCEL: return "CANCEL";
        default: return "UNKNOWN";
    }
}

template<>
OrderType stringToEnum(const string& str) {
    if (str == "MARKET" || str == "market") return OrderType::MARKET;
    if (str == "LIMIT" || str == "limit") return OrderType::LIMIT;
    if (str == "CANCEL" || str == "cancel") return OrderType::CANCEL;
    throw invalid_argument("Invalid order type: " + str);
}

template<>
string enumToString(OrderStatus status) {
    switch (status) {
        case OrderStatus::PENDING: return "PENDING";
        case OrderStatus::PARTIAL: return "PARTIAL";
        case OrderStatus::FILLED: return "FILLED";
        case OrderStatus::CANCELLED: return "CANCELLED";
        case OrderStatus::REJECTED: return "REJECTED";
        default: return "UNKNOWN";
    }
}

// Time utilities
Timestamp getCurrentTimestamp() {
    auto now = chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return chrono::duration_cast<chrono::microseconds>(duration);
}

string timestampToString(Timestamp ts) {
    auto seconds = chrono::duration_cast<chrono::seconds>(ts);
    auto micros = ts - seconds;
    
    time_t time_t_val = static_cast<time_t>(seconds.count());
    auto tm = *gmtime(&time_t_val);
    
    ostringstream oss;
    oss << put_time(&tm, "%Y-%m-%d %H:%M:%S");
    oss << "." << setfill('0') << setw(6) << micros.count();
    
    return oss.str();
}

Timestamp stringToTimestamp(const string& str) {
    tm tm_val = {};
    istringstream ss(str);
    ss >> get_time(&tm_val, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        throw invalid_argument("Invalid timestamp format: " + str);
    }
    
    auto time_t_val = mktime(&tm_val);
    return chrono::duration_cast<Timestamp>(chrono::seconds(time_t_val));
}

} // namespace deepquote 