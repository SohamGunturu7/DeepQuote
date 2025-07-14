#pragma once

#include <cstdint>
#include <string>
#include <chrono>
#include <limits>
using namespace std;

namespace deepquote {

// Basic types
using OrderId = uint64_t;
using Price = double;
using Quantity = double;
using Timestamp = chrono::microseconds;

// Enums
enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

enum class OrderType : uint8_t {
    MARKET = 0,
    LIMIT = 1,
    CANCEL = 2
};

enum class OrderStatus : uint8_t {
    PENDING = 0,
    PARTIAL = 1,
    FILLED = 2,
    CANCELLED = 3,
    REJECTED = 4
};

// Constants
constexpr Price PRICE_EPSILON = 1e-8;
constexpr Quantity QUANTITY_EPSILON = 1e-8;
constexpr OrderId INVALID_ORDER_ID = numeric_limits<OrderId>::max();

// Utility functions
inline bool isValidPrice(Price price) {
    return price > PRICE_EPSILON && isfinite(price);
}

inline bool isValidQuantity(Quantity qty) {
    return qty > QUANTITY_EPSILON && isfinite(qty);
}

inline bool isValidOrderId(OrderId id) {
    return id != INVALID_ORDER_ID;
}

// String conversions
template<typename T>
string enumToString(T value);

template<typename T>
T stringToEnum(const string& str);

// Time utilities
Timestamp getCurrentTimestamp();
string timestampToString(Timestamp ts);
Timestamp stringToTimestamp(const string& str);

} // namespace deepquote 