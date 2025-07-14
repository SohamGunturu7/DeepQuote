#include "core/order.h"
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace deepquote {

// Order implementation
Order::Order(OrderId id, Side side, OrderType type, Price price, 
             Quantity quantity, const std::string& symbol, 
             const std::string& trader_id)
    : id(id), side(side), type(type), price(price), quantity(quantity),
      filled_quantity(0), status(OrderStatus::PENDING), 
      timestamp(getCurrentTimestamp()), created_time(getCurrentTimestamp()),
      symbol(symbol), trader_id(trader_id) {
}

bool Order::isValid() const {
    return isValidOrderId(id) && 
           isValidPrice(price) && 
           isValidQuantity(quantity) &&
           !symbol.empty() && 
           !trader_id.empty() &&
           filled_quantity >= 0 &&
           filled_quantity <= quantity;
}

bool Order::isFullyFilled() const {
    return status == OrderStatus::FILLED || 
           (filled_quantity >= quantity - QUANTITY_EPSILON);
}

bool Order::isPartiallyFilled() const {
    return status == OrderStatus::PARTIAL || 
           (filled_quantity > QUANTITY_EPSILON && !isFullyFilled());
}

bool Order::isActive() const {
    return status == OrderStatus::PENDING || status == OrderStatus::PARTIAL;
}

Quantity Order::getRemainingQuantity() const {
    return std::max(0.0, quantity - filled_quantity);
}

double Order::getFillRatio() const {
    return quantity > 0 ? filled_quantity / quantity : 0.0;
}

std::string Order::toString() const {
    std::ostringstream oss;
    oss << "Order{id=" << id 
        << ", side=" << enumToString(side)
        << ", type=" << enumToString(type)
        << ", price=" << price
        << ", qty=" << quantity
        << ", filled=" << filled_quantity
        << ", status=" << enumToString(status)
        << ", symbol=" << symbol
        << ", trader=" << trader_id
        << ", timestamp=" << timestampToString(timestamp)
        << "}";
    return oss.str();
}

// Trade implementation
Trade::Trade(OrderId buy_id, OrderId sell_id, Price price, 
             Quantity quantity, const std::string& symbol)
    : buy_order_id(buy_id), sell_order_id(sell_id), price(price),
      quantity(quantity), timestamp(getCurrentTimestamp()), symbol(symbol) {
}

bool Trade::isValid() const {
    return isValidOrderId(buy_order_id) && 
           isValidOrderId(sell_order_id) &&
           isValidPrice(price) && 
           isValidQuantity(quantity) &&
           !symbol.empty();
}

std::string Trade::toString() const {
    std::ostringstream oss;
    oss << "Trade{buy_id=" << buy_order_id
        << ", sell_id=" << sell_order_id
        << ", price=" << price
        << ", qty=" << quantity
        << ", symbol=" << symbol
        << ", timestamp=" << timestampToString(timestamp)
        << "}";
    return oss.str();
}

// Order book level implementation
OrderBookLevel::OrderBookLevel(Price price, Quantity quantity, int count)
    : price(price), total_quantity(quantity), order_count(count) {
}

bool OrderBookLevel::isValid() const {
    return isValidPrice(price) && 
           isValidQuantity(total_quantity) && 
           order_count > 0;
}

std::string OrderBookLevel::toString() const {
    std::ostringstream oss;
    oss << "Level{price=" << price 
        << ", qty=" << total_quantity
        << ", count=" << order_count
        << "}";
    return oss.str();
}

// Order book snapshot implementation
OrderBookSnapshot::OrderBookSnapshot(const std::string& symbol, Timestamp timestamp)
    : symbol(symbol), timestamp(timestamp), mid_price(0), spread(0), 
      bid_depth(0), ask_depth(0) {
}

bool OrderBookSnapshot::isValid() const {
    return !symbol.empty() && !bids.empty() && !asks.empty();
}

Price OrderBookSnapshot::getBestPrice(Side side) const {
    if (side == Side::BUY) {
        return bids.empty() ? 0 : bids.front().price;
    } else {
        return asks.empty() ? 0 : asks.front().price;
    }
}

Price OrderBookSnapshot::getMidPrice() const {
    Price best_bid = getBestPrice(Side::BUY);
    Price best_ask = getBestPrice(Side::SELL);
    return (best_bid + best_ask) / 2.0;
}

Price OrderBookSnapshot::getSpread() const {
    Price best_bid = getBestPrice(Side::BUY);
    Price best_ask = getBestPrice(Side::SELL);
    return best_ask - best_bid;
}

Quantity OrderBookSnapshot::getDepth(Side side) const {
    const auto& levels = (side == Side::BUY) ? bids : asks;
    Quantity depth = 0;
    for (const auto& level : levels) {
        depth += level.total_quantity;
    }
    return depth;
}

std::string OrderBookSnapshot::toString() const {
    std::ostringstream oss;
    oss << "OrderBook{symbol=" << symbol
        << ", timestamp=" << timestampToString(timestamp)
        << ", best_bid=" << getBestPrice(Side::BUY)
        << ", best_ask=" << getBestPrice(Side::SELL)
        << ", mid_price=" << getMidPrice()
        << ", spread=" << getSpread()
        << ", bid_levels=" << bids.size()
        << ", ask_levels=" << asks.size()
        << "}";
    return oss.str();
}

// Order manager implementation
OrderId OrderManager::createOrder(const Order& order) {
    if (!order.isValid()) {
        throw std::invalid_argument("Invalid order");
    }
    
    auto new_order = std::make_shared<Order>(order);
    new_order->id = next_order_id_++;
    new_order->timestamp = getCurrentTimestamp();
    new_order->created_time = getCurrentTimestamp();
    
    orders_.push_back(new_order);
    return new_order->id;
}

bool OrderManager::cancelOrder(OrderId order_id) {
    auto order = findOrderById(order_id);
    if (order && order->isActive()) {
        order->status = OrderStatus::CANCELLED;
        order->timestamp = getCurrentTimestamp();
        return true;
    }
    return false;
}

bool OrderManager::modifyOrder(OrderId order_id, Price new_price, Quantity new_quantity) {
    auto order = findOrderById(order_id);
    if (order && order->isActive() && 
        isValidPrice(new_price) && isValidQuantity(new_quantity)) {
        order->price = new_price;
        order->quantity = new_quantity;
        order->timestamp = getCurrentTimestamp();
        return true;
    }
    return false;
}

std::shared_ptr<Order> OrderManager::getOrder(OrderId order_id) const {
    return findOrderById(order_id);
}

std::vector<std::shared_ptr<Order>> OrderManager::getActiveOrders() const {
    std::vector<std::shared_ptr<Order>> active_orders;
    for (const auto& order : orders_) {
        if (order->isActive()) {
            active_orders.push_back(order);
        }
    }
    return active_orders;
}

std::vector<std::shared_ptr<Order>> OrderManager::getOrdersBySymbol(const std::string& symbol) const {
    return findOrdersBySymbol(symbol);
}

void OrderManager::processTrade(const Trade& trade) {
    trades_.push_back(trade);
}

std::vector<Trade> OrderManager::getTrades() const {
    return trades_;
}

std::shared_ptr<Order> OrderManager::findOrderById(OrderId id) const {
    for (const auto& order : orders_) {
        if (order->id == id) {
            return order;
        }
    }
    return nullptr;
}

std::vector<std::shared_ptr<Order>> OrderManager::findOrdersBySymbol(const std::string& symbol) const {
    std::vector<std::shared_ptr<Order>> symbol_orders;
    for (const auto& order : orders_) {
        if (order->symbol == symbol) {
            symbol_orders.push_back(order);
        }
    }
    return symbol_orders;
}

} // namespace deepquote 