#include "market/order_book.h"
#include <algorithm>
#include <stdexcept>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

OrderBook::OrderBook(const string& symbol) : symbol_(symbol) {
}

// ============================================================================
// Order Management
// ============================================================================

bool OrderBook::addOrder(shared_ptr<Order> order) {
    if (!isValidOrder(order)) {
        return false;
    }
    
    // Check if order already exists
    if (orders_by_id_.find(order->id) != orders_by_id_.end()) {
        return false;
    }
    
    // Add to order book structure
    addOrderToPriceLevel(order);
    
    // Add to fast lookup map
    orders_by_id_[order->id] = order;
    
    return true;
}

bool OrderBook::cancelOrder(OrderId order_id) {
    auto order = getOrder(order_id);
    if (!order || !order->isActive()) {
        return false;
    }
    
    // Remove from price level
    removeOrderFromPriceLevel(order_id, order->price, order->side);
    
    // Remove from fast lookup map
    orders_by_id_.erase(order_id);
    
    // Update order status
    order->status = OrderStatus::CANCELLED;
    order->timestamp = getCurrentTimestamp();
    
    return true;
}

bool OrderBook::modifyOrder(OrderId order_id, Price new_price, Quantity new_quantity) {
    auto order = getOrder(order_id);
    if (!order || !order->isActive()) {
        return false;
    }
    
    // Check if new quantity is valid
    if (new_quantity < order->filled_quantity) {
        return false;
    }
    
    // Remove from current price level
    removeOrderFromPriceLevel(order_id, order->price, order->side);
    
    // Update order
    order->price = new_price;
    order->quantity = new_quantity;
    order->timestamp = getCurrentTimestamp();
    
    // Add to new price level
    addOrderToPriceLevel(order);
    
    return true;
}

// ============================================================================
// Order Queries
// ============================================================================

shared_ptr<Order> OrderBook::getOrder(OrderId order_id) const {
    auto it = orders_by_id_.find(order_id);
    return it != orders_by_id_.end() ? it->second : nullptr;
}

vector<shared_ptr<Order>> OrderBook::getOrdersAtPrice(Price price, Side side) const {
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            return it->second;
        }
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            return it->second;
        }
    }
    return {};
}

vector<shared_ptr<Order>> OrderBook::getActiveOrders() const {
    vector<shared_ptr<Order>> active_orders;
    for (const auto& pair : orders_by_id_) {
        if (pair.second->isActive()) {
            active_orders.push_back(pair.second);
        }
    }
    return active_orders;
}

// ============================================================================
// Market Data
// ============================================================================

Price OrderBook::getBestBid() const {
    return bids_.empty() ? 0.0 : bids_.begin()->first;
}

Price OrderBook::getBestAsk() const {
    return asks_.empty() ? 0.0 : asks_.begin()->first;
}

Price OrderBook::getMidPrice() const {
    Price best_bid = getBestBid();
    Price best_ask = getBestAsk();
    if (best_bid > 0 && best_ask > 0) {
        return (best_bid + best_ask) / 2.0;
    }
    return std::numeric_limits<Price>::quiet_NaN();
}

Price OrderBook::getSpread() const {
    Price best_bid = getBestBid();
    Price best_ask = getBestAsk();
    
    if (best_bid > 0 && best_ask > 0) {
        return best_ask - best_bid;
    }
    return 0.0;
}

Quantity OrderBook::getBidDepth() const {
    Quantity depth = 0;
    for (const auto& level : bids_) {
        for (const auto& order : level.second) {
            depth += order->getRemainingQuantity();
        }
    }
    return depth;
}

Quantity OrderBook::getAskDepth() const {
    Quantity depth = 0;
    for (const auto& level : asks_) {
        for (const auto& order : level.second) {
            depth += order->getRemainingQuantity();
        }
    }
    return depth;
}

Quantity OrderBook::getDepthAtPrice(Price price, Side side) const {
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            Quantity depth = 0;
            for (const auto& order : it->second) {
                depth += order->getRemainingQuantity();
            }
            return depth;
        }
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            Quantity depth = 0;
            for (const auto& order : it->second) {
                depth += order->getRemainingQuantity();
            }
            return depth;
        }
    }
    return 0;
}

// ============================================================================
// Order Book Snapshots
// ============================================================================

OrderBookSnapshot OrderBook::getSnapshot() const {
    OrderBookSnapshot snapshot(symbol_, getCurrentTimestamp());
    snapshot.bids = getBidLevels();
    snapshot.asks = getAskLevels();
    snapshot.mid_price = getMidPrice();
    snapshot.spread = getSpread();
    snapshot.bid_depth = getBidDepth();
    snapshot.ask_depth = getAskDepth();
    return snapshot;
}

vector<OrderBookLevel> OrderBook::getBidLevels(int max_levels) const {
    return getLevels(bids_, max_levels);
}

vector<OrderBookLevel> OrderBook::getAskLevels(int max_levels) const {
    return getLevels(asks_, max_levels);
}

// ============================================================================
// Utility Methods
// ============================================================================

bool OrderBook::isEmpty() const {
    return bids_.empty() && asks_.empty();
}

size_t OrderBook::getOrderCount() const {
    return orders_by_id_.size();
}

string OrderBook::getSymbol() const {
    return symbol_;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void OrderBook::removeOrderFromPriceLevel(OrderId order_id, Price price, Side side) {
    if (side == Side::BUY) {
        auto it = bids_.find(price);
        if (it != bids_.end()) {
            auto& orders = it->second;
            orders.erase(
                remove_if(orders.begin(), orders.end(),
                         [order_id](const shared_ptr<Order>& order) {
                             return order->id == order_id;
                         }),
                orders.end()
            );
            
            // Remove empty price level
            if (orders.empty()) {
                bids_.erase(it);
            }
        }
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end()) {
            auto& orders = it->second;
            orders.erase(
                remove_if(orders.begin(), orders.end(),
                         [order_id](const shared_ptr<Order>& order) {
                             return order->id == order_id;
                         }),
                orders.end()
            );
            
            // Remove empty price level
            if (orders.empty()) {
                asks_.erase(it);
            }
        }
    }
}

void OrderBook::addOrderToPriceLevel(shared_ptr<Order> order) {
    if (order->side == Side::BUY) {
        bids_[order->price].push_back(order);
    } else {
        asks_[order->price].push_back(order);
    }
}

bool OrderBook::isValidOrder(const shared_ptr<Order>& order) const {
    return order && 
           order->isValid() && 
           order->symbol == symbol_ &&
           order->isActive();
}

} // namespace deepquote 