#pragma once

#include "core/types.h"
#include "core/order.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>
using namespace std;

namespace deepquote {

// ============================================================================
// Order Book Implementation
// ============================================================================

class OrderBook {
public:
    OrderBook(const string& symbol);
    ~OrderBook() = default;
    
    // Order management
    bool addOrder(shared_ptr<Order> order);
    bool cancelOrder(OrderId order_id);
    bool modifyOrder(OrderId order_id, Price new_price, Quantity new_quantity);
    
    // Order queries
    shared_ptr<Order> getOrder(OrderId order_id) const;
    vector<shared_ptr<Order>> getOrdersAtPrice(Price price, Side side) const;
    vector<shared_ptr<Order>> getActiveOrders() const;
    
    // Market data
    Price getBestBid() const;
    Price getBestAsk() const;
    Price getMidPrice() const;
    Price getSpread() const;
    Quantity getBidDepth() const;
    Quantity getAskDepth() const;
    Quantity getDepthAtPrice(Price price, Side side) const;
    
    // Order book snapshots
    OrderBookSnapshot getSnapshot() const;
    vector<OrderBookLevel> getBidLevels(int max_levels = 10) const;
    vector<OrderBookLevel> getAskLevels(int max_levels = 10) const;
    
    // Utility
    bool isEmpty() const;
    size_t getOrderCount() const;
    string getSymbol() const;
    
private:
    // Order book structure using price-time priority
    // Bids: highest price first (descending)
    // Asks: lowest price first (ascending)
    map<Price, vector<shared_ptr<Order>>, greater<Price>> bids_;
    map<Price, vector<shared_ptr<Order>>, less<Price>> asks_;
    
    // Fast order lookup by ID
    unordered_map<OrderId, shared_ptr<Order>> orders_by_id_;
    
    // Symbol for this order book
    string symbol_;
    
    // Helper methods
    void removeOrderFromPriceLevel(OrderId order_id, Price price, Side side);
    void addOrderToPriceLevel(shared_ptr<Order> order);
    template<typename Compare>
    vector<OrderBookLevel> getLevels(const map<Price, vector<shared_ptr<Order>>, Compare>& price_levels, 
                                   int max_levels) const;
    bool isValidOrder(const shared_ptr<Order>& order) const;
};

// ============================================================================
// Template Implementation
// ============================================================================

template<typename Compare>
vector<OrderBookLevel> OrderBook::getLevels(const map<Price, vector<shared_ptr<Order>>, Compare>& price_levels, 
                                          int max_levels) const {
    vector<OrderBookLevel> levels;
    int count = 0;
    
    for (const auto& level : price_levels) {
        if (count >= max_levels) break;
        
        Quantity total_quantity = 0;
        for (const auto& order : level.second) {
            total_quantity += order->getRemainingQuantity();
        }
        
        levels.emplace_back(level.first, total_quantity, level.second.size());
        count++;
    }
    
    return levels;
}

} // namespace deepquote 