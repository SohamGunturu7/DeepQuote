#pragma once

#include "core/types.h"
#include "core/order.h"
#include "market/order_book.h"
#include <vector>
#include <memory>
#include <functional>
using namespace std;

namespace deepquote {

// ============================================================================
// Matching Engine Implementation
// ============================================================================

class MatchingEngine {
public:
    MatchingEngine(const string& symbol);
    ~MatchingEngine() = default;
    
    // Order processing
    vector<Trade> processOrder(shared_ptr<Order> order);
    vector<Trade> processMarketOrder(shared_ptr<Order> order);
    vector<Trade> processLimitOrder(shared_ptr<Order> order);
    
    // Order book access
    OrderBook& getOrderBook() { return order_book_; }
    const OrderBook& getOrderBook() const { return order_book_; }
    
    // Market data
    Price getBestBid() const { return order_book_.getBestBid(); }
    Price getBestAsk() const { return order_book_.getBestAsk(); }
    Price getMidPrice() const { return order_book_.getMidPrice(); }
    Price getSpread() const { return order_book_.getSpread(); }
    
    // Order book snapshots
    OrderBookSnapshot getSnapshot() const { return order_book_.getSnapshot(); }
    
    // Order book access
    size_t getOrderCount() const { return order_book_.getOrderCount(); }
    
    // Trade callback (for external systems)
    using TradeCallback = function<void(const Trade&)>;
    void setTradeCallback(TradeCallback callback) { trade_callback_ = callback; }
    
private:
    OrderBook order_book_;
    TradeCallback trade_callback_;
    
    // Helper methods
    vector<Trade> matchOrder(shared_ptr<Order> order);
    Trade createTrade(shared_ptr<Order> buy_order, shared_ptr<Order> sell_order, 
                     Price price, Quantity quantity);
    void updateOrderStatus(shared_ptr<Order> order);
    bool canMatch(shared_ptr<Order> order1, shared_ptr<Order> order2) const;
    Price getMatchPrice(shared_ptr<Order> incoming, shared_ptr<Order> resting) const;
};

} // namespace deepquote 