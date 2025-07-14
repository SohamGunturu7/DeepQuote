#pragma once

#include "core/types.h"
#include <string>
#include <vector>
#include <memory>
using namespace std;

namespace deepquote {

// Forward declarations
class Order;
class Trade;

// Order structure
class Order {
public:
    OrderId id;
    Side side;
    OrderType type;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    OrderStatus status;
    Timestamp timestamp;
    Timestamp created_time;
    string symbol;
    string trader_id;
    
    bool is_market_maker_order = false;
    string strategy_id;
    
    Order() = default;
    
    Order(OrderId id, Side side, OrderType type, Price price, 
          Quantity quantity, const string& symbol, 
          const string& trader_id);
    
    bool isValid() const;
    bool isFullyFilled() const;
    bool isPartiallyFilled() const;
    bool isActive() const;
    Quantity getRemainingQuantity() const;
    double getFillRatio() const;
    
    string toString() const;
};

// Trade structure
class Trade {
public:
    OrderId buy_order_id;
    OrderId sell_order_id;
    Price price;
    Quantity quantity;
    Timestamp timestamp;
    string symbol;
    
    bool involves_market_maker = false;
    string market_maker_id;
    
    Trade() = default;
    
    Trade(OrderId buy_id, OrderId sell_id, Price price, 
          Quantity quantity, const string& symbol);
    
    bool isValid() const;
    string toString() const;
};

// Order book level
class OrderBookLevel {
public:
    Price price;
    Quantity total_quantity;
    int order_count;
    
    OrderBookLevel() = default;
    OrderBookLevel(Price price, Quantity quantity, int count = 1);
    
    bool isValid() const;
    string toString() const;
};

// Order book snapshot
class OrderBookSnapshot {
public:
    string symbol;
    Timestamp timestamp;
    vector<OrderBookLevel> bids;
    vector<OrderBookLevel> asks;
    
    Price mid_price;
    Price spread;
    Quantity bid_depth;
    Quantity ask_depth;
    
    OrderBookSnapshot() = default;
    OrderBookSnapshot(const string& symbol, Timestamp timestamp);
    
    bool isValid() const;
    Price getBestPrice(Side side) const;
    Price getMidPrice() const;
    Price getSpread() const;
    Quantity getDepth(Side side) const;
    
    string toString() const;
};

// Order management
class OrderManager {
public:
    OrderManager() = default;
    virtual ~OrderManager() = default;
    
    virtual OrderId createOrder(const Order& order);
    virtual bool cancelOrder(OrderId order_id);
    virtual bool modifyOrder(OrderId order_id, Price new_price, Quantity new_quantity);
    
    virtual shared_ptr<Order> getOrder(OrderId order_id) const;
    virtual vector<shared_ptr<Order>> getActiveOrders() const;
    virtual vector<shared_ptr<Order>> getOrdersBySymbol(const string& symbol) const;
    
    virtual void processTrade(const Trade& trade);
    virtual vector<Trade> getTrades() const;
    
protected:
    shared_ptr<Order> findOrderById(OrderId id) const;
    vector<shared_ptr<Order>> findOrdersBySymbol(const string& symbol) const;
    
    vector<shared_ptr<Order>> orders_;
    vector<Trade> trades_;
    OrderId next_order_id_ = 1;
};

} // namespace deepquote 