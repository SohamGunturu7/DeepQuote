#include "strategies/market_making_strategy.h"
#include "market/market_simulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

MarketMakingStrategy::MarketMakingStrategy(const MarketMakingConfig& config)
    : TradingStrategy(config), config_(config) {
}

// ============================================================================
// Strategy Lifecycle
// ============================================================================

void MarketMakingStrategy::initialize() {
    active_ = true;
    
    // Initialize state tracking for each symbol
    for (const auto& symbol : config_.symbols) {
        active_orders_[symbol] = vector<shared_ptr<Order>>();
        current_spreads_[symbol] = config_.spread_target;
        inventory_skews_[symbol] = 0.0;
        volatilities_[symbol] = 0.0;
    }
    
    cout << "Market Making Strategy initialized for symbols: ";
    for (const auto& symbol : config_.symbols) {
        cout << symbol << " ";
    }
    cout << endl;
}

void MarketMakingStrategy::update(const MarketData& data) {
    if (!active_ || !simulator_) return;
    
    vector<shared_ptr<Order>> new_orders;
    
    for (const auto& symbol : config_.symbols) {
        // Get current market data
        auto it = data.order_books.find(symbol);
        if (it == data.order_books.end()) continue;
        
        const OrderBookSnapshot& snapshot = it->second;
        double mid_price = snapshot.getMidPrice();
        
        if (mid_price <= 0) continue;
        
        // Update volatility
        auto price_it = data.price_history.find(symbol);
        if (price_it != data.price_history.end()) {
            updateVolatility(symbol, price_it->second);
        }
        
        // Calculate optimal spread
        double optimal_spread = calculateOptimalSpread(symbol, mid_price, volatilities_[symbol]);
        current_spreads_[symbol] = optimal_spread;
        
        // Calculate inventory skew
        inventory_skews_[symbol] = calculateInventorySkew(symbol);
        
        // Cancel existing orders if spread changed significantly
        if (abs(optimal_spread - current_spreads_[symbol]) > config_.spread_target * 0.5) {
            cancelAllOrders(symbol);
        }
        
        // Place new orders
        placeOrders(symbol, mid_price, optimal_spread);
    }
    
    // Update risk metrics
    updateRiskMetrics();
}

void MarketMakingStrategy::shutdown() {
    active_ = false;
    
    // Cancel all active orders
    for (const auto& symbol : config_.symbols) {
        cancelAllOrders(symbol);
    }
    
    cout << "Market Making Strategy shutdown" << endl;
}

// ============================================================================
// Core Strategy Interface
// ============================================================================

vector<shared_ptr<Order>> MarketMakingStrategy::generateOrders(const MarketData& data) {
    if (!active_) return vector<shared_ptr<Order>>();
    
    vector<shared_ptr<Order>> orders;
    
    for (const auto& symbol : config_.symbols) {
        auto it = data.order_books.find(symbol);
        if (it == data.order_books.end()) continue;
        
        const OrderBookSnapshot& snapshot = it->second;
        double mid_price = snapshot.getMidPrice();
        
        if (mid_price <= 0) continue;
        
        // Calculate optimal spread
        double optimal_spread = calculateOptimalSpread(symbol, mid_price, volatilities_[symbol]);
        
        // Place orders
        double bid_price = mid_price - (optimal_spread * mid_price / 2.0);
        double ask_price = mid_price + (optimal_spread * mid_price / 2.0);
        
        // Adjust prices based on inventory skew
        double inventory_adjustment = inventory_skews_[symbol] * config_.inventory_scale * mid_price;
        bid_price -= inventory_adjustment;
        ask_price -= inventory_adjustment;
        
        // Place bid order
        if (shouldPlaceOrder(symbol, Side::BUY, bid_price)) {
            auto bid_order = createOrder(symbol, Side::BUY, OrderType::LIMIT, 
                                       config_.order_size, bid_price);
            if (bid_order) {
                orders.push_back(bid_order);
                trackOrder(bid_order);
            }
        }
        
        // Place ask order
        if (shouldPlaceOrder(symbol, Side::SELL, ask_price)) {
            auto ask_order = createOrder(symbol, Side::SELL, OrderType::LIMIT, 
                                       config_.order_size, ask_price);
            if (ask_order) {
                orders.push_back(ask_order);
                trackOrder(ask_order);
            }
        }
    }
    
    return orders;
}

void MarketMakingStrategy::onTrade(const Trade& trade) {
    // Update inventory tracking
    if (simulator_ && simulator_->hasTrader(config_.trader_id)) {
        auto& trader = simulator_->getTrader(config_.trader_id);
        double position = trader.getInventory(trade.symbol);
        current_positions_[trade.symbol] = position;
    }
    
    // Remove filled orders from tracking
    untrackOrder(trade.buy_order_id);
    untrackOrder(trade.sell_order_id);
}

void MarketMakingStrategy::onOrderUpdate(const shared_ptr<Order>& order) {
    if (!order) return;
    
    // Update order tracking based on status
    if (order->status == OrderStatus::FILLED || order->status == OrderStatus::CANCELLED) {
        untrackOrder(order->id);
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

void MarketMakingStrategy::cancelAllOrders(const string& symbol) {
    auto it = active_orders_.find(symbol);
    if (it == active_orders_.end()) return;
    
    for (auto& order : it->second) {
        if (order && order->isActive()) {
            // In a real system, we would send cancel requests to the exchange
            // For now, we just mark them as cancelled
            order->status = OrderStatus::CANCELLED;
        }
    }
    
    it->second.clear();
}

void MarketMakingStrategy::placeOrders(const string& symbol, double mid_price, double spread) {
    double bid_price = mid_price - (spread * mid_price / 2.0);
    double ask_price = mid_price + (spread * mid_price / 2.0);
    
    // Adjust prices based on inventory skew
    double inventory_adjustment = inventory_skews_[symbol] * config_.inventory_scale * mid_price;
    bid_price -= inventory_adjustment;
    ask_price -= inventory_adjustment;
    
    // Place bid orders
    int bid_count = getOrderCount(symbol, Side::BUY);
    while (bid_count < config_.max_orders_per_side) {
        auto order = createOrder(symbol, Side::BUY, OrderType::LIMIT, 
                               config_.order_size, bid_price);
        if (order) {
            trackOrder(order);
            bid_count++;
        } else {
            break;
        }
    }
    
    // Place ask orders
    int ask_count = getOrderCount(symbol, Side::SELL);
    while (ask_count < config_.max_orders_per_side) {
        auto order = createOrder(symbol, Side::SELL, OrderType::LIMIT, 
                               config_.order_size, ask_price);
        if (order) {
            trackOrder(order);
            ask_count++;
        } else {
            break;
        }
    }
}

double MarketMakingStrategy::calculateOptimalSpread(const string& symbol, double mid_price, double volatility) {
    double base_spread = config_.spread_target;
    
    if (!config_.dynamic_spread) {
        return base_spread;
    }
    
    // Adjust spread based on volatility
    double volatility_adjustment = volatility * config_.volatility_scale;
    
    // Adjust spread based on inventory
    double inventory_adjustment = abs(inventory_skews_[symbol]) * config_.inventory_scale;
    
    double optimal_spread = base_spread + volatility_adjustment + inventory_adjustment;
    
    // Ensure minimum spread
    return max(optimal_spread, config_.spread_target * 0.5);
}

double MarketMakingStrategy::calculateInventorySkew(const string& symbol) {
    if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) return 0.0;
    
    auto& trader = simulator_->getTrader(config_.trader_id);
    double position = trader.getInventory(symbol);
    double max_position = position_limits_[symbol];
    
    if (max_position <= 0) return 0.0;
    
    return (position - config_.inventory_target) / max_position;
}

void MarketMakingStrategy::updateVolatility(const string& symbol, const vector<double>& prices) {
    volatilities_[symbol] = calculateVolatility(prices, config_.volatility_period);
}

bool MarketMakingStrategy::shouldPlaceOrder(const string& symbol, Side side, double price) {
    // Check if we already have enough orders on this side
    if (getOrderCount(symbol, side) >= config_.max_orders_per_side) {
        return false;
    }
    
    // Check if price is reasonable
    if (price <= 0) return false;
    
    // Check risk limits
    return checkRiskLimits();
}

// ============================================================================
// Order Management
// ============================================================================

void MarketMakingStrategy::trackOrder(const shared_ptr<Order>& order) {
    if (!order) return;
    
    auto it = active_orders_.find(order->symbol);
    if (it != active_orders_.end()) {
        it->second.push_back(order);
    }
}

void MarketMakingStrategy::untrackOrder(OrderId order_id) {
    for (auto& kv : active_orders_) {
        auto& orders = kv.second;
        orders.erase(remove_if(orders.begin(), orders.end(),
                              [order_id](const shared_ptr<Order>& order) {
                                  return order && order->id == order_id;
                              }), orders.end());
    }
}

int MarketMakingStrategy::getOrderCount(const string& symbol, Side side) const {
    auto it = active_orders_.find(symbol);
    if (it == active_orders_.end()) return 0;
    
    int count = 0;
    for (const auto& order : it->second) {
        if (order && order->side == side && order->isActive()) {
            count++;
        }
    }
    
    return count;
}

// ============================================================================
// Accessors
// ============================================================================

double MarketMakingStrategy::getCurrentSpread(const string& symbol) const {
    auto it = current_spreads_.find(symbol);
    return (it != current_spreads_.end()) ? it->second : config_.spread_target;
}

double MarketMakingStrategy::getInventorySkew(const string& symbol) const {
    auto it = inventory_skews_.find(symbol);
    return (it != inventory_skews_.end()) ? it->second : 0.0;
}

} // namespace deepquote 