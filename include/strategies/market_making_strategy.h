#pragma once

#include "strategies/trading_strategy.h"
#include <unordered_map>
using namespace std;

namespace deepquote {

// ============================================================================
// Market Making Strategy Configuration
// ============================================================================

struct MarketMakingConfig : public StrategyConfig {
    double spread_target;           // Target spread as percentage of mid-price
    double order_size;              // Size of each order
    int max_orders_per_side;        // Maximum orders per side
    double inventory_target;        // Target inventory (0 = neutral)
    double inventory_scale;         // How much to adjust spread based on inventory
    double volatility_scale;        // How much to adjust spread based on volatility
    int volatility_period;         // Period for volatility calculation
    bool dynamic_spread;           // Whether to adjust spread dynamically
    
    MarketMakingConfig() : spread_target(0.001), order_size(10.0), max_orders_per_side(3),
                          inventory_target(0.0), inventory_scale(0.1), volatility_scale(1.0),
                          volatility_period(20), dynamic_spread(true) {}
};

// ============================================================================
// Market Making Strategy Implementation
// ============================================================================

class MarketMakingStrategy : public TradingStrategy {
public:
    MarketMakingStrategy(const MarketMakingConfig& config);
    ~MarketMakingStrategy() = default;
    
    // Core strategy interface
    vector<shared_ptr<Order>> generateOrders(const MarketData& data) override;
    void onTrade(const Trade& trade) override;
    void onOrderUpdate(const shared_ptr<Order>& order) override;
    
    // Strategy lifecycle
    void initialize() override;
    void update(const MarketData& data) override;
    void shutdown() override;
    
    // Market making specific methods
    void setSpreadTarget(double spread) { config_.spread_target = spread; }
    void setOrderSize(double size) { config_.order_size = size; }
    void setInventoryTarget(double target) { config_.inventory_target = target; }
    
    // Accessors
    const MarketMakingConfig& getConfig() const { return config_; }
    double getCurrentSpread(const string& symbol) const;
    double getInventorySkew(const string& symbol) const;
    
private:
    MarketMakingConfig config_;
    
    // State tracking
    unordered_map<string, vector<shared_ptr<Order>>> active_orders_;
    unordered_map<string, double> current_spreads_;
    unordered_map<string, double> inventory_skews_;
    unordered_map<string, double> volatilities_;
    
    // Helper methods
    void cancelAllOrders(const string& symbol);
    void placeOrders(const string& symbol, double mid_price, double spread);
    double calculateOptimalSpread(const string& symbol, double mid_price, double volatility);
    double calculateInventorySkew(const string& symbol);
    void updateVolatility(const string& symbol, const vector<double>& prices);
    bool shouldPlaceOrder(const string& symbol, Side side, double price);
    
    // Order management
    void trackOrder(const shared_ptr<Order>& order);
    void untrackOrder(OrderId order_id);
    int getOrderCount(const string& symbol, Side side) const;
};

} // namespace deepquote 