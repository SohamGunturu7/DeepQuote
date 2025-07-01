#pragma once

#include "market/market_simulator.h"
#include "strategies/trading_strategy.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>

namespace deepquote {

// ============================================================================
// Market Maker Configuration
// ============================================================================

struct MarketMakerConfig {
    std::string trader_id;
    std::vector<std::string> symbols;
    double base_price;
    double spread_pct;
    double order_size;
    int max_orders_per_side;
    int update_interval_ms;
    bool adaptive_spread;
    double min_spread_pct;
    double max_spread_pct;
    int volatility_window;
    
    MarketMakerConfig() 
        : trader_id("market_maker"), base_price(100.0), spread_pct(0.001),
          order_size(10.0), max_orders_per_side(3), update_interval_ms(100),
          adaptive_spread(false), min_spread_pct(0.0005), max_spread_pct(0.005),
          volatility_window(20) {}
};

// ============================================================================
// Market Maker Implementation
// ============================================================================

class MarketMaker {
public:
    MarketMaker(MarketSimulator* simulator, const MarketMakerConfig& config);
    ~MarketMaker();
    
    // Lifecycle
    void start();
    void stop();
    bool isRunning() const { return running_.load(); }
    
    // Configuration
    void setSpread(double spread_pct) { config_.spread_pct = spread_pct; }
    void setOrderSize(double size) { config_.order_size = size; }
    void setUpdateInterval(int ms) { config_.update_interval_ms = ms; }
    
    // Statistics
    struct Stats {
        std::string trader_id;
        double cash;
        double total_pnl;
        std::unordered_map<std::string, int> active_orders;
        bool running;
    };
    
    Stats getStats() const;
    
private:
    MarketSimulator* simulator_;
    MarketMakerConfig config_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    
    // State tracking
    std::unordered_map<std::string, std::vector<std::shared_ptr<Order>>> active_orders_;
    std::unordered_map<std::string, std::vector<double>> price_history_;
    std::unordered_map<std::string, double> volatilities_;
    
    // Order ID generation
    std::atomic<OrderId> next_order_id_;
    
    // Worker thread function
    void run();
    
    // Core market making logic
    void updateQuotes();
    void updateSymbolQuotes(const std::string& symbol);
    void cancelOldOrders(const std::string& symbol);
    void placeBidOrders(const std::string& symbol, double bid_price);
    void placeAskOrders(const std::string& symbol, double ask_price);
    
    // Helper methods
    OrderId getNextOrderId() { return next_order_id_++; }
    double calculateAdaptiveSpread(const std::string& symbol, double mid_price);
    void updateVolatility(const std::string& symbol, double price);
    std::shared_ptr<Order> createOrder(const std::string& symbol, Side side, 
                                     double price, double quantity);
};

} // namespace deepquote 