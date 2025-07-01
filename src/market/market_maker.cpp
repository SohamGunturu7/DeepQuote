#include "market/market_maker.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

namespace deepquote {

// ============================================================================
// Constructor/Destructor
// ============================================================================

MarketMaker::MarketMaker(MarketSimulator* simulator, const MarketMakerConfig& config)
    : simulator_(simulator), config_(config), running_(false), next_order_id_(1) {
    
    // Register trader with simulator
    simulator_->registerTrader(config_.trader_id, 1000000.0);  // $1M initial capital
    
    // Initialize state tracking
    for (const auto& symbol : config_.symbols) {
        active_orders_[symbol] = std::vector<std::shared_ptr<Order>>();
        price_history_[symbol] = std::vector<double>();
        volatilities_[symbol] = 0.0;
    }
    
    std::cout << "Market Maker initialized for symbols: ";
    for (const auto& symbol : config_.symbols) {
        std::cout << symbol << " ";
    }
    std::cout << std::endl;
}

MarketMaker::~MarketMaker() {
    stop();
}

// ============================================================================
// Lifecycle Methods
// ============================================================================

void MarketMaker::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    worker_thread_ = std::thread(&MarketMaker::run, this);
    
    std::cout << "Market Maker started for symbols: ";
    for (const auto& symbol : config_.symbols) {
        std::cout << symbol << " ";
    }
    std::cout << std::endl;
}

void MarketMaker::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    // Cancel all active orders
    for (const auto& symbol : config_.symbols) {
        cancelOldOrders(symbol);
    }
    
    std::cout << "Market Maker stopped" << std::endl;
}

// ============================================================================
// Worker Thread
// ============================================================================

void MarketMaker::run() {
    while (running_.load()) {
        try {
            updateQuotes();
            
            // Sleep for update interval
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.update_interval_ms));
        } catch (const std::exception& e) {
            std::cerr << "Market Maker error: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.update_interval_ms));
        }
    }
}

// ============================================================================
// Core Market Making Logic
// ============================================================================

void MarketMaker::updateQuotes() {
    for (const auto& symbol : config_.symbols) {
        updateSymbolQuotes(symbol);
    }
}

void MarketMaker::updateSymbolQuotes(const std::string& symbol) {
    // Get current market data
    double best_bid = simulator_->getBestBid(symbol);
    double best_ask = simulator_->getBestAsk(symbol);
    
    // Calculate target prices
    double mid_price;
    if (best_bid > 0 && best_ask > 0) {
        // Use existing market prices
        mid_price = (best_bid + best_ask) / 2.0;
    } else {
        // Use base price if no market exists
        mid_price = config_.base_price;
    }
    
    // Update price history for volatility calculation
    updateVolatility(symbol, mid_price);
    
    // Calculate spread
    double spread_pct = config_.adaptive_spread ? 
        calculateAdaptiveSpread(symbol, mid_price) : config_.spread_pct;
    
    // Calculate bid and ask prices
    double spread = mid_price * spread_pct;
    double bid_price = mid_price - spread / 2.0;
    double ask_price = mid_price + spread / 2.0;
    
    // Cancel old orders
    cancelOldOrders(symbol);
    
    // Place new orders
    placeBidOrders(symbol, bid_price);
    placeAskOrders(symbol, ask_price);
}

void MarketMaker::cancelOldOrders(const std::string& symbol) {
    auto it = active_orders_.find(symbol);
    if (it == active_orders_.end()) {
        return;
    }
    
    for (auto& order : it->second) {
        if (order && order->isActive()) {
            // In a real system, we would send cancel requests to the exchange
            // For now, we just mark them as cancelled
            order->status = OrderStatus::CANCELLED;
        }
    }
    
    it->second.clear();
}

void MarketMaker::placeBidOrders(const std::string& symbol, double bid_price) {
    for (int i = 0; i < config_.max_orders_per_side; ++i) {
        // Slightly lower price for each additional order
        double price = bid_price - (i * 0.01);  // $0.01 increments
        
        auto order = createOrder(symbol, Side::BUY, price, config_.order_size);
        if (order) {
            // Process the order
            auto trades = simulator_->processOrder(order);
            
            // Track the order
            active_orders_[symbol].push_back(order);
        }
    }
}

void MarketMaker::placeAskOrders(const std::string& symbol, double ask_price) {
    for (int i = 0; i < config_.max_orders_per_side; ++i) {
        // Slightly higher price for each additional order
        double price = ask_price + (i * 0.01);  // $0.01 increments
        
        auto order = createOrder(symbol, Side::SELL, price, config_.order_size);
        if (order) {
            // Process the order
            auto trades = simulator_->processOrder(order);
            
            // Track the order
            active_orders_[symbol].push_back(order);
        }
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

std::shared_ptr<Order> MarketMaker::createOrder(const std::string& symbol, Side side, 
                                               double price, double quantity) {
    auto order = std::make_shared<Order>();
    order->id = getNextOrderId();
    order->side = side;
    order->type = OrderType::LIMIT;
    order->price = price;
    order->quantity = quantity;
    order->symbol = symbol;
    order->trader_id = config_.trader_id;
    order->strategy_id = "market_maker";
    
    return order;
}

double MarketMaker::calculateAdaptiveSpread(const std::string& symbol, double mid_price) {
    auto it = price_history_.find(symbol);
    if (it == price_history_.end() || it->second.size() < 2) {
        return config_.min_spread_pct;
    }
    
    // Calculate price volatility
    const auto& prices = it->second;
    std::vector<double> returns;
    returns.reserve(prices.size() - 1);
    
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i-1] > 0) {
            returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
        }
    }
    
    if (returns.empty()) {
        return config_.min_spread_pct;
    }
    
    // Calculate standard deviation
    double mean = 0.0;
    for (double ret : returns) {
        mean += ret;
    }
    mean /= returns.size();
    
    double variance = 0.0;
    for (double ret : returns) {
        double diff = ret - mean;
        variance += diff * diff;
    }
    variance /= returns.size();
    
    double volatility = std::sqrt(variance);
    
    // Adjust spread based on volatility
    // Higher volatility = wider spread
    double volatility_factor = std::min(volatility * 100.0, 1.0);  // Scale volatility
    double adaptive_spread = config_.min_spread_pct + 
        (config_.max_spread_pct - config_.min_spread_pct) * volatility_factor;
    
    return adaptive_spread;
}

void MarketMaker::updateVolatility(const std::string& symbol, double price) {
    auto it = price_history_.find(symbol);
    if (it == price_history_.end()) {
        price_history_[symbol] = std::vector<double>();
        it = price_history_.find(symbol);
    }
    
    it->second.push_back(price);
    
    // Keep only recent prices
    if (it->second.size() > static_cast<size_t>(config_.volatility_window)) {
        it->second.erase(it->second.begin());
    }
}

// ============================================================================
// Statistics
// ============================================================================

MarketMaker::Stats MarketMaker::getStats() const {
    Stats stats;
    stats.trader_id = config_.trader_id;
    stats.running = running_.load();
    
    // Get trader stats if available
    try {
        if (simulator_->hasTrader(config_.trader_id)) {
            const auto& trader = simulator_->getTrader(config_.trader_id);
            stats.cash = trader.getCash();
            stats.total_pnl = trader.getRealizedPnL();
        }
    } catch (...) {
        stats.cash = 0.0;
        stats.total_pnl = 0.0;
    }
    
    // Count active orders
    for (const auto& symbol : config_.symbols) {
        auto it = active_orders_.find(symbol);
        if (it != active_orders_.end()) {
            stats.active_orders[symbol] = it->second.size();
        } else {
            stats.active_orders[symbol] = 0;
        }
    }
    
    return stats;
}

} // namespace deepquote 