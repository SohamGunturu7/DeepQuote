#include "strategies/mean_reversion_strategy.h"
#include "market/market_simulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

MeanReversionStrategy::MeanReversionStrategy(const MeanReversionConfig& config)
    : TradingStrategy(config), config_(config) {
}

// ============================================================================
// Strategy Lifecycle
// ============================================================================

void MeanReversionStrategy::initialize() {
    active_ = true;
    
    // Initialize state tracking for each symbol
    for (const auto& symbol : config_.symbols) {
        moving_averages_[symbol] = 0.0;
        standard_deviations_[symbol] = 0.0;
        z_scores_[symbol] = 0.0;
        bollinger_upper_[symbol] = 0.0;
        bollinger_lower_[symbol] = 0.0;
        price_history_[symbol] = vector<double>();
    }
    
    cout << "Mean Reversion Strategy initialized for symbols: ";
    for (const auto& symbol : config_.symbols) {
        cout << symbol << " ";
    }
    cout << endl;
}

void MeanReversionStrategy::update(const MarketData& data) {
    if (!active_) return;
    
    // Update price history and indicators for each symbol
    for (const auto& symbol : config_.symbols) {
        auto it = data.order_books.find(symbol);
        if (it == data.order_books.end()) continue;
        
        const OrderBookSnapshot& snapshot = it->second;
        double current_price = snapshot.getMidPrice();
        
        if (current_price <= 0) continue;
        
        // Update price history
        updatePriceHistory(symbol, current_price);
        
        // Update indicators
        updateIndicators(symbol, price_history_[symbol]);
        
        // Calculate current z-score
        z_scores_[symbol] = calculateZScore(symbol, current_price);
    }
    
    // Update risk metrics
    updateRiskMetrics();
}

void MeanReversionStrategy::shutdown() {
    active_ = false;
    cout << "Mean Reversion Strategy shutdown" << endl;
}

// ============================================================================
// Core Strategy Interface
// ============================================================================

vector<shared_ptr<Order>> MeanReversionStrategy::generateOrders(const MarketData& data) {
    if (!active_) return vector<shared_ptr<Order>>();
    
    vector<shared_ptr<Order>> orders;
    
    // Generate exit signals first (to close existing positions)
    auto exit_orders = generateExitSignals(data);
    orders.insert(orders.end(), exit_orders.begin(), exit_orders.end());
    
    // Generate entry signals (to open new positions)
    auto entry_orders = generateEntrySignals(data);
    orders.insert(orders.end(), entry_orders.begin(), entry_orders.end());
    
    return orders;
}

void MeanReversionStrategy::onTrade(const Trade& trade) {
    // Update position tracking
    if (simulator_ && simulator_->hasTrader(config_.trader_id)) {
        auto& trader = simulator_->getTrader(config_.trader_id);
        double position = trader.getInventory(trade.symbol);
        current_positions_[trade.symbol] = position;
    }
}

void MeanReversionStrategy::onOrderUpdate(const shared_ptr<Order>& order) {
    // No special handling needed for mean reversion strategy
}

// ============================================================================
// Accessors
// ============================================================================

double MeanReversionStrategy::getCurrentZScore(const string& symbol) const {
    auto it = z_scores_.find(symbol);
    return (it != z_scores_.end()) ? it->second : 0.0;
}

double MeanReversionStrategy::getMovingAverage(const string& symbol) const {
    auto it = moving_averages_.find(symbol);
    return (it != moving_averages_.end()) ? it->second : 0.0;
}

double MeanReversionStrategy::getBollingerUpper(const string& symbol) const {
    auto it = bollinger_upper_.find(symbol);
    return (it != bollinger_upper_.end()) ? it->second : 0.0;
}

double MeanReversionStrategy::getBollingerLower(const string& symbol) const {
    auto it = bollinger_lower_.find(symbol);
    return (it != bollinger_lower_.end()) ? it->second : 0.0;
}

// ============================================================================
// Helper Methods
// ============================================================================

void MeanReversionStrategy::updateIndicators(const string& symbol, const vector<double>& prices) {
    if (prices.size() < config_.lookback_period) return;
    
    // Calculate moving average
    moving_averages_[symbol] = calculateMovingAverage(prices, config_.lookback_period);
    
    // Calculate standard deviation
    double mean = moving_averages_[symbol];
    double variance = 0.0;
    
    for (int i = prices.size() - config_.lookback_period; i < prices.size(); ++i) {
        double diff = prices[i] - mean;
        variance += diff * diff;
    }
    variance /= config_.lookback_period;
    
    standard_deviations_[symbol] = sqrt(variance);
    
    // Calculate Bollinger Bands
    if (config_.use_bollinger_bands && standard_deviations_[symbol] > 0) {
        bollinger_upper_[symbol] = mean + (config_.bollinger_multiplier * standard_deviations_[symbol]);
        bollinger_lower_[symbol] = mean - (config_.bollinger_multiplier * standard_deviations_[symbol]);
    }
}

double MeanReversionStrategy::calculateZScore(const string& symbol, double current_price) {
    auto ma_it = moving_averages_.find(symbol);
    auto std_it = standard_deviations_.find(symbol);
    
    if (ma_it == moving_averages_.end() || std_it == standard_deviations_.end()) {
        return 0.0;
    }
    
    double mean = ma_it->second;
    double std_dev = std_it->second;
    
    if (std_dev <= 0) return 0.0;
    
    return (current_price - mean) / std_dev;
}

double MeanReversionStrategy::calculatePositionSize(const string& symbol, double z_score) {
    double base_size = config_.position_size;
    
    if (!config_.dynamic_position_sizing) {
        return base_size;
    }
    
    // Scale position size based on z-score magnitude
    double z_score_abs = abs(z_score);
    double scale_factor = min(z_score_abs / config_.entry_threshold, 2.0);
    
    double position_size = base_size * scale_factor;
    
    // Apply maximum position size limit
    if (simulator_ && simulator_->hasTrader(config_.trader_id)) {
        auto& trader = simulator_->getTrader(config_.trader_id);
        double capital = trader.getCash();
        double max_size = capital * config_.max_position_size;
        position_size = min(position_size, max_size);
    }
    
    return position_size;
}

bool MeanReversionStrategy::shouldEnterLong(const string& symbol, double z_score) const {
    // Enter long when price is significantly below mean (negative z-score)
    return z_score < -config_.entry_threshold;
}

bool MeanReversionStrategy::shouldEnterShort(const string& symbol, double z_score) const {
    // Enter short when price is significantly above mean (positive z-score)
    return z_score > config_.entry_threshold;
}

bool MeanReversionStrategy::shouldExitLong(const string& symbol, double z_score) const {
    // Exit long when price approaches or crosses above mean
    return z_score > -config_.exit_threshold;
}

bool MeanReversionStrategy::shouldExitShort(const string& symbol, double z_score) const {
    // Exit short when price approaches or crosses below mean
    return z_score < config_.exit_threshold;
}

void MeanReversionStrategy::updatePriceHistory(const string& symbol, double price) {
    auto it = price_history_.find(symbol);
    if (it != price_history_.end()) {
        it->second.push_back(price);
        
        // Keep only recent prices (2x lookback period for safety)
        if (it->second.size() > config_.lookback_period * 2) {
            it->second.erase(it->second.begin());
        }
    }
}

// ============================================================================
// Signal Generation
// ============================================================================

vector<shared_ptr<Order>> MeanReversionStrategy::generateEntrySignals(const MarketData& data) {
    vector<shared_ptr<Order>> orders;
    
    for (const auto& symbol : config_.symbols) {
        if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) continue;
        
        auto& trader = simulator_->getTrader(config_.trader_id);
        double current_position = trader.getInventory(symbol);
        double z_score = getCurrentZScore(symbol);
        
        // Check for long entry
        if (current_position == 0 && shouldEnterLong(symbol, z_score)) {
            double position_size = calculatePositionSize(symbol, z_score);
            auto order = createOrder(symbol, Side::BUY, OrderType::MARKET, position_size, 0.0);
            if (order) {
                orders.push_back(order);
            }
        }
        
        // Check for short entry
        if (current_position == 0 && shouldEnterShort(symbol, z_score)) {
            double position_size = calculatePositionSize(symbol, z_score);
            auto order = createOrder(symbol, Side::SELL, OrderType::MARKET, position_size, 0.0);
            if (order) {
                orders.push_back(order);
            }
        }
    }
    
    return orders;
}

vector<shared_ptr<Order>> MeanReversionStrategy::generateExitSignals(const MarketData& data) {
    vector<shared_ptr<Order>> orders;
    
    for (const auto& symbol : config_.symbols) {
        if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) continue;
        
        auto& trader = simulator_->getTrader(config_.trader_id);
        double current_position = trader.getInventory(symbol);
        double z_score = getCurrentZScore(symbol);
        
        // Exit long position
        if (current_position > 0 && shouldExitLong(symbol, z_score)) {
            auto order = createOrder(symbol, Side::SELL, OrderType::MARKET, current_position, 0.0);
            if (order) {
                orders.push_back(order);
            }
        }
        
        // Exit short position
        if (current_position < 0 && shouldExitShort(symbol, z_score)) {
            auto order = createOrder(symbol, Side::BUY, OrderType::MARKET, abs(current_position), 0.0);
            if (order) {
                orders.push_back(order);
            }
        }
    }
    
    return orders;
}

} // namespace deepquote 