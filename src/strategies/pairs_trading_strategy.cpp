#include "strategies/pairs_trading_strategy.h"
#include "market/market_simulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

PairsTradingStrategy::PairsTradingStrategy(const PairsTradingConfig& config)
    : TradingStrategy(config), config_(config) {
}

// ============================================================================
// Strategy Lifecycle
// ============================================================================

void PairsTradingStrategy::initialize() {
    active_ = true;
    
    // Initialize price history for all symbols in pairs
    for (const auto& pair : config_.pairs) {
        price_history_[pair.first] = vector<double>();
        price_history_[pair.second] = vector<double>();
        spread_history_[pair] = vector<double>();
    }
    
    cout << "Pairs Trading Strategy initialized for pairs: ";
    for (const auto& pair : config_.pairs) {
        cout << "(" << pair.first << ", " << pair.second << ") ";
    }
    cout << endl;
}

void PairsTradingStrategy::update(const MarketData& data) {
    if (!active_) return;
    
    // Update price history for all symbols
    for (const auto& pair : config_.pairs) {
        auto it1 = data.order_books.find(pair.first);
        auto it2 = data.order_books.find(pair.second);
        
        if (it1 != data.order_books.end() && it2 != data.order_books.end()) {
            double price1 = it1->second.getMidPrice();
            double price2 = it2->second.getMidPrice();
            
            if (price1 > 0 && price2 > 0) {
                updatePriceHistory(pair.first, price1);
                updatePriceHistory(pair.second, price2);
                
                // Calculate and store spread
                double spread = calculateSpread(pair.first, pair.second, price1, price2);
                spread_history_[pair].push_back(spread);
                
                // Keep only recent spreads
                if (spread_history_[pair].size() > config_.lookback_period * 2) {
                    spread_history_[pair].erase(spread_history_[pair].begin());
                }
            }
        }
    }
    
    // Update correlations and spread statistics
    updateCorrelations();
    updateSpreadStatistics();
    
    // Update risk metrics
    updateRiskMetrics();
}

void PairsTradingStrategy::shutdown() {
    active_ = false;
    cout << "Pairs Trading Strategy shutdown" << endl;
}

// ============================================================================
// Core Strategy Interface
// ============================================================================

vector<shared_ptr<Order>> PairsTradingStrategy::generateOrders(const MarketData& data) {
    if (!active_) return vector<shared_ptr<Order>>();
    
    vector<shared_ptr<Order>> orders;
    
    // Generate exit signals first
    auto exit_orders = generateExitSignals(data);
    orders.insert(orders.end(), exit_orders.begin(), exit_orders.end());
    
    // Generate entry signals
    auto entry_orders = generateEntrySignals(data);
    orders.insert(orders.end(), entry_orders.begin(), entry_orders.end());
    
    return orders;
}

void PairsTradingStrategy::onTrade(const Trade& trade) {
    // Update position tracking
    if (simulator_ && simulator_->hasTrader(config_.trader_id)) {
        auto& trader = simulator_->getTrader(config_.trader_id);
        double position = trader.getInventory(trade.symbol);
        current_positions_[trade.symbol] = position;
    }
}

void PairsTradingStrategy::onOrderUpdate(const shared_ptr<Order>& order) {
    // No special handling needed for pairs trading strategy
}

// ============================================================================
// Pairs Trading Specific Methods
// ============================================================================

void PairsTradingStrategy::addPair(const string& symbol1, const string& symbol2) {
    auto pair = makePair(symbol1, symbol2);
    config_.pairs.push_back(pair);
    
    // Initialize tracking for new pair
    price_history_[symbol1] = vector<double>();
    price_history_[symbol2] = vector<double>();
    spread_history_[pair] = vector<double>();
}

void PairsTradingStrategy::removePair(const string& symbol1, const string& symbol2) {
    auto pair = makePair(symbol1, symbol2);
    
    // Remove from pairs list
    config_.pairs.erase(remove(config_.pairs.begin(), config_.pairs.end(), pair), config_.pairs.end());
    
    // Clean up tracking data
    price_history_.erase(symbol1);
    price_history_.erase(symbol2);
    spread_history_.erase(pair);
    correlations_.erase(pair);
    spread_means_.erase(pair);
    spread_stds_.erase(pair);
    spread_z_scores_.erase(pair);
    cointegration_status_.erase(pair);
}

// ============================================================================
// Accessors
// ============================================================================

double PairsTradingStrategy::getSpreadZScore(const string& symbol1, const string& symbol2) const {
    auto pair = makePair(symbol1, symbol2);
    auto it = spread_z_scores_.find(pair);
    return (it != spread_z_scores_.end()) ? it->second : 0.0;
}

double PairsTradingStrategy::getCorrelation(const string& symbol1, const string& symbol2) const {
    auto pair = makePair(symbol1, symbol2);
    auto it = correlations_.find(pair);
    return (it != correlations_.end()) ? it->second : 0.0;
}

bool PairsTradingStrategy::isPairCointegrated(const string& symbol1, const string& symbol2) const {
    auto pair = makePair(symbol1, symbol2);
    auto it = cointegration_status_.find(pair);
    return (it != cointegration_status_.end()) ? it->second : false;
}

// ============================================================================
// Helper Methods
// ============================================================================

void PairsTradingStrategy::updatePriceHistory(const string& symbol, double price) {
    auto it = price_history_.find(symbol);
    if (it != price_history_.end()) {
        it->second.push_back(price);
        
        // Keep only recent prices
        if (it->second.size() > config_.lookback_period * 2) {
            it->second.erase(it->second.begin());
        }
    }
}

void PairsTradingStrategy::updateCorrelations() {
    for (const auto& pair : config_.pairs) {
        auto it1 = price_history_.find(pair.first);
        auto it2 = price_history_.find(pair.second);
        
        if (it1 != price_history_.end() && it2 != price_history_.end()) {
            const vector<double>& prices1 = it1->second;
            const vector<double>& prices2 = it2->second;
            
            if (prices1.size() >= config_.lookback_period && prices2.size() >= config_.lookback_period) {
                double correlation = calculateCorrelation(prices1, prices2);
                correlations_[pair] = correlation;
            }
        }
    }
}

void PairsTradingStrategy::updateSpreadStatistics() {
    for (const auto& pair : config_.pairs) {
        auto it = spread_history_.find(pair);
        if (it == spread_history_.end() || it->second.size() < config_.lookback_period) continue;
        
        const vector<double>& spreads = it->second;
        
        // Calculate mean
        double sum = 0.0;
        for (int i = spreads.size() - config_.lookback_period; i < spreads.size(); ++i) {
            sum += spreads[i];
        }
        double mean = sum / config_.lookback_period;
        spread_means_[pair] = mean;
        
        // Calculate standard deviation
        double variance = 0.0;
        for (int i = spreads.size() - config_.lookback_period; i < spreads.size(); ++i) {
            double diff = spreads[i] - mean;
            variance += diff * diff;
        }
        variance /= config_.lookback_period;
        spread_stds_[pair] = sqrt(variance);
        
        // Calculate current z-score
        if (spread_stds_[pair] > 0) {
            double current_spread = spreads.back();
            spread_z_scores_[pair] = (current_spread - mean) / spread_stds_[pair];
        }
        
        // Test cointegration if enabled
        if (config_.use_cointegration) {
            cointegration_status_[pair] = testCointegration(pair.first, pair.second);
        }
    }
}

double PairsTradingStrategy::calculateSpread(const string& symbol1, const string& symbol2, 
                                            double price1, double price2) {
    // Simple price ratio spread (log ratio for stationarity)
    return log(price1 / price2);
}

double PairsTradingStrategy::calculateZScore(const string& symbol1, const string& symbol2, double spread) {
    auto pair = makePair(symbol1, symbol2);
    
    auto mean_it = spread_means_.find(pair);
    auto std_it = spread_stds_.find(pair);
    
    if (mean_it == spread_means_.end() || std_it == spread_stds_.end()) {
        return 0.0;
    }
    
    double mean = mean_it->second;
    double std_dev = std_it->second;
    
    if (std_dev <= 0) return 0.0;
    
    return (spread - mean) / std_dev;
}

bool PairsTradingStrategy::testCointegration(const string& symbol1, const string& symbol2) {
    // Simplified cointegration test using correlation
    // In a real implementation, you would use Engle-Granger or Johansen test
    auto pair = makePair(symbol1, symbol2);
    auto it = correlations_.find(pair);
    
    if (it == correlations_.end()) return false;
    
    // High correlation suggests cointegration
    return abs(it->second) > config_.correlation_threshold;
}

double PairsTradingStrategy::calculatePositionSize(const string& symbol1, const string& symbol2, double z_score) {
    double base_size = config_.position_size;
    
    if (!config_.dynamic_position_sizing) {
        return base_size;
    }
    
    // Scale position size based on z-score magnitude
    double z_score_abs = abs(z_score);
    double scale_factor = min(z_score_abs / config_.entry_threshold, 2.0);
    
    return base_size * scale_factor;
}

bool PairsTradingStrategy::shouldEnterLongShort(const string& symbol1, const string& symbol2, double z_score) const {
    // Enter long-short when spread is significantly negative (z_score < -threshold)
    return z_score < -config_.entry_threshold;
}

bool PairsTradingStrategy::shouldEnterShortLong(const string& symbol1, const string& symbol2, double z_score) const {
    // Enter short-long when spread is significantly positive (z_score > threshold)
    return z_score > config_.entry_threshold;
}

bool PairsTradingStrategy::shouldExitPosition(const string& symbol1, const string& symbol2, double z_score) const {
    // Exit when spread approaches zero
    return abs(z_score) < config_.exit_threshold;
}

// ============================================================================
// Signal Generation
// ============================================================================

vector<shared_ptr<Order>> PairsTradingStrategy::generateEntrySignals(const MarketData& data) {
    vector<shared_ptr<Order>> orders;
    
    for (const auto& pair : config_.pairs) {
        if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) continue;
        
        // Check if pair is suitable for trading
        if (config_.use_cointegration && !isPairCointegrated(pair.first, pair.second)) {
            continue;
        }
        
        double correlation = getCorrelation(pair.first, pair.second);
        if (abs(correlation) < config_.correlation_threshold) {
            continue;
        }
        
        // Check if we already have a position in this pair
        if (hasActivePosition(pair.first, pair.second)) {
            continue;
        }
        
        double z_score = getSpreadZScore(pair.first, pair.second);
        
        // Generate long-short signal
        if (shouldEnterLongShort(pair.first, pair.second, z_score)) {
            double position_size = calculatePositionSize(pair.first, pair.second, z_score);
            
            auto order1 = createOrder(pair.first, Side::BUY, OrderType::MARKET, position_size, 0.0);
            auto order2 = createOrder(pair.second, Side::SELL, OrderType::MARKET, position_size, 0.0);
            
            if (order1) orders.push_back(order1);
            if (order2) orders.push_back(order2);
        }
        
        // Generate short-long signal
        if (shouldEnterShortLong(pair.first, pair.second, z_score)) {
            double position_size = calculatePositionSize(pair.first, pair.second, z_score);
            
            auto order1 = createOrder(pair.first, Side::SELL, OrderType::MARKET, position_size, 0.0);
            auto order2 = createOrder(pair.second, Side::BUY, OrderType::MARKET, position_size, 0.0);
            
            if (order1) orders.push_back(order1);
            if (order2) orders.push_back(order2);
        }
    }
    
    return orders;
}

vector<shared_ptr<Order>> PairsTradingStrategy::generateExitSignals(const MarketData& data) {
    vector<shared_ptr<Order>> orders;
    
    for (const auto& pair : config_.pairs) {
        if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) continue;
        
        if (!hasActivePosition(pair.first, pair.second)) continue;
        
        double z_score = getSpreadZScore(pair.first, pair.second);
        
        if (shouldExitPosition(pair.first, pair.second, z_score)) {
            auto& trader = simulator_->getTrader(config_.trader_id);
            double position1 = trader.getInventory(pair.first);
            double position2 = trader.getInventory(pair.second);
            
            // Close positions
            if (position1 > 0) {
                auto order = createOrder(pair.first, Side::SELL, OrderType::MARKET, position1, 0.0);
                if (order) orders.push_back(order);
            } else if (position1 < 0) {
                auto order = createOrder(pair.first, Side::BUY, OrderType::MARKET, abs(position1), 0.0);
                if (order) orders.push_back(order);
            }
            
            if (position2 > 0) {
                auto order = createOrder(pair.second, Side::SELL, OrderType::MARKET, position2, 0.0);
                if (order) orders.push_back(order);
            } else if (position2 < 0) {
                auto order = createOrder(pair.second, Side::BUY, OrderType::MARKET, abs(position2), 0.0);
                if (order) orders.push_back(order);
            }
        }
    }
    
    return orders;
}

// ============================================================================
// Utility Methods
// ============================================================================

pair<string, string> PairsTradingStrategy::makePair(const string& symbol1, const string& symbol2) const {
    // Ensure consistent ordering for pair keys
    return (symbol1 < symbol2) ? make_pair(symbol1, symbol2) : make_pair(symbol2, symbol1);
}

bool PairsTradingStrategy::hasActivePosition(const string& symbol1, const string& symbol2) const {
    if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) return false;
    
    auto& trader = simulator_->getTrader(config_.trader_id);
    double position1 = trader.getInventory(symbol1);
    double position2 = trader.getInventory(symbol2);
    
    // Check if we have opposite positions in the pair
    return (position1 > 0 && position2 < 0) || (position1 < 0 && position2 > 0);
}

} // namespace deepquote 