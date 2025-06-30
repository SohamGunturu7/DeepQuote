#include "strategies/trading_strategy.h"
#include "market/market_simulator.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

TradingStrategy::TradingStrategy(const StrategyConfig& config)
    : config_(config), simulator_(nullptr), active_(false),
      total_pnl_(0.0), realized_pnl_(0.0), unrealized_pnl_(0.0),
      max_drawdown_(0.0), peak_value_(config.initial_capital) {
    
    // Initialize position limits
    for (const auto& symbol : config_.symbols) {
        position_limits_[symbol] = config_.max_position_size;
        current_positions_[symbol] = 0.0;
    }
    
    initializeRiskMetrics();
}

// ============================================================================
// Performance Tracking
// ============================================================================

double TradingStrategy::getTotalPnL() const {
    return total_pnl_;
}

double TradingStrategy::getRealizedPnL() const {
    return realized_pnl_;
}

double TradingStrategy::getUnrealizedPnL() const {
    return unrealized_pnl_;
}

double TradingStrategy::getSharpeRatio() const {
    if (returns_history_.size() < 2) return 0.0;
    
    double mean_return = accumulate(returns_history_.begin(), returns_history_.end(), 0.0) 
                        / returns_history_.size();
    
    double variance = 0.0;
    for (double ret : returns_history_) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= (returns_history_.size() - 1);
    
    double std_dev = sqrt(variance);
    return std_dev > 0.0 ? mean_return / std_dev : 0.0;
}

double TradingStrategy::getMaxDrawdown() const {
    return max_drawdown_;
}

// ============================================================================
// Risk Management
// ============================================================================

bool TradingStrategy::checkRiskLimits() const {
    if (!config_.enable_risk_management) return true;
    
    // Check position limits
    for (const auto& kv : current_positions_) {
        if (abs(kv.second) > position_limits_.at(kv.first)) {
            return false;
        }
    }
    
    // Check drawdown limit
    if (max_drawdown_ > config_.max_drawdown) {
        return false;
    }
    
    return true;
}

void TradingStrategy::updateRiskMetrics() {
    if (!simulator_) return;
    
    // Update P&L
    realized_pnl_ = 0.0;
    unrealized_pnl_ = 0.0;
    
    for (const auto& symbol : config_.symbols) {
        if (simulator_->hasTrader(config_.trader_id)) {
            auto& trader = simulator_->getTrader(config_.trader_id);
            realized_pnl_ += trader.getRealizedPnL();
            
            // Get current price for unrealized P&L
            double current_price = simulator_->getMidPrice(symbol);
            if (current_price > 0) {
                double position = trader.getInventory(symbol);
                double avg_cost = trader.getAverageCost(symbol);
                if (position != 0 && avg_cost > 0) {
                    unrealized_pnl_ += (current_price - avg_cost) * position;
                }
            }
        }
    }
    
    total_pnl_ = realized_pnl_ + unrealized_pnl_;
    
    // Update drawdown
    double current_value = config_.initial_capital + total_pnl_;
    if (current_value > peak_value_) {
        peak_value_ = current_value;
    } else {
        double drawdown = (peak_value_ - current_value) / peak_value_;
        if (drawdown > max_drawdown_) {
            max_drawdown_ = drawdown;
        }
    }
    
    // Update returns history (simplified - could be more sophisticated)
    if (!returns_history_.empty()) {
        double prev_value = config_.initial_capital + total_pnl_ - unrealized_pnl_;
        double current_value = config_.initial_capital + total_pnl_;
        if (prev_value > 0) {
            double ret = (current_value - prev_value) / prev_value;
            returns_history_.push_back(ret);
            
            // Keep only last 252 days (trading days in a year)
            if (returns_history_.size() > 252) {
                returns_history_.erase(returns_history_.begin());
            }
        }
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

shared_ptr<Order> TradingStrategy::createOrder(const string& symbol, Side side, 
                                              OrderType type, double quantity, double price) {
    if (!canPlaceOrder(symbol, side, quantity, price)) {
        return nullptr;
    }
    
    // Generate order ID (in a real system, this would come from the exchange)
    static OrderId next_order_id = 1;
    
    auto order = make_shared<Order>(next_order_id++, side, type, price, quantity, 
                                   symbol, config_.trader_id);
    
    // Update position tracking
    double position_change = (side == Side::BUY) ? quantity : -quantity;
    updatePosition(symbol, position_change);
    
    return order;
}

bool TradingStrategy::canPlaceOrder(const string& symbol, Side side, 
                                   double quantity, double price) const {
    if (!active_ || !simulator_) return false;
    
    // Check if symbol is in our allowed symbols
    if (find(config_.symbols.begin(), config_.symbols.end(), symbol) == config_.symbols.end()) {
        return false;
    }
    
    // Check position limits
    double current_position = current_positions_.at(symbol);
    double new_position = current_position + ((side == Side::BUY) ? quantity : -quantity);
    
    if (abs(new_position) > position_limits_.at(symbol)) {
        return false;
    }
    
    // Check risk limits
    if (!checkRiskLimits()) {
        return false;
    }
    
    return true;
}

void TradingStrategy::updatePosition(const string& symbol, double quantity) {
    current_positions_[symbol] += quantity;
}

double TradingStrategy::calculatePnL(const string& symbol, double current_price) const {
    if (!simulator_ || !simulator_->hasTrader(config_.trader_id)) return 0.0;
    
    auto& trader = simulator_->getTrader(config_.trader_id);
    double position = trader.getInventory(symbol);
    double avg_cost = trader.getAverageCost(symbol);
    
    if (position == 0 || avg_cost == 0) return 0.0;
    
    return (current_price - avg_cost) * position;
}

// ============================================================================
// Technical Analysis Helpers
// ============================================================================

double TradingStrategy::calculateMovingAverage(const vector<double>& prices, int period) const {
    if (prices.size() < period) return 0.0;
    
    double sum = 0.0;
    for (int i = prices.size() - period; i < prices.size(); ++i) {
        sum += prices[i];
    }
    return sum / period;
}

double TradingStrategy::calculateVolatility(const vector<double>& prices, int period) const {
    if (prices.size() < period + 1) return 0.0;
    
    vector<double> returns;
    for (int i = prices.size() - period; i < prices.size(); ++i) {
        if (i > 0 && prices[i-1] > 0) {
            returns.push_back((prices[i] - prices[i-1]) / prices[i-1]);
        }
    }
    
    if (returns.empty()) return 0.0;
    
    double mean_return = accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    
    return sqrt(variance);
}

double TradingStrategy::calculateCorrelation(const vector<double>& prices1, 
                                            const vector<double>& prices2) const {
    if (prices1.size() != prices2.size() || prices1.size() < 2) return 0.0;
    
    // Calculate returns
    vector<double> returns1, returns2;
    for (int i = 1; i < prices1.size(); ++i) {
        if (prices1[i-1] > 0 && prices2[i-1] > 0) {
            returns1.push_back((prices1[i] - prices1[i-1]) / prices1[i-1]);
            returns2.push_back((prices2[i] - prices2[i-1]) / prices2[i-1]);
        }
    }
    
    if (returns1.size() < 2) return 0.0;
    
    double mean1 = accumulate(returns1.begin(), returns1.end(), 0.0) / returns1.size();
    double mean2 = accumulate(returns2.begin(), returns2.end(), 0.0) / returns2.size();
    
    double numerator = 0.0;
    double denom1 = 0.0;
    double denom2 = 0.0;
    
    for (int i = 0; i < returns1.size(); ++i) {
        double diff1 = returns1[i] - mean1;
        double diff2 = returns2[i] - mean2;
        numerator += diff1 * diff2;
        denom1 += diff1 * diff1;
        denom2 += diff2 * diff2;
    }
    
    double denominator = sqrt(denom1 * denom2);
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

// ============================================================================
// Private Methods
// ============================================================================

void TradingStrategy::initializeRiskMetrics() {
    returns_history_.clear();
    total_pnl_ = 0.0;
    realized_pnl_ = 0.0;
    unrealized_pnl_ = 0.0;
    max_drawdown_ = 0.0;
    peak_value_ = config_.initial_capital;
}

} // namespace deepquote 