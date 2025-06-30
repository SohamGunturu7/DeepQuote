#include "market/rl_trader.h"
#include "core/order.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace std;

namespace deepquote {

// ============================================================================
// RL Trader Implementation
// ============================================================================

RLTrader::RLTrader(const string& trader_id, const string& strategy_type, double initial_cash)
    : Trader(trader_id, initial_cash),
      strategy_type_(strategy_type),
      agent_id_(trader_id),
      peak_pnl_(0.0),
      episode_start_pnl_(0.0) {
    
    // Initialize stats
    stats_.cash = initial_cash;
    stats_.episode_count = 0;
    stats_.total_trades = 0;
    stats_.winning_trades = 0;
    stats_.win_rate = 0.0;
    stats_.sharpe_ratio = 0.0;
    stats_.max_drawdown = 0.0;
    stats_.current_drawdown = 0.0;
}

void RLTrader::resetEpisode() {
    // Store episode stats before resetting
    stats_.cumulative_reward += stats_.episode_reward;
    stats_.reward_history.push_back(stats_.episode_reward);
    
    // Reset episode-specific stats
    stats_.episode_reward = 0.0;
    episode_start_pnl_ = getRealizedPnL() + getUnrealizedPnL({});
    
    // Increment episode count
    stats_.episode_count++;
    
    // Update performance metrics
    updatePerformanceMetrics();
}

void RLTrader::addReward(double reward) {
    stats_.episode_reward += reward;
}

void RLTrader::onTrade(const Trade& trade, bool is_buyer, double fee) {
    // Call parent method first
    Trader::onTrade(trade, is_buyer, fee);
    
    // Update RL-specific stats
    stats_.total_trades++;
    stats_.cash = getCash();
    stats_.realized_pnl = getRealizedPnL();
    
    // Check if this was a winning trade
    double trade_pnl = (is_buyer ? -1 : 1) * trade.quantity * trade.price;
    if (trade_pnl > 0) {
        stats_.winning_trades++;
    }
    
    // Update win rate
    updateWinRate();
    
    // Update performance metrics
    updatePerformanceMetrics();
}

void RLTrader::markToMarket(const unordered_map<string, double>& mark_prices) {
    // Call parent method first
    Trader::markToMarket(mark_prices);
    
    // Update RL stats
    stats_.unrealized_pnl = getUnrealizedPnL(mark_prices);
    stats_.total_pnl = stats_.realized_pnl + stats_.unrealized_pnl;
    
    // Track P&L history for Sharpe ratio calculation
    stats_.pnl_history.push_back(stats_.total_pnl);
    
    // Keep only recent history (last 1000 points)
    if (stats_.pnl_history.size() > 1000) {
        stats_.pnl_history.erase(stats_.pnl_history.begin());
    }
    
    // Update drawdown
    updateDrawdown();
    
    // Update Sharpe ratio
    updateSharpeRatio();
}

void RLTrader::reset(double initial_cash) {
    // Call parent method first
    Trader::reset(initial_cash);
    
    // Reset RL-specific stats
    stats_.cash = initial_cash;
    stats_.realized_pnl = 0.0;
    stats_.unrealized_pnl = 0.0;
    stats_.total_pnl = 0.0;
    stats_.episode_reward = 0.0;
    stats_.cumulative_reward = 0.0;
    stats_.episode_count = 0;
    stats_.total_trades = 0;
    stats_.winning_trades = 0;
    stats_.win_rate = 0.0;
    stats_.sharpe_ratio = 0.0;
    stats_.max_drawdown = 0.0;
    stats_.current_drawdown = 0.0;
    
    // Clear history
    stats_.reward_history.clear();
    stats_.pnl_history.clear();
    stats_.returns_window.clear();
    
    // Reset tracking variables
    peak_pnl_ = 0.0;
    episode_start_pnl_ = 0.0;
}

bool RLTrader::placeOrder(const Order& order) {
    // This method will be called by the market simulator
    // For now, we just track that an order was placed
    // The actual order placement logic will be handled by the market simulator
    
    // Validate order
    if (order.quantity <= 0 || order.price <= 0) {
        return false;
    }
    
    // Check if we have enough cash for buy orders
    if (order.side == Side::BUY) {
        double required_cash = order.quantity * order.price;
        if (getCash() < required_cash) {
            return false;  // Insufficient cash
        }
    }
    
    // Check if we have enough inventory for sell orders
    if (order.side == Side::SELL) {
        double current_inventory = getInventory(order.symbol);
        if (current_inventory < order.quantity) {
            return false;  // Insufficient inventory
        }
    }
    
    return true;  // Order is valid
}

void RLTrader::cancelAllOrders() {
    // This will be implemented when we add order management
    // For now, it's a placeholder
}

void RLTrader::updatePerformanceMetrics() {
    updateWinRate();
    updateDrawdown();
    updateSharpeRatio();
}

void RLTrader::updateWinRate() {
    if (stats_.total_trades > 0) {
        stats_.win_rate = static_cast<double>(stats_.winning_trades) / stats_.total_trades;
    }
}

void RLTrader::updateDrawdown() {
    double current_pnl = stats_.total_pnl;
    
    // Update peak P&L
    if (current_pnl > peak_pnl_) {
        peak_pnl_ = current_pnl;
    }
    
    // Calculate current drawdown
    if (peak_pnl_ > 0) {
        stats_.current_drawdown = (peak_pnl_ - current_pnl) / peak_pnl_;
    } else {
        stats_.current_drawdown = 0.0;
    }
    
    // Update max drawdown
    if (stats_.current_drawdown > stats_.max_drawdown) {
        stats_.max_drawdown = stats_.current_drawdown;
    }
}

void RLTrader::updateSharpeRatio() {
    stats_.sharpe_ratio = calculateSharpeRatio();
}

double RLTrader::calculateSharpeRatio() const {
    if (stats_.pnl_history.size() < 2) {
        return 0.0;
    }
    
    // Calculate returns
    vector<double> returns;
    for (size_t i = 1; i < stats_.pnl_history.size(); ++i) {
        double return_val = stats_.pnl_history[i] - stats_.pnl_history[i-1];
        returns.push_back(return_val);
    }
    
    if (returns.empty()) {
        return 0.0;
    }
    
    // Calculate mean and standard deviation
    double mean_return = accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();
    
    double std_dev = sqrt(variance);
    
    // Calculate Sharpe ratio (assuming risk-free rate of 0)
    if (std_dev > 0) {
        return mean_return / std_dev;
    }
    
    return 0.0;
}

double RLTrader::calculateDrawdown() const {
    if (stats_.pnl_history.empty()) {
        return 0.0;
    }
    
    double peak = stats_.pnl_history[0];
    double max_drawdown = 0.0;
    
    for (double pnl : stats_.pnl_history) {
        if (pnl > peak) {
            peak = pnl;
        }
        
        double drawdown = (peak - pnl) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }
    
    return max_drawdown;
}

} // namespace deepquote 