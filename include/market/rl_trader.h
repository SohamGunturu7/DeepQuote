#pragma once

#include "market/trader.h"
#include "core/order.h"
#include "core/types.h"
#include <string>
#include <vector>
#include <deque>
#include <memory>
using namespace std;

namespace deepquote {

// ============================================================================
// RL Trader Statistics
// ============================================================================

struct RLTraderStats {
    // Basic trader stats (inherited from Trader)
    double cash;
    double realized_pnl;
    double unrealized_pnl;
    double total_pnl;
    
    // RL-specific stats
    double episode_reward;
    double cumulative_reward;
    int episode_count;
    int total_trades;
    int winning_trades;
    double win_rate;
    double sharpe_ratio;
    double max_drawdown;
    double current_drawdown;
    
    // Performance tracking
    vector<double> reward_history;
    vector<double> pnl_history;
    deque<double> returns_window;  // For rolling Sharpe calculation
    
    RLTraderStats() : cash(0.0), realized_pnl(0.0), unrealized_pnl(0.0), total_pnl(0.0),
                     episode_reward(0.0), cumulative_reward(0.0), episode_count(0),
                     total_trades(0), winning_trades(0), win_rate(0.0), sharpe_ratio(0.0),
                     max_drawdown(0.0), current_drawdown(0.0) {}
};

// ============================================================================
// RL Trader Class: Extends Trader with RL-specific functionality
// ============================================================================

class RLTrader : public Trader {
public:
    RLTrader(const string& trader_id, const string& strategy_type, double initial_cash = 100000.0);
    ~RLTrader() = default;

    // Strategy information
    const string& getStrategyType() const { return strategy_type_; }
    const string& getAgentId() const { return agent_id_; }

    // RL-specific methods
    void resetEpisode();
    void addReward(double reward);
    double getEpisodeReward() const { return stats_.episode_reward; }
    double getCumulativeReward() const { return stats_.cumulative_reward; }
    int getEpisodeCount() const { return stats_.episode_count; }
    
    // Enhanced statistics
    const RLTraderStats& getStats() const { return stats_; }
    double getSharpeRatio() const { return stats_.sharpe_ratio; }
    double getWinRate() const { return stats_.win_rate; }
    double getMaxDrawdown() const { return stats_.max_drawdown; }
    
    // Override Trader methods to track RL metrics
    void onTrade(const Trade& trade, bool is_buyer, double fee = 0.0);
    void markToMarket(const unordered_map<string, double>& mark_prices);
    void reset(double initial_cash = 100000.0);

    // RL-specific order placement
    bool placeOrder(const Order& order);
    void cancelAllOrders();
    
    // Performance analysis
    void updatePerformanceMetrics();
    double calculateSharpeRatio() const;
    double calculateDrawdown() const;

private:
    string strategy_type_;  // "PPO", "SAC", "MarketMaking", etc.
    string agent_id_;       // Unique identifier for this RL agent
    RLTraderStats stats_;   // RL-specific statistics
    
    // Performance tracking
    double peak_pnl_;       // Highest P&L reached
    double episode_start_pnl_; // P&L at start of episode
    
    // Helper methods
    void updateWinRate();
    void updateDrawdown();
    void updateSharpeRatio();
};

} // namespace deepquote 