#pragma once

#include "strategies/trading_strategy.h"
#include <unordered_map>
using namespace std;

namespace deepquote {

// ============================================================================
// Mean Reversion Strategy Configuration
// ============================================================================

struct MeanReversionConfig : public StrategyConfig {
    int lookback_period;           // Period for moving average calculation
    double entry_threshold;        // Standard deviations from mean to enter position
    double exit_threshold;         // Standard deviations from mean to exit position
    double position_size;          // Size of each position
    bool use_bollinger_bands;      // Whether to use Bollinger Bands
    double bollinger_multiplier;   // Multiplier for Bollinger Bands
    bool dynamic_position_sizing;  // Whether to adjust position size based on deviation
    double max_position_size;      // Maximum position size as percentage of capital
    
    MeanReversionConfig() : lookback_period(20), entry_threshold(2.0), exit_threshold(0.5),
                           position_size(100.0), use_bollinger_bands(true), bollinger_multiplier(2.0),
                           dynamic_position_sizing(true), max_position_size(0.1) {}
};

// ============================================================================
// Mean Reversion Strategy Implementation
// ============================================================================

class MeanReversionStrategy : public TradingStrategy {
public:
    MeanReversionStrategy(const MeanReversionConfig& config);
    ~MeanReversionStrategy() = default;
    
    // Core strategy interface
    vector<shared_ptr<Order>> generateOrders(const MarketData& data) override;
    void onTrade(const Trade& trade) override;
    void onOrderUpdate(const shared_ptr<Order>& order) override;
    
    // Strategy lifecycle
    void initialize() override;
    void update(const MarketData& data) override;
    void shutdown() override;
    
    // Mean reversion specific methods
    void setLookbackPeriod(int period) { config_.lookback_period = period; }
    void setEntryThreshold(double threshold) { config_.entry_threshold = threshold; }
    void setExitThreshold(double threshold) { config_.exit_threshold = threshold; }
    
    // Accessors
    const MeanReversionConfig& getConfig() const { return config_; }
    double getCurrentZScore(const string& symbol) const;
    double getMovingAverage(const string& symbol) const;
    double getBollingerUpper(const string& symbol) const;
    double getBollingerLower(const string& symbol) const;
    
private:
    MeanReversionConfig config_;
    
    // State tracking
    unordered_map<string, double> moving_averages_;
    unordered_map<string, double> standard_deviations_;
    unordered_map<string, double> z_scores_;
    unordered_map<string, double> bollinger_upper_;
    unordered_map<string, double> bollinger_lower_;
    unordered_map<string, vector<double>> price_history_;
    
    // Helper methods
    void updateIndicators(const string& symbol, const vector<double>& prices);
    double calculateZScore(const string& symbol, double current_price);
    double calculatePositionSize(const string& symbol, double z_score);
    bool shouldEnterLong(const string& symbol, double z_score) const;
    bool shouldEnterShort(const string& symbol, double z_score) const;
    bool shouldExitLong(const string& symbol, double z_score) const;
    bool shouldExitShort(const string& symbol, double z_score) const;
    void updatePriceHistory(const string& symbol, double price);
    
    // Signal generation
    vector<shared_ptr<Order>> generateEntrySignals(const MarketData& data);
    vector<shared_ptr<Order>> generateExitSignals(const MarketData& data);
};

} // namespace deepquote 