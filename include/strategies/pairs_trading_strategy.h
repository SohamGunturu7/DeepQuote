#pragma once

#include "strategies/trading_strategy.h"
#include <unordered_map>
#include <utility>
using namespace std;

namespace deepquote {

// ============================================================================
// Custom Hash Function for String Pairs
// ============================================================================

struct PairHash {
    template <class T1, class T2>
    size_t operator() (const pair<T1, T2>& p) const {
        auto h1 = hash<T1>{}(p.first);
        auto h2 = hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// ============================================================================
// Pairs Trading Strategy Configuration
// ============================================================================

struct PairsTradingConfig : public StrategyConfig {
    vector<pair<string, string>> pairs;  // Pairs of symbols to trade
    int lookback_period;                 // Period for correlation calculation
    double entry_threshold;              // Z-score threshold for entry
    double exit_threshold;               // Z-score threshold for exit
    double position_size;                // Size of each position
    double correlation_threshold;        // Minimum correlation to trade pair
    bool use_cointegration;             // Whether to use cointegration test
    double cointegration_threshold;      // P-value threshold for cointegration
    bool dynamic_position_sizing;        // Whether to adjust position size based on spread
    
    PairsTradingConfig() : lookback_period(60), entry_threshold(2.0), exit_threshold(0.5),
                          position_size(50.0), correlation_threshold(0.7), use_cointegration(true),
                          cointegration_threshold(0.05), dynamic_position_sizing(true) {}
};

// ============================================================================
// Pairs Trading Strategy Implementation
// ============================================================================

class PairsTradingStrategy : public TradingStrategy {
public:
    PairsTradingStrategy(const PairsTradingConfig& config);
    ~PairsTradingStrategy() = default;
    
    // Core strategy interface
    vector<shared_ptr<Order>> generateOrders(const MarketData& data) override;
    void onTrade(const Trade& trade) override;
    void onOrderUpdate(const shared_ptr<Order>& order) override;
    
    // Strategy lifecycle
    void initialize() override;
    void update(const MarketData& data) override;
    void shutdown() override;
    
    // Pairs trading specific methods
    void addPair(const string& symbol1, const string& symbol2);
    void removePair(const string& symbol1, const string& symbol2);
    void setEntryThreshold(double threshold) { config_.entry_threshold = threshold; }
    void setExitThreshold(double threshold) { config_.exit_threshold = threshold; }
    
    // Accessors
    const PairsTradingConfig& getConfig() const { return config_; }
    double getSpreadZScore(const string& symbol1, const string& symbol2) const;
    double getCorrelation(const string& symbol1, const string& symbol2) const;
    bool isPairCointegrated(const string& symbol1, const string& symbol2) const;
    
private:
    PairsTradingConfig config_;
    
    // State tracking
    unordered_map<string, vector<double>> price_history_;
    unordered_map<pair<string, string>, double, PairHash> correlations_;
    unordered_map<pair<string, string>, double, PairHash> spread_means_;
    unordered_map<pair<string, string>, double, PairHash> spread_stds_;
    unordered_map<pair<string, string>, double, PairHash> spread_z_scores_;
    unordered_map<pair<string, string>, bool, PairHash> cointegration_status_;
    unordered_map<pair<string, string>, vector<double>, PairHash> spread_history_;
    
    // Helper methods
    void updatePriceHistory(const string& symbol, double price);
    void updateCorrelations();
    void updateSpreadStatistics();
    double calculateSpread(const string& symbol1, const string& symbol2, double price1, double price2);
    double calculateZScore(const string& symbol1, const string& symbol2, double spread);
    bool testCointegration(const string& symbol1, const string& symbol2);
    double calculatePositionSize(const string& symbol1, const string& symbol2, double z_score);
    bool shouldEnterLongShort(const string& symbol1, const string& symbol2, double z_score) const;
    bool shouldEnterShortLong(const string& symbol1, const string& symbol2, double z_score) const;
    bool shouldExitPosition(const string& symbol1, const string& symbol2, double z_score) const;
    
    // Signal generation
    vector<shared_ptr<Order>> generateEntrySignals(const MarketData& data);
    vector<shared_ptr<Order>> generateExitSignals(const MarketData& data);
    
    // Utility methods
    pair<string, string> makePair(const string& symbol1, const string& symbol2) const;
    bool hasActivePosition(const string& symbol1, const string& symbol2) const;
};

} // namespace deepquote 