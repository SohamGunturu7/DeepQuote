#pragma once

#include "core/types.h"
#include "core/order.h"
#include "market/market_simulator.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
using namespace std;

namespace deepquote {

// Forward declarations
class MarketSimulator;
class Trade;

// Market data structure for strategies
struct MarketData {
    unordered_map<string, OrderBookSnapshot> order_books;
    unordered_map<string, vector<double>> price_history;
    unordered_map<string, vector<double>> volume_history;
    Timestamp timestamp;
    
    unordered_map<string, double> moving_averages;
    unordered_map<string, double> volatilities;
    unordered_map<string, double> correlations;
    
    MarketData() = default;
};

// Strategy configuration
struct StrategyConfig {
    string strategy_id;
    string trader_id;
    vector<string> symbols;
    double initial_capital;
    double max_position_size;
    double max_drawdown;
    bool enable_risk_management;
    
    StrategyConfig() : initial_capital(10000.0), max_position_size(1000.0), 
                       max_drawdown(0.1), enable_risk_management(true) {}
};

// Base trading strategy class
class TradingStrategy {
public:
    TradingStrategy(const StrategyConfig& config);
    virtual ~TradingStrategy() = default;
    
    virtual vector<shared_ptr<Order>> generateOrders(const MarketData& data) = 0;
    virtual void onTrade(const Trade& trade) = 0;
    virtual void onOrderUpdate(const shared_ptr<Order>& order) = 0;
    
    virtual void initialize() = 0;
    virtual void update(const MarketData& data) = 0;
    virtual void shutdown() = 0;
    
    void setMarketSimulator(MarketSimulator* simulator) { simulator_ = simulator; }
    void setTraderId(const string& trader_id) { config_.trader_id = trader_id; }
    
    const string& getStrategyId() const { return config_.strategy_id; }
    const string& getTraderId() const { return config_.trader_id; }
    const vector<string>& getSymbols() const { return config_.symbols; }
    bool isActive() const { return active_; }
    
    double getTotalPnL() const;
    double getRealizedPnL() const;
    double getUnrealizedPnL() const;
    double getSharpeRatio() const;
    double getMaxDrawdown() const;
    
    bool checkRiskLimits() const;
    void updateRiskMetrics();
    
protected:
    StrategyConfig config_;
    MarketSimulator* simulator_;
    bool active_;
    
    vector<double> returns_history_;
    double total_pnl_;
    double realized_pnl_;
    double unrealized_pnl_;
    double max_drawdown_;
    double peak_value_;
    
    unordered_map<string, double> current_positions_;
    unordered_map<string, double> position_limits_;
    
    shared_ptr<Order> createOrder(const string& symbol, Side side, OrderType type, 
                                 double quantity, double price);
    bool canPlaceOrder(const string& symbol, Side side, double quantity, double price) const;
    void updatePosition(const string& symbol, double quantity);
    double calculatePnL(const string& symbol, double current_price) const;
    
    double calculateMovingAverage(const vector<double>& prices, int period) const;
    double calculateVolatility(const vector<double>& prices, int period) const;
    double calculateCorrelation(const vector<double>& prices1, 
                               const vector<double>& prices2) const;
    
private:
    void initializeRiskMetrics();
};

} // namespace deepquote 