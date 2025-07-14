#pragma once

#include "core/types.h"
#include "core/order.h"
#include "market/matching_engine.h"
#include "market/trader.h"
#include "market/rl_trader.h"
#include "market/market_events.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
using namespace std;

namespace deepquote {

// Market simulator implementation
class MarketSimulator {
public:
    MarketSimulator(const vector<string>& symbols);
    ~MarketSimulator() = default;
    
    vector<Trade> processOrder(shared_ptr<Order> order);
    vector<Trade> processOrders(const vector<shared_ptr<Order>>& orders);
    
    OrderBookSnapshot getSnapshot(const string& symbol) const;
    vector<OrderBookSnapshot> getAllSnapshots() const;
    
    Price getBestBid(const string& symbol) const;
    Price getBestAsk(const string& symbol) const;
    Price getMidPrice(const string& symbol) const;
    Price getSpread(const string& symbol) const;
    
    MatchingEngine& getMatchingEngine(const string& symbol);
    const MatchingEngine& getMatchingEngine(const string& symbol) const;
    
    vector<string> getSymbols() const;
    bool hasSymbol(const string& symbol) const;
    size_t getSymbolCount() const;
    
    using TradeCallback = function<void(const Trade&)>;
    void setGlobalTradeCallback(TradeCallback callback);
    void setSymbolTradeCallback(const string& symbol, TradeCallback callback);
    
    size_t getTotalOrderCount() const;
    size_t getTotalTradeCount() const;
    vector<Trade> getAllTrades() const;
    
    // Trader management
    void registerTrader(const string& trader_id, double initial_cash = 0.0);
    void unregisterTrader(const string& trader_id);
    bool hasTrader(const string& trader_id) const;
    vector<string> getTraderIds() const;
    size_t getTraderCount() const;
    
    Trader& getTrader(const string& trader_id);
    const Trader& getTrader(const string& trader_id) const;
    
    // RL trader management
    void addRLTrader(const shared_ptr<RLTrader>& rl_trader);
    void removeRLTrader(const string& agent_id);
    bool hasRLTrader(const string& agent_id) const;
    vector<string> getRLTraderIds() const;
    size_t getRLTraderCount() const;
    
    shared_ptr<RLTrader> getRLTrader(const string& agent_id);
    shared_ptr<const RLTrader> getRLTrader(const string& agent_id) const;
    vector<shared_ptr<RLTrader>> getAllRLTraders();
    vector<shared_ptr<const RLTrader>> getAllRLTraders() const;
    
    unordered_map<string, RLTraderStats> getAllRLTraderStats() const;
    void resetAllRLEpisodes();
    void addRewardToRLTrader(const string& agent_id, double reward);
    
    unordered_map<string, double> getMarkPrices() const;
    void updateMarkPrices(const unordered_map<string, double>& mark_prices);
    void markAllTradersToMarket();
    
    double getTotalMarketValue() const;
    double getTotalRealizedPnL() const;
    double getTotalUnrealizedPnL() const;
    
    // Market events and price movement
    void enableMarketEvents(bool enable = true);
    void setEventProbability(double probability);
    void updateMarketEvents(double dt);
    vector<MarketEvent> getActiveEvents() const;
    size_t getActiveEventCount() const;
    
    void generatePriceMovement(double dt);
    void setPriceVolatility(const string& symbol, double volatility);
    void setPriceDrift(const string& symbol, double drift);
    
    bool isEmpty() const;
    void reset();
    void resetTraders();
    void resetRLTraders();
    
    std::shared_ptr<Trader> getTraderPtr(const std::string& trader_id);
    
private:
    unordered_map<string, unique_ptr<MatchingEngine>> engines_;
    unordered_map<string, shared_ptr<Trader>> traders_;
    unordered_map<string, shared_ptr<RLTrader>> rl_traders_;
    unordered_map<string, double> mark_prices_;
    unordered_map<OrderId, string> order_to_trader_;
    TradeCallback global_trade_callback_;
    vector<Trade> all_trades_;
    
    unique_ptr<MarketEventGenerator> event_generator_;
    unordered_map<string, MicrostructureNoise> noise_generators_;
    bool market_events_enabled_;
    double last_update_time_;
    
    void initializeEngine(const string& symbol);
    void onTrade(const Trade& trade);
    bool isValidOrder(const shared_ptr<Order>& order) const;
    void updateTraderPositions(const Trade& trade);
    void trackOrder(const shared_ptr<Order>& order);
    void updateRLTraderPositions(const Trade& trade);
};

} // namespace deepquote 