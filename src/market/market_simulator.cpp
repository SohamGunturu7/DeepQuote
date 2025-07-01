#include "market/market_simulator.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

MarketSimulator::MarketSimulator(const vector<string>& symbols) 
    : market_events_enabled_(false), last_update_time_(0.0) {
    
    for (const auto& symbol : symbols) {
        initializeEngine(symbol);
    }
    
    // Initialize market events system
    event_generator_ = std::make_unique<MarketEventGenerator>(symbols);
    
    // Initialize microstructure noise for each symbol
    for (const auto& symbol : symbols) {
        noise_generators_[symbol] = MicrostructureNoise(0.001, 0.1);
    }
}

// ============================================================================
// Order Processing
// ============================================================================

vector<Trade> MarketSimulator::processOrder(shared_ptr<Order> order) {
    if (!isValidOrder(order)) {
        throw invalid_argument("Invalid order for market simulator");
    }
    
    // Find the matching engine for this symbol
    auto it = engines_.find(order->symbol);
    if (it == engines_.end()) {
        throw invalid_argument("Symbol not found: " + order->symbol);
    }
    
    // Track the order for trader position updates
    trackOrder(order);
    
    // Set up trade callback to capture all trades
    it->second->setTradeCallback([this](const Trade& trade) {
        onTrade(trade);
    });
    
    // Process the order
    return it->second->processOrder(order);
}

vector<Trade> MarketSimulator::processOrders(const vector<shared_ptr<Order>>& orders) {
    vector<Trade> all_trades;
    
    for (const auto& order : orders) {
        auto trades = processOrder(order);
        all_trades.insert(all_trades.end(), trades.begin(), trades.end());
    }
    
    return all_trades;
}

// ============================================================================
// Market Data Access
// ============================================================================

OrderBookSnapshot MarketSimulator::getSnapshot(const string& symbol) const {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        throw invalid_argument("Symbol not found: " + symbol);
    }
    return it->second->getSnapshot();
}

vector<OrderBookSnapshot> MarketSimulator::getAllSnapshots() const {
    vector<OrderBookSnapshot> snapshots;
    snapshots.reserve(engines_.size());
    
    for (const auto& pair : engines_) {
        snapshots.push_back(pair.second->getSnapshot());
    }
    
    return snapshots;
}

Price MarketSimulator::getBestBid(const string& symbol) const {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        return 0.0;
    }
    return it->second->getBestBid();
}

Price MarketSimulator::getBestAsk(const string& symbol) const {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        return 0.0;
    }
    return it->second->getBestAsk();
}

Price MarketSimulator::getMidPrice(const string& symbol) const {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        return 0.0;
    }
    return it->second->getMidPrice();
}

Price MarketSimulator::getSpread(const string& symbol) const {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        return 0.0;
    }
    return it->second->getSpread();
}

// ============================================================================
// Order Book Access
// ============================================================================

MatchingEngine& MarketSimulator::getMatchingEngine(const string& symbol) {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        throw invalid_argument("Symbol not found: " + symbol);
    }
    return *(it->second);
}

const MatchingEngine& MarketSimulator::getMatchingEngine(const string& symbol) const {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        throw invalid_argument("Symbol not found: " + symbol);
    }
    return *(it->second);
}

// ============================================================================
// Symbol Management
// ============================================================================

vector<string> MarketSimulator::getSymbols() const {
    vector<string> symbols;
    symbols.reserve(engines_.size());
    
    for (const auto& pair : engines_) {
        symbols.push_back(pair.first);
    }
    
    return symbols;
}

bool MarketSimulator::hasSymbol(const string& symbol) const {
    return engines_.find(symbol) != engines_.end();
}

size_t MarketSimulator::getSymbolCount() const {
    return engines_.size();
}

// ============================================================================
// Trade Callbacks
// ============================================================================

void MarketSimulator::setGlobalTradeCallback(TradeCallback callback) {
    global_trade_callback_ = callback;
}

void MarketSimulator::setSymbolTradeCallback(const string& symbol, TradeCallback callback) {
    auto it = engines_.find(symbol);
    if (it == engines_.end()) {
        throw invalid_argument("Symbol not found: " + symbol);
    }
    it->second->setTradeCallback(callback);
}

// ============================================================================
// Market Statistics
// ============================================================================

size_t MarketSimulator::getTotalOrderCount() const {
    size_t total = 0;
    for (const auto& pair : engines_) {
        total += pair.second->getOrderCount();
    }
    return total;
}

size_t MarketSimulator::getTotalTradeCount() const {
    return all_trades_.size();
}

vector<Trade> MarketSimulator::getAllTrades() const {
    return all_trades_;
}

// ============================================================================
// Trader Management
// ============================================================================

void MarketSimulator::registerTrader(const string& trader_id, double initial_cash) {
    if (trader_id.empty()) {
        throw invalid_argument("Trader ID cannot be empty");
    }
    
    if (traders_.find(trader_id) != traders_.end()) {
        throw invalid_argument("Trader already exists: " + trader_id);
    }
    
    traders_[trader_id] = std::make_shared<Trader>(trader_id, initial_cash);
}

void MarketSimulator::unregisterTrader(const string& trader_id) {
    auto it = traders_.find(trader_id);
    if (it == traders_.end()) {
        throw invalid_argument("Trader not found: " + trader_id);
    }
    
    traders_.erase(it);
}

bool MarketSimulator::hasTrader(const string& trader_id) const {
    return traders_.find(trader_id) != traders_.end();
}

vector<string> MarketSimulator::getTraderIds() const {
    vector<string> trader_ids;
    trader_ids.reserve(traders_.size());
    
    for (const auto& pair : traders_) {
        trader_ids.push_back(pair.first);
    }
    
    return trader_ids;
}

size_t MarketSimulator::getTraderCount() const {
    return traders_.size();
}

Trader& MarketSimulator::getTrader(const string& trader_id) {
    auto it = traders_.find(trader_id);
    if (it == traders_.end()) {
        throw invalid_argument("Trader not found: " + trader_id);
    }
    return *(it->second);
}

const Trader& MarketSimulator::getTrader(const string& trader_id) const {
    auto it = traders_.find(trader_id);
    if (it == traders_.end()) {
        throw invalid_argument("Trader not found: " + trader_id);
    }
    return *(it->second);
}

unordered_map<string, double> MarketSimulator::getMarkPrices() const {
    return mark_prices_;
}

void MarketSimulator::updateMarkPrices(const unordered_map<string, double>& mark_prices) {
    mark_prices_ = mark_prices;
}

void MarketSimulator::markAllTradersToMarket() {
    for (auto& pair : traders_) {
        pair.second->markToMarket(mark_prices_);
    }
    
    // Also mark RL traders to market
    for (auto& pair : rl_traders_) {
        pair.second->markToMarket(mark_prices_);
    }
}

double MarketSimulator::getTotalMarketValue() const {
    double total = 0.0;
    for (const auto& pair : traders_) {
        total += pair.second->getCash();
        
        // Add unrealized value of positions
        for (const auto& pos_pair : pair.second->getPositions()) {
            const string& symbol = pos_pair.first;
            const Position& pos = pos_pair.second;
            
            auto it = mark_prices_.find(symbol);
            if (it != mark_prices_.end() && !std::isnan(it->second)) {
                total += pos.quantity * it->second;
            }
        }
    }
    return total;
}

double MarketSimulator::getTotalRealizedPnL() const {
    double total = 0.0;
    for (const auto& pair : traders_) {
        total += pair.second->getRealizedPnL();
    }
    return total;
}

double MarketSimulator::getTotalUnrealizedPnL() const {
    double total = 0.0;
    for (const auto& pair : traders_) {
        total += pair.second->getUnrealizedPnL(mark_prices_);
    }
    return total;
}

// ============================================================================
// RL Trader Management
// ============================================================================

void MarketSimulator::addRLTrader(const shared_ptr<RLTrader>& rl_trader) {
    if (!rl_trader) {
        throw invalid_argument("RL trader cannot be null");
    }
    
    const string& agent_id = rl_trader->getAgentId();
    if (agent_id.empty()) {
        throw invalid_argument("RL trader agent ID cannot be empty");
    }
    
    if (rl_traders_.find(agent_id) != rl_traders_.end()) {
        throw invalid_argument("RL trader already exists: " + agent_id);
    }
    
    rl_traders_[agent_id] = rl_trader;
}

void MarketSimulator::removeRLTrader(const string& agent_id) {
    auto it = rl_traders_.find(agent_id);
    if (it == rl_traders_.end()) {
        throw invalid_argument("RL trader not found: " + agent_id);
    }
    
    rl_traders_.erase(it);
}

bool MarketSimulator::hasRLTrader(const string& agent_id) const {
    return rl_traders_.find(agent_id) != rl_traders_.end();
}

vector<string> MarketSimulator::getRLTraderIds() const {
    vector<string> agent_ids;
    agent_ids.reserve(rl_traders_.size());
    
    for (const auto& pair : rl_traders_) {
        agent_ids.push_back(pair.first);
    }
    
    return agent_ids;
}

size_t MarketSimulator::getRLTraderCount() const {
    return rl_traders_.size();
}

shared_ptr<RLTrader> MarketSimulator::getRLTrader(const string& agent_id) {
    auto it = rl_traders_.find(agent_id);
    if (it == rl_traders_.end()) {
        throw invalid_argument("RL trader not found: " + agent_id);
    }
    return it->second;
}

shared_ptr<const RLTrader> MarketSimulator::getRLTrader(const string& agent_id) const {
    auto it = rl_traders_.find(agent_id);
    if (it == rl_traders_.end()) {
        throw invalid_argument("RL trader not found: " + agent_id);
    }
    return it->second;
}

vector<shared_ptr<RLTrader>> MarketSimulator::getAllRLTraders() {
    vector<shared_ptr<RLTrader>> result;
    result.reserve(rl_traders_.size());
    
    for (auto& pair : rl_traders_) {
        result.push_back(pair.second);
    }
    
    return result;
}

vector<shared_ptr<const RLTrader>> MarketSimulator::getAllRLTraders() const {
    vector<shared_ptr<const RLTrader>> result;
    result.reserve(rl_traders_.size());
    
    for (const auto& pair : rl_traders_) {
        result.push_back(pair.second);
    }
    
    return result;
}

unordered_map<string, RLTraderStats> MarketSimulator::getAllRLTraderStats() const {
    unordered_map<string, RLTraderStats> stats;
    
    for (const auto& pair : rl_traders_) {
        stats[pair.first] = pair.second->getStats();
    }
    
    return stats;
}

void MarketSimulator::resetAllRLEpisodes() {
    for (auto& pair : rl_traders_) {
        pair.second->resetEpisode();
    }
}

void MarketSimulator::addRewardToRLTrader(const string& agent_id, double reward) {
    auto it = rl_traders_.find(agent_id);
    if (it == rl_traders_.end()) {
        throw invalid_argument("RL trader not found: " + agent_id);
    }
    
    it->second->addReward(reward);
}

// ============================================================================
// Utility Methods
// ============================================================================

bool MarketSimulator::isEmpty() const {
    return getTotalOrderCount() == 0;
}

void MarketSimulator::reset() {
    // Clear all trades
    all_trades_.clear();
    
    // Reset all matching engines
    for (auto& pair : engines_) {
        // Create a new matching engine for each symbol
        pair.second = make_unique<MatchingEngine>(pair.first);
    }
    
    // Reset mark prices
    mark_prices_.clear();
    
    // Clear order tracking
    order_to_trader_.clear();
}

void MarketSimulator::resetTraders() {
    traders_.clear();
}

void MarketSimulator::resetRLTraders() {
    rl_traders_.clear();
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void MarketSimulator::initializeEngine(const string& symbol) {
    if (symbol.empty()) {
        throw invalid_argument("Symbol cannot be empty");
    }
    
    if (engines_.find(symbol) != engines_.end()) {
        throw invalid_argument("Symbol already exists: " + symbol);
    }
    
    engines_[symbol] = make_unique<MatchingEngine>(symbol);
}

void MarketSimulator::onTrade(const Trade& trade) {
    // Store the trade
    all_trades_.push_back(trade);
    
    // Update trader positions
    updateTraderPositions(trade);
    
    // Update RL trader positions
    updateRLTraderPositions(trade);
    
    // Call global callback if set
    if (global_trade_callback_) {
        global_trade_callback_(trade);
    }
}

void MarketSimulator::updateTraderPositions(const Trade& trade) {
    // Find the traders involved in this trade using order IDs
    auto buyer_it = order_to_trader_.find(trade.buy_order_id);
    auto seller_it = order_to_trader_.find(trade.sell_order_id);
    
    // Update buyer's position
    if (buyer_it != order_to_trader_.end() && hasTrader(buyer_it->second)) {
        getTrader(buyer_it->second).onTrade(trade, true, 0.0); // true = is_buyer
    }
    
    // Update seller's position
    if (seller_it != order_to_trader_.end() && hasTrader(seller_it->second)) {
        getTrader(seller_it->second).onTrade(trade, false, 0.0); // false = is_seller
    }
}

void MarketSimulator::updateRLTraderPositions(const Trade& trade) {
    // Find the RL traders involved in this trade using order IDs
    auto buyer_it = order_to_trader_.find(trade.buy_order_id);
    auto seller_it = order_to_trader_.find(trade.sell_order_id);
    
    // Update buyer's position
    if (buyer_it != order_to_trader_.end() && hasRLTrader(buyer_it->second)) {
        getRLTrader(buyer_it->second)->onTrade(trade, true, 0.0); // true = is_buyer
    }
    
    // Update seller's position
    if (seller_it != order_to_trader_.end() && hasRLTrader(seller_it->second)) {
        getRLTrader(seller_it->second)->onTrade(trade, false, 0.0); // false = is_seller
    }
}

bool MarketSimulator::isValidOrder(const shared_ptr<Order>& order) const {
    if (!order || !order->isValid()) {
        return false;
    }
    
    // Check if the symbol exists
    if (!hasSymbol(order->symbol)) {
        return false;
    }
    
    return true;
}

void MarketSimulator::trackOrder(const shared_ptr<Order>& order) {
    if (order && !order->trader_id.empty()) {
        order_to_trader_[order->id] = order->trader_id;
    }
}

std::shared_ptr<Trader> MarketSimulator::getTraderPtr(const std::string& trader_id) {
    auto it = traders_.find(trader_id);
    if (it != traders_.end()) {
        return it->second;
    }
    return nullptr;
}

// ============================================================================
// Market Events and Price Movement
// ============================================================================

void MarketSimulator::enableMarketEvents(bool enable) {
    market_events_enabled_ = enable;
    if (enable) {
        std::cout << "Market events enabled - realistic price movements and random events active" << std::endl;
    } else {
        std::cout << "Market events disabled" << std::endl;
    }
}

void MarketSimulator::setEventProbability(double probability) {
    if (event_generator_) {
        event_generator_->setBaseEventProbability(probability);
    }
}

void MarketSimulator::updateMarketEvents(double dt) {
    if (!market_events_enabled_ || !event_generator_) {
        return;
    }
    
    // Update the event generator
    event_generator_->update(dt);
    
    // Generate price movements based on active events
    generatePriceMovement(dt);
}

vector<MarketEvent> MarketSimulator::getActiveEvents() const {
    if (event_generator_) {
        return event_generator_->getActiveEvents();
    }
    return {};
}

size_t MarketSimulator::getActiveEventCount() const {
    if (event_generator_) {
        return event_generator_->getActiveEventCount();
    }
    return 0;
}

void MarketSimulator::generatePriceMovement(double dt) {
    if (!market_events_enabled_ || !event_generator_) {
        return;
    }
    
    // Get current time
    auto now = std::chrono::steady_clock::now();
    double current_time = std::chrono::duration<double>(now.time_since_epoch()).count();
    
    if (last_update_time_ == 0.0) {
        last_update_time_ = current_time;
        return;
    }
    
    double actual_dt = current_time - last_update_time_;
    last_update_time_ = current_time;
    
    // Generate price movements for each symbol
    for (const auto& symbol : getSymbols()) {
        // Get current mid price
        double current_price = getMidPrice(symbol);
        if (current_price <= 0.0) {
            current_price = 100.0; // Default price if no orders
        }
        
        // Generate price change from event generator
        double price_change = event_generator_->generatePriceChange(symbol, current_price, actual_dt);
        
        // Add microstructure noise
        auto noise_it = noise_generators_.find(symbol);
        if (noise_it != noise_generators_.end()) {
            double noise = noise_it->second.generateNoise(actual_dt);
            price_change += noise * current_price;
        }
        
        // Apply price change by creating market orders
        if (std::abs(price_change) > 0.001 * current_price) { // Only if change is significant
            // Create a "market maker" order to absorb the price movement
            auto order = std::make_shared<Order>();
            order->id = 999999; // Special ID for market events
            order->symbol = symbol;
            order->trader_id = "market_events";
            order->strategy_id = "price_movement";
            order->type = OrderType::MARKET;
            order->quantity = 100.0; // Large quantity to move price
            order->price = 0.0; // Market order
            
            if (price_change > 0) {
                // Price going up - create buy order
                order->side = Side::BUY;
            } else {
                // Price going down - create sell order
                order->side = Side::SELL;
                order->quantity = -order->quantity;
            }
            
            // Process the order (this will move the price)
            try {
                processOrder(order);
            } catch (...) {
                // Ignore errors from market event orders
            }
        }
    }
}

void MarketSimulator::setPriceVolatility(const string& symbol, double volatility) {
    // This would update the price model volatility
    // For now, we'll just store it for future use
    std::cout << "Set volatility for " << symbol << " to " << volatility << std::endl;
}

void MarketSimulator::setPriceDrift(const string& symbol, double drift) {
    // This would update the price model drift
    // For now, we'll just store it for future use
    std::cout << "Set drift for " << symbol << " to " << drift << std::endl;
}

} // namespace deepquote 