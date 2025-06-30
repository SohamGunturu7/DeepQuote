#pragma once

#include "core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
using namespace std;

namespace deepquote {

class Trade;

// ============================================================================
// Position Structure
// ============================================================================

struct Position {
    double quantity;      // Current position size (positive = long, negative = short)
    double average_cost;  // Average cost per unit
    double total_cost;    // Total cost basis
    
    Position() : quantity(0.0), average_cost(0.0), total_cost(0.0) {}
    Position(double qty, double cost) : quantity(qty), average_cost(cost), total_cost(qty * cost) {}
};

// ============================================================================
// Trader Class: Tracks user-specific metrics and trade history
// ============================================================================

class Trader {
public:
    Trader(const string& trader_id, double initial_cash = 0.0);
    ~Trader() = default;

    // Accessors
    const string& getId() const { return trader_id_; }
    double getCash() const;
    double getInventory(const string& symbol) const;
    double getAverageCost(const string& symbol) const;
    double getRealizedPnL() const;
    double getUnrealizedPnL(const unordered_map<string, double>& mark_prices) const;
    const vector<Trade>& getTradeHistory() const { return trade_history_; }
    const unordered_map<string, Position>& getPositions() const { return positions_; }

    // Update on trade
    void onTrade(const Trade& trade, bool is_buyer, double fee = 0.0);

    // Mark-to-market inventory for all symbols
    void markToMarket(const unordered_map<string, double>& mark_prices);

    // Reset account
    void reset(double initial_cash = 0.0);

private:
    string trader_id_;
    double cash_;
    double realized_pnl_;
    double unrealized_pnl_;
    vector<Trade> trade_history_;
    unordered_map<string, Position> positions_; // symbol -> position
    
    // Helper methods
    void updatePosition(const string& symbol, double quantity, double price, bool is_buy);
    double calculateRealizedPnL(const string& symbol, double quantity, double price, bool is_buy);
};

} // namespace deepquote 