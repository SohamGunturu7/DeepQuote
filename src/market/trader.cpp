#include "market/trader.h"
#include "core/order.h"
#include "core/types.h"
#include <algorithm>
#include <limits>
#include <cmath>
using namespace std;

namespace deepquote {

// Helper function to round to two decimal places
static double round2(double value) {
    return std::round(value * 100.0) / 100.0;
}

Trader::Trader(const string& trader_id, double initial_cash)
    : trader_id_(trader_id), cash_(initial_cash), realized_pnl_(0.0), unrealized_pnl_(0.0) {}

double Trader::getCash() const {
    return round2(cash_);
}

double Trader::getInventory(const string& symbol) const {
    auto it = positions_.find(symbol);
    return (it != positions_.end()) ? it->second.quantity : 0.0;
}

double Trader::getAverageCost(const string& symbol) const {
    auto it = positions_.find(symbol);
    return (it != positions_.end()) ? round2(it->second.average_cost) : 0.0;
}

double Trader::getUnrealizedPnL(const unordered_map<string, double>& mark_prices) const {
    double total = 0.0;
    for (const auto& kv : positions_) {
        const string& symbol = kv.first;
        const Position& pos = kv.second;
        
        auto it = mark_prices.find(symbol);
        if (it != mark_prices.end() && !std::isnan(it->second)) {
            double current_price = it->second;
            // Unrealized P&L = (current_price - average_cost) * quantity
            total += (current_price - pos.average_cost) * pos.quantity;
        }
    }
    return round2(total);
}

void Trader::onTrade(const Trade& trade, bool is_buyer, double fee) {
    const string& symbol = trade.symbol;
    double qty = trade.quantity;
    double price = trade.price;
    double value = qty * price;

    // Calculate realized P&L before updating position
    double realized_pnl_change = calculateRealizedPnL(symbol, qty, price, is_buyer);
    realized_pnl_ += realized_pnl_change;

    // Update cash
    if (is_buyer) {
        cash_ -= value + fee;
    } else {
        cash_ += value - fee;
    }

    // Update position
    updatePosition(symbol, qty, price, is_buyer);

    // Add to trade history
    trade_history_.push_back(trade);
}

void Trader::updatePosition(const string& symbol, double quantity, double price, bool is_buyer) {
    auto it = positions_.find(symbol);
    
    if (it == positions_.end()) {
        // New position
        if (is_buyer) {
            positions_[symbol] = Position(quantity, price);
        } else {
            positions_[symbol] = Position(-quantity, price);
        }
    } else {
        Position& pos = it->second;
        double old_qty = pos.quantity;
        double new_qty = is_buyer ? pos.quantity + quantity : pos.quantity - quantity;
        
        if (new_qty == 0.0) {
            // Position closed
            positions_.erase(it);
        } else if ((old_qty > 0 && new_qty > 0) || (old_qty < 0 && new_qty < 0)) {
            // Same direction - update average cost
            if (is_buyer) {
                // Buying more - add to total cost
                pos.total_cost += quantity * price;
                pos.quantity = new_qty;
                pos.average_cost = pos.total_cost / abs(new_qty);
            } else {
                // Selling some - keep same average cost, just reduce quantity
                pos.quantity = new_qty;
                pos.total_cost = pos.average_cost * abs(new_qty);
            }
        } else {
            // Direction change - treat as new position
            pos.quantity = new_qty;
            pos.average_cost = price;
            pos.total_cost = new_qty * price;
        }
    }
}

double Trader::calculateRealizedPnL(const string& symbol, double quantity, double price, bool is_buyer) {
    auto it = positions_.find(symbol);
    if (it == positions_.end()) {
        return 0.0; // No existing position, no realized P&L
    }
    
    const Position& pos = it->second;
    double existing_qty = pos.quantity;
    
    // Check if this trade closes or reduces a position
    if (existing_qty > 0 && !is_buyer) {
        // Selling long position
        double trade_qty = min(abs(existing_qty), quantity);
        double cost_basis = pos.average_cost;
        return trade_qty * (price - cost_basis);
    } else if (existing_qty < 0 && is_buyer) {
        // Buying to cover short position
        double trade_qty = min(abs(existing_qty), quantity);
        double cost_basis = pos.average_cost;
        return trade_qty * (cost_basis - price);
    }
    
    return 0.0; // No realized P&L for this trade
}

void Trader::markToMarket(const unordered_map<string, double>& mark_prices) {
    unrealized_pnl_ = getUnrealizedPnL(mark_prices);
}

void Trader::reset(double initial_cash) {
    cash_ = initial_cash;
    positions_.clear();
    realized_pnl_ = 0.0;
    unrealized_pnl_ = 0.0;
    trade_history_.clear();
}

double Trader::getRealizedPnL() const {
    return round2(realized_pnl_);
}

} // namespace deepquote