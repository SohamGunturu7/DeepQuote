#include "market/matching_engine.h"
#include <algorithm>
#include <stdexcept>
using namespace std;

namespace deepquote {

// ============================================================================
// Constructor
// ============================================================================

MatchingEngine::MatchingEngine(const string& symbol) : order_book_(symbol) {
}

// ============================================================================
// Order Processing
// ============================================================================

vector<Trade> MatchingEngine::processOrder(shared_ptr<Order> order) {
    if (!order || !order->isValid()) {
        throw invalid_argument("Invalid order");
    }
    
    // Route to appropriate processor based on order type
    switch (order->type) {
        case OrderType::MARKET:
            return processMarketOrder(order);
        case OrderType::LIMIT:
            return processLimitOrder(order);
        case OrderType::CANCEL:
            // Handle cancellation (not implemented yet)
            return {};
        default:
            throw invalid_argument("Unsupported order type");
    }
}

vector<Trade> MatchingEngine::processMarketOrder(shared_ptr<Order> order) {
    vector<Trade> trades;
    
    // Market orders execute immediately against the best available prices
    // until fully filled or no more liquidity
    Quantity remaining_qty = order->quantity;
    
    while (remaining_qty > QUANTITY_EPSILON) {
        // Find the best price to match against
        Price best_price = 0;
        if (order->side == Side::BUY) {
            best_price = order_book_.getBestAsk();
        } else {
            best_price = order_book_.getBestBid();
        }
        
        if (best_price <= 0) {
            // No liquidity available
            break;
        }
        
        // Get all orders at the best price
        auto orders_at_price = order_book_.getOrdersAtPrice(best_price, 
            (order->side == Side::BUY) ? Side::SELL : Side::BUY);
        
        if (orders_at_price.empty()) {
            break;
        }
        
        // Match against orders at this price level
        for (auto& resting_order : orders_at_price) {
            if (remaining_qty <= QUANTITY_EPSILON) break;
            if (!resting_order->isActive()) continue;
            
            // Calculate match quantity
            Quantity match_qty = min(remaining_qty, resting_order->getRemainingQuantity());
            
            // Create trade
            Trade trade = createTrade(
                (order->side == Side::BUY) ? order : resting_order,
                (order->side == Side::SELL) ? order : resting_order,
                best_price, match_qty
            );
            
            trades.push_back(trade);
            
            // Update quantities
            remaining_qty -= match_qty;
            order->filled_quantity += match_qty;
            resting_order->filled_quantity += match_qty;
            
            // Update order statuses
            updateOrderStatus(order);
            updateOrderStatus(resting_order);
            
            // Call trade callback if set
            if (trade_callback_) {
                trade_callback_(trade);
            }
        }
    }
    
    // Market orders are either fully filled or rejected if no liquidity
    if (remaining_qty > QUANTITY_EPSILON) {
        order->status = OrderStatus::REJECTED;
    }
    
    return trades;
}

vector<Trade> MatchingEngine::processLimitOrder(shared_ptr<Order> order) {
    // First try to match the limit order against existing orders
    vector<Trade> trades = matchOrder(order);
    
    // If order still has remaining quantity, add it to the order book
    if (order->getRemainingQuantity() > QUANTITY_EPSILON) {
        order_book_.addOrder(order);
    }
    
    return trades;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

vector<Trade> MatchingEngine::matchOrder(shared_ptr<Order> order) {
    vector<Trade> trades;
    
    while (order->getRemainingQuantity() > QUANTITY_EPSILON) {
        // Find the best price to match against
        Price best_price = 0;
        if (order->side == Side::BUY) {
            best_price = order_book_.getBestAsk();
        } else {
            best_price = order_book_.getBestBid();
        }
        
        // Check if we can match at this price
        if (best_price <= 0) break;
        
        bool can_match = false;
        if (order->side == Side::BUY) {
            can_match = order->price >= best_price;  // Buy order can match if price >= ask
        } else {
            can_match = order->price <= best_price;  // Sell order can match if price <= bid
        }
        
        if (!can_match) break;
        
        // Get orders at the best price
        auto orders_at_price = order_book_.getOrdersAtPrice(best_price, 
            (order->side == Side::BUY) ? Side::SELL : Side::BUY);
        
        if (orders_at_price.empty()) break;
        
        // Match against orders at this price level
        for (auto& resting_order : orders_at_price) {
            if (order->getRemainingQuantity() <= QUANTITY_EPSILON) break;
            if (!resting_order->isActive()) continue;
            
            // Calculate match quantity
            Quantity match_qty = min(order->getRemainingQuantity(), 
                                   resting_order->getRemainingQuantity());
            
            // Create trade
            Trade trade = createTrade(
                (order->side == Side::BUY) ? order : resting_order,
                (order->side == Side::SELL) ? order : resting_order,
                best_price, match_qty
            );
            
            trades.push_back(trade);
            
            // Update quantities
            order->filled_quantity += match_qty;
            resting_order->filled_quantity += match_qty;
            
            // Update order statuses
            updateOrderStatus(order);
            updateOrderStatus(resting_order);
            
            // Call trade callback if set
            if (trade_callback_) {
                trade_callback_(trade);
            }
        }
    }
    
    return trades;
}

Trade MatchingEngine::createTrade(shared_ptr<Order> buy_order, shared_ptr<Order> sell_order, 
                                 Price price, Quantity quantity) {
    return Trade(buy_order->id, sell_order->id, price, quantity, order_book_.getSymbol());
}

void MatchingEngine::updateOrderStatus(shared_ptr<Order> order) {
    if (order->isFullyFilled()) {
        order->status = OrderStatus::FILLED;
    } else if (order->isPartiallyFilled()) {
        order->status = OrderStatus::PARTIAL;
    }
    order->timestamp = getCurrentTimestamp();
}

bool MatchingEngine::canMatch(shared_ptr<Order> order1, shared_ptr<Order> order2) const {
    // Orders can match if they're on opposite sides and prices cross
    if (order1->side == order2->side) return false;
    
    shared_ptr<Order> buy_order = (order1->side == Side::BUY) ? order1 : order2;
    shared_ptr<Order> sell_order = (order1->side == Side::SELL) ? order1 : order2;
    
    return buy_order->price >= sell_order->price;
}

Price MatchingEngine::getMatchPrice(shared_ptr<Order> incoming, shared_ptr<Order> resting) const {
    // Price priority: resting order price takes precedence
    return resting->price;
}

} // namespace deepquote 