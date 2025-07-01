#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "core/types.h"
#include "core/order.h"
#include "market/trader.h"
#include "market/rl_trader.h"
#include "market/market_maker.h"
#include "market/market_simulator.h"
#include "market/market_events.h"
#include "market/order_book.h"

namespace py = pybind11;
using namespace deepquote;

PYBIND11_MODULE(deepquote_simulator, m) {
    m.doc() = "DeepQuote Market Simulator Python Bindings"; // Optional module docstring
    
    // ============================================================================
    // Core Types
    // ============================================================================
    
    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL)
        .export_values();
    
    py::enum_<OrderType>(m, "OrderType")
        .value("MARKET", OrderType::MARKET)
        .value("LIMIT", OrderType::LIMIT)
        .value("CANCEL", OrderType::CANCEL)
        .export_values();
    
    py::enum_<OrderStatus>(m, "OrderStatus")
        .value("PENDING", OrderStatus::PENDING)
        .value("PARTIAL", OrderStatus::PARTIAL)
        .value("FILLED", OrderStatus::FILLED)
        .value("CANCELLED", OrderStatus::CANCELLED)
        .value("REJECTED", OrderStatus::REJECTED)
        .export_values();
    
    // ============================================================================
    // Order Class
    // ============================================================================
    
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def(py::init<OrderId, Side, OrderType, Price, Quantity, const std::string&, const std::string&>())
        .def_readwrite("id", &Order::id)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("price", &Order::price)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("filled_quantity", &Order::filled_quantity)
        .def_readwrite("status", &Order::status)
        .def_readwrite("timestamp", &Order::timestamp)
        .def_readwrite("created_time", &Order::created_time)
        .def_readwrite("symbol", &Order::symbol)
        .def_readwrite("trader_id", &Order::trader_id)
        .def_readwrite("is_market_maker_order", &Order::is_market_maker_order)
        .def_readwrite("strategy_id", &Order::strategy_id)
        .def("is_valid", &Order::isValid)
        .def("is_fully_filled", &Order::isFullyFilled)
        .def("is_partially_filled", &Order::isPartiallyFilled)
        .def("is_active", &Order::isActive)
        .def("get_remaining_quantity", &Order::getRemainingQuantity)
        .def("get_fill_ratio", &Order::getFillRatio)
        .def("to_string", &Order::toString)
        .def("__str__", &Order::toString);
    
    // ============================================================================
    // Trade Class
    // ============================================================================
    
    py::class_<Trade>(m, "Trade")
        .def(py::init<>())
        .def(py::init<OrderId, OrderId, Price, Quantity, const std::string&>())
        .def_readwrite("buy_order_id", &Trade::buy_order_id)
        .def_readwrite("sell_order_id", &Trade::sell_order_id)
        .def_readwrite("price", &Trade::price)
        .def_readwrite("quantity", &Trade::quantity)
        .def_readwrite("timestamp", &Trade::timestamp)
        .def_readwrite("symbol", &Trade::symbol)
        .def_readwrite("involves_market_maker", &Trade::involves_market_maker)
        .def_readwrite("market_maker_id", &Trade::market_maker_id)
        .def("is_valid", &Trade::isValid)
        .def("to_string", &Trade::toString)
        .def("__str__", &Trade::toString);
    
    // ============================================================================
    // RLTraderStats Structure
    // ============================================================================
    
    py::class_<RLTraderStats>(m, "RLTraderStats")
        .def(py::init<>())
        .def_readwrite("cash", &RLTraderStats::cash)
        .def_readwrite("realized_pnl", &RLTraderStats::realized_pnl)
        .def_readwrite("unrealized_pnl", &RLTraderStats::unrealized_pnl)
        .def_readwrite("total_pnl", &RLTraderStats::total_pnl)
        .def_readwrite("episode_reward", &RLTraderStats::episode_reward)
        .def_readwrite("cumulative_reward", &RLTraderStats::cumulative_reward)
        .def_readwrite("episode_count", &RLTraderStats::episode_count)
        .def_readwrite("total_trades", &RLTraderStats::total_trades)
        .def_readwrite("winning_trades", &RLTraderStats::winning_trades)
        .def_readwrite("win_rate", &RLTraderStats::win_rate)
        .def_readwrite("sharpe_ratio", &RLTraderStats::sharpe_ratio)
        .def_readwrite("max_drawdown", &RLTraderStats::max_drawdown)
        .def_readwrite("current_drawdown", &RLTraderStats::current_drawdown)
        .def_readwrite("reward_history", &RLTraderStats::reward_history)
        .def_readwrite("pnl_history", &RLTraderStats::pnl_history);
    
    // ============================================================================
    // Trader Class
    // ============================================================================
    py::class_<Trader, std::shared_ptr<Trader>>(m, "Trader")
        .def(py::init<const std::string&, double>(), py::arg("trader_id"), py::arg("initial_cash") = 100000.0)
        .def("get_id", &Trader::getId)
        .def("get_cash", &Trader::getCash)
        .def("get_inventory", &Trader::getInventory)
        .def("get_average_cost", &Trader::getAverageCost)
        .def("get_realized_pnl", &Trader::getRealizedPnL)
        .def("get_unrealized_pnl", &Trader::getUnrealizedPnL)
        .def("get_positions", &Trader::getPositions)
        .def("get_trade_history", &Trader::getTradeHistory)
        .def("reset", &Trader::reset)
        .def("mark_to_market", &Trader::markToMarket);
    
    // ============================================================================
    // RLTrader Class
    // ============================================================================
    
    py::class_<RLTrader, std::shared_ptr<RLTrader>>(m, "RLTrader")
        .def(py::init<const std::string&, const std::string&, double>(),
             py::arg("trader_id"), py::arg("strategy_type"), py::arg("initial_cash") = 100000.0)
        .def("get_id", &RLTrader::getId)
        .def("get_strategy_type", &RLTrader::getStrategyType)
        .def("get_agent_id", &RLTrader::getAgentId)
        .def("get_cash", &RLTrader::getCash)
        .def("get_inventory", [](RLTrader& self, py::args args) {
            if (args.size() == 0) {
                // Return total inventory (sum over all symbols or first symbol)
                auto positions = self.getPositions();
                if (positions.empty()) {
                    return 0.0;
                }
                // Return the first symbol's inventory quantity
                return positions.begin()->second.quantity;
            } else if (args.size() == 1) {
                return self.getInventory(args[0].cast<std::string>());
            } else {
                throw std::invalid_argument("get_inventory() takes at most one argument (symbol)");
            }
        })
        .def("get_average_cost", &RLTrader::getAverageCost)
        .def("get_realized_pnl", &RLTrader::getRealizedPnL)
        .def("get_unrealized_pnl", [](RLTrader& self, py::args args) {
            if (args.size() == 0) {
                // Return 0 if no mark prices provided (simplified)
                return 0.0;
            } else if (args.size() == 1) {
                return self.getUnrealizedPnL(args[0].cast<std::unordered_map<std::string, double>>());
            } else {
                throw std::invalid_argument("get_unrealized_pnl() takes at most one argument (mark_prices)");
            }
        })
        .def("get_positions", &RLTrader::getPositions)
        .def("get_trade_history", &RLTrader::getTradeHistory)
        .def("reset_episode", &RLTrader::resetEpisode)
        .def("add_reward", &RLTrader::addReward)
        .def("get_episode_reward", &RLTrader::getEpisodeReward)
        .def("get_cumulative_reward", &RLTrader::getCumulativeReward)
        .def("get_episode_count", &RLTrader::getEpisodeCount)
        .def("get_stats", &RLTrader::getStats)
        .def("get_sharpe_ratio", &RLTrader::getSharpeRatio)
        .def("get_win_rate", &RLTrader::getWinRate)
        .def("get_max_drawdown", &RLTrader::getMaxDrawdown)
        .def("on_trade", &RLTrader::onTrade)
        .def("mark_to_market", &RLTrader::markToMarket)
        .def("reset", [](RLTrader& self, py::args args) {
            if (args.size() == 0) {
                self.reset(self.getCash());
            } else if (args.size() == 1) {
                self.reset(args[0].cast<double>());
            } else {
                throw std::invalid_argument("reset() takes at most one argument (initial_cash)");
            }
        })
        .def("place_order", &RLTrader::placeOrder)
        .def("cancel_all_orders", &RLTrader::cancelAllOrders)
        .def("update_performance_metrics", &RLTrader::updatePerformanceMetrics)
        .def("calculate_sharpe_ratio", &RLTrader::calculateSharpeRatio)
        .def("calculate_drawdown", &RLTrader::calculateDrawdown);
    
    // ============================================================================
    // MarketSimulator Class
    // ============================================================================
    
    py::class_<MarketSimulator>(m, "MarketSimulator")
        .def(py::init<const std::vector<std::string>&>())
        
        // Order processing
        .def("process_order", [](MarketSimulator& self, const Order& order) {
            return self.processOrder(std::make_shared<Order>(order));
        })
        .def("process_orders", &MarketSimulator::processOrders)
        
        // Market data access
        .def("get_snapshot", &MarketSimulator::getSnapshot)
        .def("get_all_snapshots", &MarketSimulator::getAllSnapshots)
        .def("get_best_bid", &MarketSimulator::getBestBid)
        .def("get_best_ask", &MarketSimulator::getBestAsk)
        .def("get_mid_price", &MarketSimulator::getMidPrice)
        .def("get_spread", &MarketSimulator::getSpread)
        
        // Symbol management
        .def("get_symbols", &MarketSimulator::getSymbols)
        .def("has_symbol", &MarketSimulator::hasSymbol)
        .def("get_symbol_count", &MarketSimulator::getSymbolCount)
        
        // Market statistics
        .def("get_total_order_count", &MarketSimulator::getTotalOrderCount)
        .def("get_total_trade_count", &MarketSimulator::getTotalTradeCount)
        .def("get_all_trades", &MarketSimulator::getAllTrades)
        
        // Regular trader management
        .def("register_trader", &MarketSimulator::registerTrader)
        .def("unregister_trader", &MarketSimulator::unregisterTrader)
        .def("has_trader", &MarketSimulator::hasTrader)
        .def("get_trader_ids", &MarketSimulator::getTraderIds)
        .def("get_trader_count", &MarketSimulator::getTraderCount)
        .def("get_trader", &MarketSimulator::getTraderPtr)
        
        // RL trader management
        .def("add_rl_trader", &MarketSimulator::addRLTrader)
        .def("remove_rl_trader", &MarketSimulator::removeRLTrader)
        .def("has_rl_trader", &MarketSimulator::hasRLTrader)
        .def("get_rl_trader_ids", &MarketSimulator::getRLTraderIds)
        .def("get_rl_trader_count", &MarketSimulator::getRLTraderCount)
        .def("get_rl_trader", static_cast<std::shared_ptr<RLTrader>(MarketSimulator::*)(const std::string&)>(&MarketSimulator::getRLTrader))
        .def("get_all_rl_traders", static_cast<std::vector<std::shared_ptr<RLTrader>>(MarketSimulator::*)()>(&MarketSimulator::getAllRLTraders))
        .def("get_all_rl_trader_stats", &MarketSimulator::getAllRLTraderStats)
        .def("reset_all_rl_episodes", &MarketSimulator::resetAllRLEpisodes)
        .def("add_reward_to_rl_trader", &MarketSimulator::addRewardToRLTrader)
        
        // Market events and price movement
        .def("enable_market_events", &MarketSimulator::enableMarketEvents)
        .def("set_event_probability", &MarketSimulator::setEventProbability)
        .def("update_market_events", &MarketSimulator::updateMarketEvents)
        .def("get_active_events", &MarketSimulator::getActiveEvents)
        .def("get_active_event_count", &MarketSimulator::getActiveEventCount)
        .def("generate_price_movement", &MarketSimulator::generatePriceMovement)
        .def("set_price_volatility", &MarketSimulator::setPriceVolatility)
        .def("set_price_drift", &MarketSimulator::setPriceDrift)
        
        // Market utilities
        .def("get_mark_prices", &MarketSimulator::getMarkPrices)
        .def("update_mark_prices", &MarketSimulator::updateMarkPrices)
        .def("mark_all_traders_to_market", &MarketSimulator::markAllTradersToMarket)
        .def("get_total_market_value", &MarketSimulator::getTotalMarketValue)
        .def("get_total_realized_pnl", &MarketSimulator::getTotalRealizedPnL)
        .def("get_total_unrealized_pnl", &MarketSimulator::getTotalUnrealizedPnL)
        .def("is_empty", &MarketSimulator::isEmpty)
        .def("reset", &MarketSimulator::reset)
        .def("reset_traders", &MarketSimulator::resetTraders)
        .def("reset_rl_traders", &MarketSimulator::resetRLTraders);
    
    // ============================================================================
    // MarketMakerConfig Structure
    // ============================================================================
    
    py::class_<MarketMakerConfig>(m, "MarketMakerConfig")
        .def(py::init<>())
        .def_readwrite("trader_id", &MarketMakerConfig::trader_id)
        .def_readwrite("symbols", &MarketMakerConfig::symbols)
        .def_readwrite("base_price", &MarketMakerConfig::base_price)
        .def_readwrite("spread_pct", &MarketMakerConfig::spread_pct)
        .def_readwrite("order_size", &MarketMakerConfig::order_size)
        .def_readwrite("max_orders_per_side", &MarketMakerConfig::max_orders_per_side)
        .def_readwrite("update_interval_ms", &MarketMakerConfig::update_interval_ms)
        .def_readwrite("adaptive_spread", &MarketMakerConfig::adaptive_spread)
        .def_readwrite("min_spread_pct", &MarketMakerConfig::min_spread_pct)
        .def_readwrite("max_spread_pct", &MarketMakerConfig::max_spread_pct)
        .def_readwrite("volatility_window", &MarketMakerConfig::volatility_window);
    
    // ============================================================================
    // MarketMakerStats Structure
    // ============================================================================
    
    py::class_<MarketMaker::Stats>(m, "MarketMakerStats")
        .def(py::init<>())
        .def_readwrite("trader_id", &MarketMaker::Stats::trader_id)
        .def_readwrite("cash", &MarketMaker::Stats::cash)
        .def_readwrite("total_pnl", &MarketMaker::Stats::total_pnl)
        .def_readwrite("active_orders", &MarketMaker::Stats::active_orders)
        .def_readwrite("running", &MarketMaker::Stats::running);
    
    // ============================================================================
    // MarketMaker Class
    // ============================================================================
    
    py::class_<MarketMaker>(m, "MarketMaker")
        .def(py::init<MarketSimulator*, const MarketMakerConfig&>(),
             py::arg("simulator"), py::arg("config"))
        .def("start", &MarketMaker::start)
        .def("stop", &MarketMaker::stop)
        .def("is_running", &MarketMaker::isRunning)
        .def("set_spread", &MarketMaker::setSpread)
        .def("set_order_size", &MarketMaker::setOrderSize)
        .def("set_update_interval", &MarketMaker::setUpdateInterval)
        .def("get_stats", &MarketMaker::getStats);
    
    // ============================================================================
    // Market Events
    // ============================================================================
    
    py::enum_<EventType>(m, "EventType")
        .value("PRICE_SHOCK", EventType::PRICE_SHOCK)
        .value("VOLATILITY_SPIKE", EventType::VOLATILITY_SPIKE)
        .value("LIQUIDITY_CRISIS", EventType::LIQUIDITY_CRISIS)
        .value("NEWS_EVENT", EventType::NEWS_EVENT)
        .value("MARKET_CRASH", EventType::MARKET_CRASH)
        .value("FLASH_CRASH", EventType::FLASH_CRASH)
        .value("PUMP_AND_DUMP", EventType::PUMP_AND_DUMP)
        .value("EARNINGS_ANNOUNCEMENT", EventType::EARNINGS_ANNOUNCEMENT)
        .value("FED_ANNOUNCEMENT", EventType::FED_ANNOUNCEMENT)
        .value("TECHNICAL_BREAKOUT", EventType::TECHNICAL_BREAKOUT)
        .value("CORRELATION_BREAKDOWN", EventType::CORRELATION_BREAKDOWN)
        .value("MICROSTRUCTURE_NOISE", EventType::MICROSTRUCTURE_NOISE)
        .export_values();
    
    py::class_<MarketEvent>(m, "MarketEvent")
        .def(py::init<EventType, const std::string&, double, double, const std::string&>())
        .def_readwrite("type", &MarketEvent::type)
        .def_readwrite("symbol", &MarketEvent::symbol)
        .def_readwrite("magnitude", &MarketEvent::magnitude)
        .def_readwrite("duration", &MarketEvent::duration)
        .def_readwrite("description", &MarketEvent::description)
        .def_readwrite("is_active", &MarketEvent::is_active);
    
    // ============================================================================
    // Utility Functions
    // ============================================================================
    
    m.def("get_current_timestamp", &getCurrentTimestamp, "Get current timestamp");
    m.def("timestamp_to_string", &timestampToString, "Convert timestamp to string");
    m.def("string_to_timestamp", &stringToTimestamp, "Convert string to timestamp");
    
    // ============================================================================
    // Module Documentation
    // ============================================================================
    
    m.attr("__version__") = "1.0.0";
    
    // Add some example usage
    m.def("create_sample_market", []() {
        std::vector<std::string> symbols = {"AAPL", "GOOGL"};
        return std::make_unique<MarketSimulator>(symbols);
    }, "Create a sample market simulator with AAPL and GOOGL");
    
    m.def("create_sample_rl_trader", [](const std::string& agent_id, const std::string& strategy_type) {
        return std::make_shared<RLTrader>(agent_id, strategy_type, 100000.0);
    }, py::arg("agent_id"), py::arg("strategy_type"), "Create a sample RL trader");
} 