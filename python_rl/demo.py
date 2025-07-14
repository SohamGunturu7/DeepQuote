#!/usr/bin/env python3
"""
DeepQuote Comprehensive Demo
"""

import time
import numpy as np
import deepquote_simulator as dq
from agents import MeanReversionAgent, MomentumAgent, MarketMakingAgent
from deepquote_env import DeepQuoteEnv

# Setup market simulator with market maker and events
def setup_market():
    print("Setting up market simulator...")
    
    symbols = ["AAPL"]
    sim = dq.MarketSimulator(symbols)
    
    sim.enable_market_events(True)
    sim.set_event_probability(0.015)
    
    config = dq.MarketMakerConfig()
    config.trader_id = "market_maker"
    config.symbols = symbols
    config.base_price = 100.0
    config.spread_pct = 0.001
    config.order_size = 30.0
    config.max_orders_per_side = 3
    config.update_interval_ms = 150
    config.adaptive_spread = True
    
    market_maker = dq.MarketMaker(sim, config)
    market_maker.start()
    
    time.sleep(1)
    
    return sim, market_maker

# Create and register traders with different strategies
def create_traders(sim):
    print("Creating traders with different strategies...")
    
    traders = []
    
    env1 = DeepQuoteEnv(symbols=["AAPL"], trader_id="meanrev", strategy_type="RL")
    trader1 = MeanReversionAgent(env1, entry_threshold=1.0, exit_threshold=0.2)
    traders.append(("MeanReversion", trader1, env1))
    sim.add_rl_trader(env1.trader)
    
    env2 = DeepQuoteEnv(symbols=["AAPL"], trader_id="momentum", strategy_type="RL")
    trader2 = MomentumAgent(env2, momentum_threshold=0.001, position_size=0.5)
    traders.append(("Momentum", trader2, env2))
    sim.add_rl_trader(env2.trader)
    
    env3 = DeepQuoteEnv(symbols=["AAPL"], trader_id="mm_agent", strategy_type="RL")
    trader3 = MarketMakingAgent(env3, order_size=15.0)
    traders.append(("MarketMaking", trader3, env3))
    sim.add_rl_trader(env3.trader)
    
    return traders

# Run the main trading simulation
def run_trading_simulation(sim, market_maker, traders, duration_seconds=30):
    print(f"\nStarting trading simulation for {duration_seconds} seconds...")
    print("Watch for actual trades being executed!")
    print("-" * 80)
    
    total_trades = 0
    trader_trades = {name: 0 for name, _, _ in traders}
    last_trade_count = 0
    start_time = time.time()
    
    print(f"Initial Market State:")
    for symbol in sim.get_symbols():
        best_bid = sim.get_best_bid(symbol)
        best_ask = sim.get_best_ask(symbol)
        spread = best_ask - best_bid
        print(f"  {symbol}: Bid=${best_bid:.2f}, Ask=${best_ask:.2f}, Spread=${spread:.2f}")
    
    print(f"\nInitial Trader Positions:")
    for trader_name, _, env in traders:
        trader = env.trader
        cash = trader.get_cash()
        inventory = trader.get_inventory("AAPL")
        pnl = trader.get_realized_pnl()
        print(f"  {trader_name:15s}: Cash=${cash:8.2f}, Inv={inventory:6.1f}, PnL=${pnl:6.2f}")
    
    print("\n" + "=" * 80)
    print("TRADING ACTIVITY")
    print("=" * 80)
    
    step = 0
    while time.time() - start_time < duration_seconds:
        sim.update_market_events(0.2)
        
        for trader_name, trader, env in traders:
            obs = env._get_obs()
            action = trader.get_action(obs)
            
            obs, reward, done, info = env.step(action)
            
            current_trades = sim.get_total_trade_count()
            if current_trades > last_trade_count:
                new_trades = current_trades - last_trade_count
                total_trades += new_trades
                trader_trades[trader_name] += new_trades
                last_trade_count = current_trades
                
                print(f"\n💰 TRADE EXECUTED! Step {step}")
                print(f"   Trader: {trader_name}")
                print(f"   Action: {action[:4]}")
                print(f"   Reward: {reward:.2f}")
                print(f"   PnL: {info.get('pnl', 0):.2f}")
                print(f"   Inventory: {info.get('inventory', 0):.1f}")
        
        elapsed = time.time() - start_time
        if step % 50 == 0 and step > 0:
            print(f"\n--- {elapsed:.1f}s elapsed ---")
            
            for symbol in sim.get_symbols():
                best_bid = sim.get_best_bid(symbol)
                best_ask = sim.get_best_ask(symbol)
                spread = best_ask - best_bid
                print(f"  {symbol}: Bid=${best_bid:.2f}, Ask=${best_ask:.2f}, Spread=${spread:.2f}")
            
            print(f"Trader Positions:")
            for trader_name, _, env in traders:
                trader = env.trader
                cash = trader.get_cash()
                inventory = trader.get_inventory("AAPL")
                pnl = trader.get_realized_pnl()
                print(f"  {trader_name:15s}: Cash=${cash:8.2f}, Inv={inventory:6.1f}, PnL=${pnl:6.2f}")
        
        step += 1
        time.sleep(0.2)
    
    return total_trades, trader_trades

# Print comprehensive final results
def print_final_results(sim, market_maker, traders, total_trades, trader_trades):
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"Trading Activity:")
    print(f"  Total Trades Executed: {total_trades}")
    print(f"  Trades by Trader:")
    for trader_name, trade_count in trader_trades.items():
        print(f"    {trader_name}: {trade_count} trades")
    
    print(f"\nTrader Performance:")
    print(f"{'Trader':<15} {'Cash':<10} {'Inventory':<10} {'PnL':<10} {'Trades':<8} {'Return':<10}")
    print("-" * 75)
    
    for trader_name, _, env in traders:
        trader = env.trader
        cash = trader.get_cash()
        inventory = trader.get_inventory("AAPL")
        pnl = trader.get_realized_pnl()
        trades = trader_trades[trader_name]
        initial_cash = 100000.0
        return_pct = ((cash + inventory * sim.get_mid_price("AAPL")) - initial_cash) / initial_cash * 100
        
        print(f"{trader_name:<15} ${cash:<9.2f} {inventory:<10.1f} ${pnl:<9.2f} {trades:<8} {return_pct:<9.2f}%")
    
    mm_stats = market_maker.get_stats()
    print(f"\nMarket Maker Performance:")
    print(f"  Cash: ${mm_stats.cash:.2f}")
    print(f"  PnL: ${mm_stats.total_pnl:.2f}")
    print(f"  Active Orders: {mm_stats.active_orders}")
    
    print(f"\nMarket Statistics:")
    for symbol in sim.get_symbols():
        best_bid = sim.get_best_bid(symbol)
        best_ask = sim.get_best_ask(symbol)
        mid_price = sim.get_mid_price(symbol)
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100
        
        print(f"  {symbol}:")
        print(f"    Mid Price: ${mid_price:.2f}")
        print(f"    Spread: ${spread:.2f} ({spread_pct:.3f}%)")
        print(f"    Bid Depth: {sim.get_bid_depth(symbol)}")
        print(f"    Ask Depth: {sim.get_ask_depth(symbol)}")

# Manual trading demo
def demo_manual_trading():
    print("Manual Trading Demo")
    print("=" * 50)
    
    sim = dq.MarketSimulator(["AAPL"])
    sim.enable_market_events(True)
    
    config = dq.MarketMakerConfig()
    config.trader_id = "mm"
    config.symbols = ["AAPL"]
    config.base_price = 100.0
    config.spread_pct = 0.002
    config.order_size = 50.0
    
    market_maker = dq.MarketMaker(sim, config)
    market_maker.start()
    
    time.sleep(1)
    
    print("Market Maker is providing liquidity...")
    print(f"AAPL: Bid=${sim.get_best_bid('AAPL'):.2f}, Ask=${sim.get_best_ask('AAPL'):.2f}")
    
    print("\nYou can now manually place orders:")
    print("1. Market buy: sim.place_market_buy('AAPL', 10)")
    print("2. Market sell: sim.place_market_sell('AAPL', 10)")
    print("3. Limit buy: sim.place_limit_buy('AAPL', 10, 99.50)")
    print("4. Limit sell: sim.place_limit_sell('AAPL', 10, 100.50)")
    print("5. Cancel all: sim.cancel_all_orders()")
    
    return sim, market_maker

# Main demo execution
def main():
    print("DeepQuote Comprehensive Demo")
    print("=" * 50)
    
    try:
        sim, market_maker = setup_market()
        traders = create_traders(sim)
        
        total_trades, trader_trades = run_trading_simulation(sim, market_maker, traders, duration_seconds=30)
        
        print_final_results(sim, market_maker, traders, total_trades, trader_trades)
        
        market_maker.stop()
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 