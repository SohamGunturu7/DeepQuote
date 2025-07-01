#!/usr/bin/env python3
"""
DeepQuote Comprehensive Demo

This demo showcases all the key features of DeepQuote:
- C++ Market Maker providing liquidity
- RL Agents trading with different strategies
- Real market events driving price movements
- Actual trade execution and position tracking
- Performance monitoring and analysis
"""

import time
import numpy as np
import deepquote_simulator as dq
from agents import MeanReversionAgent, MomentumAgent, MarketMakingAgent
from deepquote_env import DeepQuoteEnv

def setup_market():
    """Setup market simulator with market maker and events"""
    print("Setting up market simulator...")
    
    # Create market simulator
    symbols = ["AAPL"]
    sim = dq.MarketSimulator(symbols)
    
    # Enable market events for realistic price movements
    sim.enable_market_events(True)
    sim.set_event_probability(0.015)  # 1.5% chance of events
    
    # Create C++ market maker
    config = dq.MarketMakerConfig()
    config.trader_id = "market_maker"
    config.symbols = symbols
    config.base_price = 100.0
    config.spread_pct = 0.001  # 0.1% spread
    config.order_size = 30.0
    config.max_orders_per_side = 3
    config.update_interval_ms = 150
    config.adaptive_spread = True
    
    market_maker = dq.MarketMaker(sim, config)
    market_maker.start()
    
    # Let market maker establish liquidity
    time.sleep(1)
    
    return sim, market_maker

def create_traders(sim):
    """Create and register traders with different strategies"""
    print("Creating traders with different strategies...")
    
    traders = []
    
    # Mean Reversion Trader
    env1 = DeepQuoteEnv(symbols=["AAPL"], trader_id="meanrev", strategy_type="RL")
    trader1 = MeanReversionAgent(env1, entry_threshold=1.0, exit_threshold=0.2)
    traders.append(("MeanReversion", trader1, env1))
    sim.add_rl_trader(env1.trader)
    
    # Momentum Trader
    env2 = DeepQuoteEnv(symbols=["AAPL"], trader_id="momentum", strategy_type="RL")
    trader2 = MomentumAgent(env2, momentum_threshold=0.001, position_size=0.5)
    traders.append(("Momentum", trader2, env2))
    sim.add_rl_trader(env2.trader)
    
    # Market Making Trader
    env3 = DeepQuoteEnv(symbols=["AAPL"], trader_id="mm_agent", strategy_type="RL")
    trader3 = MarketMakingAgent(env3, order_size=15.0)
    traders.append(("MarketMaking", trader3, env3))
    sim.add_rl_trader(env3.trader)
    
    return traders

def run_trading_simulation(sim, market_maker, traders, duration_seconds=30):
    """Run the main trading simulation"""
    print(f"\nStarting trading simulation for {duration_seconds} seconds...")
    print("Watch for actual trades being executed!")
    print("-" * 80)
    
    # Track performance
    total_trades = 0
    trader_trades = {name: 0 for name, _, _ in traders}
    last_trade_count = 0
    start_time = time.time()
    
    # Initial market state
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
        # Update market events
        sim.update_market_events(0.2)  # 200ms time step
        
        # Let each trader take an action
        for trader_name, trader, env in traders:
            # Get observation and action
            obs = env._get_obs()
            action = trader.get_action(obs)
            
            # Process action
            obs, reward, done, info = env.step(action)
            
            # Check for trades
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
        
        # Print progress every 10 seconds
        elapsed = time.time() - start_time
        if step % 50 == 0 and step > 0:
            print(f"\n--- {elapsed:.1f}s elapsed ---")
            
            # Show current market state
            for symbol in sim.get_symbols():
                best_bid = sim.get_best_bid(symbol)
                best_ask = sim.get_best_ask(symbol)
                spread = best_ask - best_bid
                print(f"  {symbol}: Bid=${best_bid:.2f}, Ask=${best_ask:.2f}, Spread=${spread:.2f}")
            
            # Show trader positions
            print(f"Trader Positions:")
            for trader_name, _, env in traders:
                trader = env.trader
                cash = trader.get_cash()
                inventory = trader.get_inventory("AAPL")
                pnl = trader.get_realized_pnl()
                print(f"  {trader_name:15s}: Cash=${cash:8.2f}, Inv={inventory:6.1f}, PnL=${pnl:6.2f}")
        
        step += 1
        time.sleep(0.2)  # 5Hz simulation
    
    return total_trades, trader_trades

def print_final_results(sim, market_maker, traders, total_trades, trader_trades):
    """Print comprehensive final results"""
    print(f"\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    # Trading activity summary
    print(f"Trading Activity:")
    print(f"  Total Trades Executed: {total_trades}")
    print(f"  Trades by Trader:")
    for trader_name, trade_count in trader_trades.items():
        print(f"    {trader_name}: {trade_count} trades")
    
    # Trader performance
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
    
    # Market maker performance
    mm_stats = market_maker.get_stats()
    print(f"\nMarket Maker Performance:")
    print(f"  Cash: ${mm_stats.cash:.2f}")
    print(f"  PnL: ${mm_stats.total_pnl:.2f}")
    print(f"  Active Orders: {mm_stats.active_orders}")
    
    # Market statistics
    print(f"\nMarket Statistics:")
    print(f"  Total Orders: {sim.get_total_order_count()}")
    print(f"  Total Trades: {sim.get_total_trade_count()}")
    print(f"  Active Events: {sim.get_active_event_count()}")
    
    # Final market state
    print(f"\nFinal Market State:")
    for symbol in sim.get_symbols():
        best_bid = sim.get_best_bid(symbol)
        best_ask = sim.get_best_ask(symbol)
        spread = best_ask - best_bid
        mid_price = sim.get_mid_price(symbol)
        print(f"  {symbol}: Bid=${best_bid:.2f}, Ask=${best_ask:.2f}, Mid=${mid_price:.2f}, Spread=${spread:.2f}")
    
    # Active market events
    active_events = sim.get_active_events()
    if active_events:
        print(f"\nActive Market Events:")
        for event in active_events:
            print(f"  {event.type}: {event.description} (Magnitude: {event.magnitude:.3f})")

def demo_manual_trading():
    """Demo manual trading to show direct order placement"""
    print("\n" + "=" * 80)
    print("MANUAL TRADING DEMO")
    print("=" * 80)
    
    # Setup market
    sim, market_maker = setup_market()
    
    # Register a manual trader
    sim.register_trader("manual_trader", 50000.0)
    trader = sim.get_trader("manual_trader")
    
    print("Manual trading with direct order placement...")
    
    # Place some test orders
    orders = []
    
    # Buy order
    buy_order = dq.Order()
    buy_order.id = 1
    buy_order.side = dq.Side.BUY
    buy_order.type = dq.OrderType.LIMIT
    buy_order.price = sim.get_best_ask("AAPL") + 0.01
    buy_order.quantity = 20.0
    buy_order.symbol = "AAPL"
    buy_order.trader_id = "manual_trader"
    buy_order.strategy_id = "manual"
    orders.append(buy_order)
    
    # Sell order
    sell_order = dq.Order()
    sell_order.id = 2
    sell_order.side = dq.Side.SELL
    sell_order.type = dq.OrderType.LIMIT
    sell_order.price = sim.get_best_bid("AAPL") - 0.01
    sell_order.quantity = 10.0
    sell_order.symbol = "AAPL"
    sell_order.trader_id = "manual_trader"
    sell_order.strategy_id = "manual"
    orders.append(sell_order)
    
    # Execute orders
    total_trades = 0
    for i, order in enumerate(orders):
        print(f"\nPlacing order {i+1}: {order.side} {order.quantity} @ ${order.price:.2f}")
        
        trades = sim.process_order(order)
        print(f"Order resulted in {len(trades)} trades")
        
        if trades:
            for trade in trades:
                total_trades += 1
                print(f"  Trade: {trade.quantity} @ ${trade.price:.2f}")
        
        print(f"  Trader cash: ${trader.get_cash():.2f}")
        print(f"  Trader inventory: {trader.get_inventory('AAPL'):.1f}")
        print(f"  Trader PnL: ${trader.get_realized_pnl():.2f}")
        
        time.sleep(0.5)
    
    print(f"\nManual Trading Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Final Cash: ${trader.get_cash():.2f}")
    print(f"  Final Inventory: {trader.get_inventory('AAPL'):.1f}")
    print(f"  Final PnL: ${trader.get_realized_pnl():.2f}")
    
    market_maker.stop()

def main():
    """Main demo function"""
    print("DeepQuote Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases:")
    print("✅ C++ Market Maker providing liquidity")
    print("✅ RL Agents with different strategies (Mean Reversion, Momentum, Market Making)")
    print("✅ Real market events driving price movements")
    print("✅ Actual trade execution and position tracking")
    print("✅ Performance monitoring and analysis")
    print("✅ Manual trading demonstration")
    print("=" * 80)
    
    try:
        # Part 1: Main trading simulation
        print("\nPART 1: MAIN TRADING SIMULATION")
        print("-" * 40)
        
        # Setup market and traders
        sim, market_maker = setup_market()
        traders = create_traders(sim)
        
        # Run simulation
        total_trades, trader_trades = run_trading_simulation(sim, market_maker, traders, duration_seconds=30)
        
        # Print results
        print_final_results(sim, market_maker, traders, total_trades, trader_trades)
        
        # Stop market maker
        market_maker.stop()
        
        # Part 2: Manual trading demo
        print("\nPART 2: MANUAL TRADING DEMO")
        print("-" * 40)
        demo_manual_trading()
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("DeepQuote is working with actual trade execution!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 