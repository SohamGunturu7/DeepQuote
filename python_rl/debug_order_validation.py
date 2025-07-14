#!/usr/bin/env python3
"""
Debug script to test order validation step by step.
"""

import numpy as np
import deepquote_simulator as dq

# Test order validation step by step
def test_order_validation():
    print("Testing order validation...")
    
    sim = dq.MarketSimulator(["AAPL"])
    
    trader = dq.RLTrader("test_agent", "DQN", 100000.0)
    sim.add_rl_trader(trader)
    
    print("\nTest 1: Minimal valid order")
    order1 = dq.Order()
    order1.id = 1
    order1.side = dq.Side.BUY
    order1.type = dq.OrderType.LIMIT
    order1.price = 100.0
    order1.quantity = 10.0
    order1.symbol = "AAPL"
    order1.trader_id = trader.get_id()
    
    print(f"Order 1 - ID: {order1.id}, Price: {order1.price}, Qty: {order1.quantity}")
    print(f"Symbol: '{order1.symbol}', Trader: '{order1.trader_id}'")
    print(f"Is valid: {order1.is_valid()}")
    
    try:
        trades = sim.process_order(order1)
        print(f"✅ Order 1 processed successfully! Trades: {len(trades)}")
    except Exception as e:
        print(f"❌ Order 1 failed: {e}")
    
    print("\nTest 2: Order with zero quantity")
    order2 = dq.Order()
    order2.id = 2
    order2.side = dq.Side.BUY
    order2.type = dq.OrderType.LIMIT
    order2.price = 100.0
    order2.quantity = 0.0
    order2.symbol = "AAPL"
    order2.trader_id = trader.get_id()
    
    print(f"Order 2 - ID: {order2.id}, Price: {order2.price}, Qty: {order2.quantity}")
    print(f"Is valid: {order2.is_valid()}")
    
    try:
        trades = sim.process_order(order2)
        print(f"✅ Order 2 processed successfully! Trades: {len(trades)}")
    except Exception as e:
        print(f"❌ Order 2 failed: {e}")
    
    print("\nTest 3: Order with zero price")
    order3 = dq.Order()
    order3.id = 3
    order3.side = dq.Side.BUY
    order3.type = dq.OrderType.LIMIT
    order3.price = 0.0
    order3.quantity = 10.0
    order3.symbol = "AAPL"
    order3.trader_id = trader.get_id()
    
    print(f"Order 3 - ID: {order3.id}, Price: {order3.price}, Qty: {order3.quantity}")
    print(f"Is valid: {order3.is_valid()}")
    
    try:
        trades = sim.process_order(order3)
        print(f"✅ Order 3 processed successfully! Trades: {len(trades)}")
    except Exception as e:
        print(f"❌ Order 3 failed: {e}")
    
    print("\nTest 4: Order with empty symbol")
    order4 = dq.Order()
    order4.id = 4
    order4.side = dq.Side.BUY
    order4.type = dq.OrderType.LIMIT
    order4.price = 100.0
    order4.quantity = 10.0
    order4.symbol = ""
    order4.trader_id = trader.get_id()
    
    print(f"Order 4 - ID: {order4.id}, Price: {order4.price}, Qty: {order4.quantity}")
    print(f"Symbol: '{order4.symbol}', Trader: '{order4.trader_id}'")
    print(f"Is valid: {order4.is_valid()}")
    
    try:
        trades = sim.process_order(order4)
        print(f"✅ Order 4 processed successfully! Trades: {len(trades)}")
    except Exception as e:
        print(f"❌ Order 4 failed: {e}")

# Test what happens when we create an order from agent action
def test_agent_action():
    print("\n" + "="*50)
    print("Testing agent action to order conversion")
    print("="*50)
    
    from deepquote_env import DeepQuoteEnv
    env = DeepQuoteEnv(symbols=["AAPL"], trader_id="test_agent", strategy_type="DQN")
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    agent_action = np.array([0, 0, 5.0, 100.0])
    print(f"Agent action: {agent_action}")
    
    try:
        obs, reward, done, info = env.step(agent_action)
        print(f"✅ Step successful! Reward: {reward}, Done: {done}")
        print(f"Info: {info}")
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_order_validation()
    test_agent_action() 