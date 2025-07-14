#!/usr/bin/env python3
"""
Debug script to see what actions agents are generating.
"""

import numpy as np
from agents import MarketMakingAgent, MeanReversionAgent, MomentumAgent
from deepquote_env import DeepQuoteEnv

# Debug what actions agents are generating
def debug_agent_actions():
    print("Debugging agent actions...")
    
    env = DeepQuoteEnv(symbols=["AAPL"], trader_id="debug_agent", strategy_type="DQN")
    obs = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    agents = [
        ("MarketMaking", MarketMakingAgent(env)),
        ("MeanReversion", MeanReversionAgent(env)),
        ("Momentum", MomentumAgent(env))
    ]
    
    for agent_name, agent in agents:
        print(f"\n{'='*50}")
        print(f"Testing {agent_name} Agent")
        print(f"{'='*50}")
        
        try:
            action = agent.get_action(obs)
            print(f"Agent action: {action}")
            print(f"Action shape: {action.shape}")
            print(f"Action type: {type(action)}")
            
            if len(action) >= 4:
                action_type = int(action[0])
                symbol_idx = int(action[1])
                quantity = float(action[2])
                price = float(action[3])
                
                print(f"Parsed action:")
                print(f"  - Action type: {action_type}")
                print(f"  - Symbol idx: {symbol_idx}")
                print(f"  - Quantity: {quantity}")
                print(f"  - Price: {price}")
                
                issues = []
                if quantity <= 0:
                    issues.append(f"Invalid quantity: {quantity}")
                if price <= 0:
                    issues.append(f"Invalid price: {price}")
                if action_type not in [0, 1, 5]:
                    issues.append(f"Unknown action type: {action_type}")
                
                if issues:
                    print(f"❌ Issues found: {issues}")
                else:
                    print("✅ Action looks valid")
                    
                    try:
                        obs, reward, done, info = env.step(action)
                        print(f"✅ Step successful! Reward: {reward}, Done: {done}")
                        print(f"Info: {info}")
                    except Exception as e:
                        print(f"❌ Step failed: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"❌ Action too short: {len(action)} values")
                
        except Exception as e:
            print(f"❌ Agent failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_agent_actions() 