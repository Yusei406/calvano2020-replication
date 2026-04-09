import pickle
import numpy as np
import os
from config import Config

# Load results
results_path = "results/20251212_013901/results.pkl"
with open(results_path, "rb") as f:
    results = pickle.load(f)

# Config to get pi_nash/monopoly roughly
config = Config()
# Using values from summary.json manually for accurate delta calculation
pi_nash = 0.222928
pi_monopoly = 0.337490

print("Finding stable and high-collusion sessions...")
print(f"{'ID':<4} | {'Delta':<8} | {'Price Std':<10} | {'Avg Price':<10}")
print("-" * 45)

candidates = []
for r in results:
    if r['converged']:
        avg_pi = r['avg_profit']
        avg_p = r['avg_price']
        
        # Calculate Delta
        delta = (avg_pi - pi_nash) / (pi_monopoly - pi_nash)
        
        # Calculate Price Stability (Standard Deviation in last 100 steps)
        # Note: We don't have the full history here, only avg_price/profit.
        # However, we can use the Q-matrices to check if the strategy is a fixed point or cycle.
        # Or simpler: The 'avg_price' is just a mean. 
        # Wait, earlier code only saved 'avg_price'.
        # Actually, let's look at the 'q_matrices' to simulate the last few steps and check stability.
        
        # We will simulate 10 steps from final_state using the learned Q-matrices
        # to check if it's a fixed point or a cycle.
        
        q_data = r['q_matrices']
        s = r['final_state']
        m_grid = config.m_grid
        
        # Simulate a bit to check cycle
        prices = []
        
        # Check if q is list and convert to numpy if needed.
        if isinstance(q_data, list):
            q = np.array(q_data)
        else:
            q = q_data

        curr_s = int(s) # Ensure s is int
        for _ in range(20):
            # Greedy actions
            a1 = np.argmax(q[0, curr_s, :])
            a2 = np.argmax(q[1, curr_s, :])
            
            # Get price (approx from index)
            # We don't have env here, but we just want to see if a1/a2 change.
            prices.append((a1, a2))
            
            curr_s = a1 * m_grid + a2
            
        # Check if all actions are the same (Fixed Point)
        p_std = np.std(prices) # If 0, it's a fixed point.
        
        is_fixed_point = (p_std == 0)
        status = "Fixed" if is_fixed_point else "Cycle"
        
        if delta > 0.7: # Only consider high collusion
            print(f"{r['session_id']:<4} | {delta:.4f}   | {status:<10} | {avg_p:.4f}")
            if is_fixed_point:
                candidates.append((r['session_id'], delta))

print("-" * 45)
candidates.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 STABLE (Fixed Point) Sessions by Delta:")
for cid, d in candidates[:5]:
    print(f"Session {cid}: Delta = {d:.4f}")

if not candidates:
    print("No fixed point sessions found with Delta > 0.7. We might need to pick the one with smallest cycle.")

