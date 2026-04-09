import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from environment import Environment
from impulse_response import ImpulseResponseAnalyzer

# Setup
results_path = "results/20251212_013901/results.pkl"
target_session_id = 10  # Best stable session
output_dir = "results/manual_test_session10"
os.makedirs(output_dir, exist_ok=True)

# Load results
with open(results_path, "rb") as f:
    results = pickle.load(f)

# Find target result
target_result = next((r for r in results if r['session_id'] == target_session_id), None)

if target_result and target_result['converged']:
    print(f"Running IRF for Session {target_session_id}")
    print(f"Avg Price: {target_result['avg_price']:.4f}")
    
    config = Config()
    env = Environment(config)
    q_matrices = target_result['q_matrices']
    final_state = target_result['final_state']
    
    analyzer = ImpulseResponseAnalyzer(config, env, target_result)
    
    # Run simulation manually to inspect data
    analyzer.run_simulation()
    history = analyzer.history
    
    print("\nTime | P1     | P2     | Pi1    | Pi2")
    print("-" * 40)
    
    ts = history['t']
    p1s = history['p1']
    p2s = history['p2']
    pi1s = history['pi1']
    pi2s = history['pi2']
    
    for i, t in enumerate(ts):
        if -2 <= t <= 5: # Show around t=0
            print(f"{t:4d} | {p1s[i]:.4f} | {p2s[i]:.4f} | {pi1s[i]:.4f} | {pi2s[i]:.4f}")
            
    # Plot
    analyzer.plot_impulse_response(
        os.path.join(output_dir, "irf_price_s9.png"),
        os.path.join(output_dir, "irf_profit_s9.png")
    )
    print(f"\nPlots saved to {output_dir}")

else:
    print(f"Session {target_session_id} not found or not converged.")

