import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from environment import Environment
from impulse_response import ImpulseResponseAnalyzer
from evaluation import Evaluator

# Setup
results_path = "results/20251218_172404/results.pkl"
output_dir = "results/check_irf_session70"
os.makedirs(output_dir, exist_ok=True)

# Load results
print(f"Loading results from {results_path}...")
with open(results_path, "rb") as f:
    results = pickle.load(f)

# Config
config = Config()
config.n_sessions = 100
env = Environment(config)

# Re-run selection logic to confirm Session 70 is indeed the target
evaluator = Evaluator(config, results, env)
target_session_id = evaluator.get_best_session_id()
print(f"Target Session ID selected by logic: {target_session_id}")

target_result = next((r for r in results if r['session_id'] == target_session_id), None)

if target_result:
    print(f"Avg Price: {target_result['avg_price']:.4f}")
    
    analyzer = ImpulseResponseAnalyzer(config, env, target_result)
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
            
    # Check if P1 and P2 are exactly same at t=-1 and t=5
    is_stable_pre = (p1s[0] == p1s[4]) and (p2s[0] == p2s[4]) # t=-5 vs t=-1
    is_stable_post = (p1s[-1] == p1s[-2]) # t=20 vs t=19
    
    print(f"\nStability Check:")
    print(f"  Pre-deviation stability: {is_stable_pre}")
    print(f"  Post-deviation stability: {is_stable_post}")
    
    # Plot for manual visual check
    analyzer.plot_impulse_response(
        os.path.join(output_dir, "check_irf_price.png"),
        os.path.join(output_dir, "check_irf_profit.png")
    )
    print(f"\nCheck plots saved to {output_dir}")

else:
    print("Target session not found.")

