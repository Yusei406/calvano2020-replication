import pickle
import numpy as np
import os
from config import Config

# Load results
results_path = "results/20251212_013901/results.pkl"
with open(results_path, "rb") as f:
    results = pickle.load(f)

# Config for calculating delta
config = Config()
# Re-calculate environment constants roughly for delta
# Note: Exact pi_nash/monopoly might be needed from environment but we can approximate or use config values if stored
# However, simpler to just look at profits.
# Or better, load environment to get pi_nash/pi_monopoly exactly.
from environment import Environment
env = Environment(config)
pi_nash = env.pi_nash[0]
pi_monopoly = env.pi_monopoly[0]

print(f"Nash Profit: {pi_nash:.4f}")
print(f"Monopoly Profit: {pi_monopoly:.4f}")

print("\n--- Session Analysis ---")
candidates = []
for r in results:
    if r['converged']:
        avg_pi = r['avg_profit']
        delta = (avg_pi - pi_nash) / (pi_monopoly - pi_nash)
        print(f"Session {r['session_id']}: Delta = {delta:.4f}, Avg Price = {r['avg_price']:.4f}, Avg Profit = {avg_pi:.4f}")
        candidates.append((r['session_id'], delta))

# Sort by delta descending
candidates.sort(key=lambda x: x[1], reverse=True)
print("\nTop 5 Sessions by Delta:")
for cid, d in candidates[:5]:
    print(f"Session {cid}: Delta = {d:.4f}")

print("\nBottom 5 Sessions by Delta:")
for cid, d in candidates[-5:]:
    print(f"Session {cid}: Delta = {d:.4f}")

