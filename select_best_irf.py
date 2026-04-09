import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from environment import Environment
from impulse_response import ImpulseResponseAnalyzer
from evaluation import Evaluator

def check_candidates():
    results_path = "results/20251218_172404/results.pkl"
    output_dir = "results/irf_candidates"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_path}...")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    config = Config()
    config.n_sessions = 100
    env = Environment(config)
    
    # 1. Find all fixed point sessions
    pi_nash = env.pi_nash[0]
    pi_monopoly = env.pi_monopoly[0]
    m_grid = config.m_grid
    
    fixed_point_candidates = []
    
    print("\nScanning for Fixed Point sessions...")
    for r in results:
        if r['converged'] and r['avg_profit'] is not None:
            # Fixed Point Check
            q = r['q_matrices']
            # Ensure numpy
            if isinstance(q, list): q = np.array(q)
            
            s = r['final_state']
            curr_s = int(s)
            prices_idx = []
            
            # Check 10 steps
            for _ in range(10):
                a1 = np.argmax(q[0, curr_s, :])
                a2 = np.argmax(q[1, curr_s, :])
                prices_idx.append((a1, a2))
                curr_s = a1 * m_grid + a2
            
            if np.std(prices_idx) == 0:
                delta = (r['avg_profit'] - pi_nash) / (pi_monopoly - pi_nash)
                fixed_point_candidates.append({
                    'id': r['session_id'],
                    'delta': delta,
                    'result': r
                })

    # Sort by Delta descending
    fixed_point_candidates.sort(key=lambda x: x['delta'], reverse=True)
    print(f"Found {len(fixed_point_candidates)} fixed point sessions.")
    
    # 2. Check IRF for top candidates (limit to top 10)
    best_candidate = None
    best_score = -np.inf # Score based on symmetry and clarity
    
    print("\nChecking IRF behavior for top candidates...")
    print(f"{'ID':<4} | {'Delta':<6} | {'Recovery Symmetry'}")
    print("-" * 40)
    
    for cand in fixed_point_candidates[:10]:
        sid = cand['id']
        analyzer = ImpulseResponseAnalyzer(config, env, cand['result'])
        analyzer.run_simulation()
        history = analyzer.history
        
        p1s = np.array(history['p1'])
        p2s = np.array(history['p2'])
        ts = np.array(history['t'])
        
        # Check recovery phase (t=1 to t=10)
        # We want small difference between p1 and p2 during recovery
        # Specifically around t=3, 4, 5
        
        # Calculate symmetry score (negative mean squared difference during recovery)
        recovery_indices = np.where((ts >= 1) & (ts <= 5))[0]
        diffs = p1s[recovery_indices] - p2s[recovery_indices]
        symmetry_score = -np.mean(diffs**2)
        
        # Also check if it actually dropped (Punishment existence)
        # Minimum price during t=1 to t=3 should be low
        punish_indices = np.where((ts >= 1) & (ts <= 3))[0]
        min_p = np.min(np.concatenate([p1s[punish_indices], p2s[punish_indices]]))
        has_punishment = min_p < config.p_nash + 0.1 # Should be close to Nash or lower
        
        status = f"Score: {symmetry_score:.4f}"
        if not has_punishment:
            status += " (Weak Punishment)"
            symmetry_score -= 100 # Penalize weak punishment
            
        print(f"{sid:<4} | {cand['delta']:.4f} | {status}")
        
        if symmetry_score > best_score:
            best_score = symmetry_score
            best_candidate = cand
            best_candidate['history'] = history # Save history for plotting

    # 3. Report Best Candidate
    if best_candidate:
        print(f"\nBest Candidate Selected: Session {best_candidate['id']}")
        print(f"Delta: {best_candidate['delta']:.4f}")
        
        # Print IRF Table
        h = best_candidate['history']
        print("\nTime | P1     | P2     | Pi1    | Pi2")
        print("-" * 40)
        ts = h['t']
        for i, t in enumerate(ts):
            if -2 <= t <= 5:
                print(f"{t:4d} | {h['p1'][i]:.4f} | {h['p2'][i]:.4f} | {h['pi1'][i]:.4f} | {h['pi2'][i]:.4f}")

        # Save Plots
        target_result = best_candidate['result']
        analyzer = ImpulseResponseAnalyzer(config, env, target_result)
        analyzer.run_simulation() # Re-run to be sure
        
        # Save to main results dir (overwrite)
        main_output_dir = "results/20251218_172404"
        analyzer.plot_impulse_response(
            os.path.join(main_output_dir, "irf_price.png"),
            os.path.join(main_output_dir, "irf_profit.png")
        )
        print(f"\nOverwritten irf_price.png and irf_profit.png in {main_output_dir}")
        
    else:
        print("No suitable candidate found.")

if __name__ == "__main__":
    check_candidates()

