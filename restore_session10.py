import os
import pickle
from config import Config
from environment import Environment
from impulse_response import ImpulseResponseAnalyzer

def restore_session10():
    results_path = "results/20251218_172404/results.pkl"
    output_dir = "results/20251218_172404"
    target_session_id = 10

    print(f"Restoring IRF images for Session {target_session_id}...")
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    target_result = next((r for r in results if r['session_id'] == target_session_id), None)
    
    if target_result:
        config = Config()
        config.n_sessions = 100
        env = Environment(config)
        
        analyzer = ImpulseResponseAnalyzer(config, env, target_result)
        analyzer.run_simulation()
        
        analyzer.plot_impulse_response(
            os.path.join(output_dir, "irf_price.png"),
            os.path.join(output_dir, "irf_profit.png")
        )
        print("Done. Images restored.")
    else:
        print("Session 10 not found.")

if __name__ == "__main__":
    restore_session10()

