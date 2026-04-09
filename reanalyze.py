import os
import pickle
import numpy as np
from config import Config
from environment import Environment
from evaluation import Evaluator
from impulse_response import ImpulseResponseAnalyzer

def main():
    # Path to the results directory
    timestamp = "20251218_172404"
    output_dir = os.path.join("results", timestamp)
    pkl_path = os.path.join(output_dir, "results.pkl")
    
    print(f"Loading results from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
        
    config = Config()
    config.n_sessions = 100 # Ensure config matches loaded results
    env = Environment(config)
    
    print("\nRe-evaluating results...")
    evaluator = Evaluator(config, results, env)
    evaluator.print_summary()
    
    # 4. インパルス応答分析 (Impulse Response Analysis)
    print("\nRunning Impulse Response Analysis on the best session...")
    best_session_id = evaluator.get_best_session_id()
    
    if best_session_id is not None:
        # 安全な検索方法
        target_result = next((r for r in results if r['session_id'] == best_session_id), None)
        
        if target_result and target_result['converged']:
            print(f"Target Session ID: {target_result['session_id']}")
            print(f"  Avg Price: {target_result['avg_price']:.4f}")
            print(f"  Avg Profit: {target_result['avg_profit']:.4f}")
            
            analyzer = ImpulseResponseAnalyzer(config, env, target_result)
            try:
                analyzer.run_simulation()
                
                irf_price_path = os.path.join(output_dir, "irf_price.png")
                irf_profit_path = os.path.join(output_dir, "irf_profit.png")
                
                analyzer.plot_impulse_response(irf_price_path, irf_profit_path)
                print("Done.")
            except Exception as e:
                print(f"Error during Impulse Response Analysis: {e}")
        else:
            print(f"Error: Session ID {best_session_id} not found in results.")
    else:
        print("Skipping Impulse Response Analysis (No converged sessions found).")
        
    print("\nRe-analysis Finished Successfully.")

if __name__ == "__main__":
    main()

