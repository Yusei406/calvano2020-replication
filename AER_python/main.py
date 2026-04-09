import os
import time
import datetime
import json
import pickle  # 追加
import numpy as np

from AER_python.config import Config
from AER_python.environment import Environment
from AER_python.agent import Agent
from AER_python.train import Trainer
from AER_python.evaluation import Evaluator
from AER_python.impulse_response import ImpulseResponseAnalyzer

def convert_for_json(o):
    """
    Numpy型などをJSON保存可能な型に変換するヘルパー関数
    変換できない型が来た場合はそのまま返し、json.dumpの標準エラー処理に任せる
    """
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

def save_config(config, output_dir):
    """実験設定をJSONとして保存"""
    params = {}
    for key in dir(config):
        if not key.startswith("_"):
            val = getattr(config, key)
            # メソッドや呼び出し可能オブジェクトは除外
            if not callable(val):
                if isinstance(val, (int, float, str, bool, list, tuple, np.integer, np.floating, np.ndarray)):
                    params[key] = convert_for_json(val)
    
    try:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(params, f, indent=4)
    except Exception as e:
        print(f"Error saving config.json: {e}")

def save_summary(summary_data, output_dir):
    """統計サマリーをJSONとして保存"""
    try:
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary_data, f, indent=4, default=convert_for_json)
        print(f"Summary saved to {os.path.join(output_dir, 'summary.json')}")
    except Exception as e:
        print(f"Error saving summary.json: {e}")

def main(test_config=None):
    print("=" * 60)
    print("AI Collusion Experiment: Replication of Calvano et al. (2020)")
    print("=" * 60)
    
    # 0. 結果保存用ディレクトリの作成
    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join("results", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output Directory: {output_dir}")
    except Exception as e:
        print(f"Critical Error creating output directory: {e}")
        return
    
    # 1. 初期化
    try:
        # テスト用のConfigが渡されたらそれを使う
        if test_config is not None:
            config = test_config
        else:
            config = Config()
        
        # 設定の保存
        save_config(config, output_dir)
        
        env = Environment(config)
        
        # エージェントの作成
        agent1 = Agent(config, env, 0)
        agent2 = Agent(config, env, 1)
        
        # トレーナーの作成
        trainer = Trainer(config, env, agent1, agent2)
    except Exception as e:
        print(f"Critical Error during initialization: {e}")
        return
    
    print(f"Configuration:")
    print(f"  Sessions: {config.n_sessions}")
    print(f"  Max Steps: {config.max_steps:,}")
    print(f"  Convergence Window: {config.convergence_window:,}")
    print(f"  Beta Decay: {config.beta_decay}")
    print("-" * 60)
    
    # 2. 実験ループ
    results = []
    total_start_time = time.time()
    
    print("Starting experiments...")
    
    for s in range(config.n_sessions):
        print(f"Session {s+1}/{config.n_sessions} running...", end="", flush=True)
        
        try:
            # セッション実行
            res = trainer.train_session(session_id=s)
            results.append(res)
            
            status = "Converged" if res['converged'] else "Failed"
            print(f" Done. [{status}, Steps: {res['steps']:,}, Time: {res['elapsed_time']:.4f}s]")
        except Exception as e:
            print(f" ERROR: {e}")
            # エラーセッションもダミーとして記録（評価時にスキップされる）
            results.append({
                'session_id': s, 
                'converged': False, 
                'steps': 0, 
                'elapsed_time': 0,
                'avg_price': None,
                'avg_profit': None,
                'q_matrices': None,
                'final_state': None
            })
        
    total_elapsed = time.time() - total_start_time
    print("-" * 60)
    print(f"All sessions completed in {total_elapsed:.2f} seconds.")
    
    # 3. 結果の集計・評価
    print("\nEvaluating results...")
    evaluator = Evaluator(config, results, env)
    
    # コンソール表示
    evaluator.print_summary()
    
    # 統計データの保存 (Evaluatorから直接取得)
    summary_data = evaluator.get_statistics()
    # 実行時間も追加
    summary_data["total_elapsed_time"] = total_elapsed
    
    save_summary(summary_data, output_dir)
    
    # 全結果データの保存 (Pickle)
    try:
        pkl_path = os.path.join(output_dir, "results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Full results saved to {pkl_path}")
    except Exception as e:
        print(f"Error saving results.pkl: {e}")
    
    # ヒストグラムの保存
    hist_path = os.path.join(output_dir, "histogram_price.png")
    evaluator.plot_price_histogram(hist_path)
    
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
        
    print("\nExperiment Finished Successfully.")

if __name__ == "__main__":
    main()
