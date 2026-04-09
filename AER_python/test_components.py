import numpy as np
import sys
import os
import time
import shutil
import tracemalloc

# パスを通す
sys.path.append(os.path.join(os.getcwd(), 'AER_python'))

from config import Config
from environment import Environment
from agent import Agent
from train import Trainer
from evaluation import Evaluator
from impulse_response import ImpulseResponseAnalyzer
from main import main 

def test_config():
    print("--- Config Test ---")
    c = Config()
    print(f"Alpha (Learning Rate): {c.alpha}")
    print(f"Beta Decay: {c.beta_decay}")
    assert c.n_agents == 2
    print("Config OK.\n")

def test_environment():
    print("--- Environment Test ---")
    c = Config()
    env = Environment(c)
    
    # 1. 価格グリッドの確認
    print(f"Price Grid Size: {len(env.prices)}")
    print(f"Min Price: {env.prices[0]:.4f}")
    print(f"Max Price: {env.prices[-1]:.4f}")
    print(f"Nash Price (Theoretical): {c.p_nash:.4f}")
    print(f"Monopoly Price (Theoretical): {c.p_monopoly:.4f}")
    
    # 2. 利潤行列の確認
    print(f"Profit Matrix Shape: {env.profits.shape}")
    assert env.profits.shape == (c.m_grid**2, 2)
    
    # 3. 利潤の妥当性確認
    idx_nash = np.abs(env.prices - c.p_nash).argmin()
    idx_mon = np.abs(env.prices - c.p_monopoly).argmin()
    
    s_nash = idx_nash * c.m_grid + idx_nash
    s_mon = idx_mon * c.m_grid + idx_mon
    
    pi_nash = env.profits[s_nash, 0]
    pi_mon = env.profits[s_mon, 0]
    
    print(f"Profit at Nash-like state (idx={idx_nash}): {pi_nash:.4f}")
    print(f"Profit at Monopoly-like state (idx={idx_mon}): {pi_mon:.4f}")
    
    if pi_mon > pi_nash:
        print("Check OK: Monopoly profit > Nash profit")
    else:
        print("WARNING: Monopoly profit is NOT greater than Nash profit. Check parameters.")
        
    print("Environment OK.\n")
    return env

def test_agent(env):
    print("--- Agent Test ---")
    c = Config()
    agent = Agent(c, env, agent_id=0)
    
    # 1. Q行列初期化
    print(f"Q Matrix Shape: {agent.Q.shape}")
    assert agent.Q.shape == (c.m_grid**2, c.m_grid)
    print(f"Initial Q Mean: {np.mean(agent.Q):.4f}")
    
    # 2. 行動選択
    state = 0
    action = agent.get_action(state)
    print(f"Selected Action: {action}")
    assert 0 <= action < c.m_grid
    
    # 3. Q値更新
    old_q = agent.Q[state, action]
    reward = 1.0
    next_state = 1
    agent.update_Q(state, action, reward, next_state)
    new_q = agent.Q[state, action]
    
    print(f"Old Q: {old_q:.4f}, New Q: {new_q:.4f}")
    assert new_q != old_q, "Q value should change after update"
    
    print("Agent OK.\n")

def test_trainer(env):
    print("--- Trainer Test (Short) ---")
    class TestConfig(Config):
        max_steps = 1000
        convergence_window = 10
        l_buffer = 10
    
    c = TestConfig()
    agent1 = Agent(c, env, 0)
    agent2 = Agent(c, env, 1)
    
    trainer = Trainer(c, env, agent1, agent2)
    
    print(f"Starting test session (max_steps={c.max_steps})...")
    result = trainer.train_session(session_id=999)
    
    print(f"Session Finished.")
    print(f"Converged: {result['converged']}")
    print(f"Steps: {result['steps']}")
    print(f"Elapsed Time: {result['elapsed_time']:.4f} sec")
    
    if result['converged']:
        print(f"Avg Price: {result['avg_price']:.4f}")
        print(f"Avg Profit: {result['avg_profit']:.4f}")
        assert result['q_matrices'] is not None
        assert result['final_state'] is not None
    else:
        print("Did not converge.")
        
    print("Trainer OK.\n")

def test_trainer_extended(env):
    print("--- Trainer Extended Test (Robustness & Benchmark) ---")
    
    # 1. 独立性テスト
    print("[1] Session Independence Test")
    c = Config()
    c.max_steps = 2000 
    c.convergence_window = 500
    
    agent1 = Agent(c, env, 0)
    agent2 = Agent(c, env, 1)
    trainer = Trainer(c, env, agent1, agent2)
    
    res1 = trainer.train_session(session_id=101)
    res2 = trainer.train_session(session_id=102)
    
    print(f"Session 101 Result (Avg Price): {res1['avg_price']}")
    print(f"Session 102 Result (Avg Price): {res2['avg_price']}")
    
    if res1['avg_price'] != res2['avg_price']:
        print("OK: Different seeds produced different results.")
    else:
        print("WARNING: Different seeds produced SAME results (or luck).")

    # 2. 未収束テスト
    print("\n[2] Non-Convergence Test")
    c.convergence_window = 100000
    c.max_steps = 100
    
    res_fail = trainer.train_session(session_id=103)
    print(f"Converged: {res_fail['converged']}")
    assert res_fail['converged'] is False
    assert res_fail['avg_price'] is None
    print("OK: Correctly handled non-convergence.")

    # 3. 速度ベンチマーク (Numba warm-up済み)
    print("\n[3] Speed Benchmark")
    benchmark_steps = 50000 
    c.max_steps = benchmark_steps
    c.convergence_window = benchmark_steps + 1
    
    print(f"Running {benchmark_steps} steps...")
    start = time.time()
    res_bench = trainer.train_session(session_id=104)
    elapsed = time.time() - start
    
    print(f"Elapsed Time for {benchmark_steps} steps: {elapsed:.4f} sec")
    
    # 推定
    total_steps = 10_000_000 # 1000万
    estimated_per_session = elapsed * (total_steps / benchmark_steps)
    print(f"Estimated time per session (10M steps): {estimated_per_session:.2f} sec")
    
    print("Trainer Extended Test OK.\n")

def test_reproducibility(env):
    print("--- Reproducibility Test ---")
    c = Config()
    c.max_steps = 5000
    c.convergence_window = 100
    
    agent1 = Agent(c, env, 0)
    agent2 = Agent(c, env, 1)
    trainer = Trainer(c, env, agent1, agent2)
    
    # 同じセッションIDで2回実行
    print("Running Session 42 (Run 1)...")
    res1 = trainer.train_session(session_id=42)
    print("Running Session 42 (Run 2)...")
    res2 = trainer.train_session(session_id=42)
    
    # 結果の比較
    match = True
    if res1['avg_price'] != res2['avg_price']: match = False
    if res1['steps'] != res2['steps']: match = False
    
    # Q行列の比較
    if not np.allclose(res1['q_matrices'][0], res2['q_matrices'][0]): match = False
    
    if match:
        print("OK: Results are perfectly reproducible.")
    else:
        print("FAIL: Results differ for same session_id.")
        print(f"Run 1 Price: {res1['avg_price']}, Run 2 Price: {res2['avg_price']}")
        
    print("Reproducibility Test OK.\n")

def test_realistic_convergence(env):
    print("--- Realistic Convergence Test ---")
    c = Config()
    # 本番に近い設定（ただし時間は節約）
    c.max_steps = 500_000 
    c.convergence_window = 5_000
    
    print(f"Running realistic simulation (Max Steps: {c.max_steps:,}, Window: {c.convergence_window:,})...")
    
    agent1 = Agent(c, env, 0)
    agent2 = Agent(c, env, 1)
    trainer = Trainer(c, env, agent1, agent2)
    
    start = time.time()
    res = trainer.train_session(session_id=777)
    elapsed = time.time() - start
    
    print(f"Converged: {res['converged']}")
    print(f"Steps to Convergence: {res['steps']:,}")
    print(f"Elapsed Time: {elapsed:.4f} sec")
    if res['converged']:
        print(f"Avg Price: {res['avg_price']:.4f}")
        print(f"Avg Profit: {res['avg_profit']:.4f}")
        
    if res['converged']:
        print("OK: Converged within realistic steps.")
    else:
        print("WARNING: Did not converge. Need larger max_steps?")
        
    print("Realistic Convergence Test OK.\n")

def test_memory_usage():
    print("--- Memory Usage Test ---")
    tracemalloc.start()
    
    # メモリ計測のため、メインの処理を一度流す（軽量Configで）
    class MemConfig(Config):
        max_steps = 1000
        n_sessions = 5
        convergence_window = 100
        
    c = MemConfig()
    # main関数を呼び出し（出力抑制してもよいがそのまま）
    main(test_config=c)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1024**2:.2f} MB")
    print(f"Peak memory usage:    {peak / 1024**2:.2f} MB")
    
    print("Memory Usage Test OK.\n")

def test_evaluator(env):
    print("--- Evaluator Test ---")
    c = Config()
    res0 = {'session_id': 0, 'converged': False, 'steps': 1000, 'avg_price': None, 'avg_profit': None}
    res1 = {'session_id': 1, 'converged': True, 'steps': 500, 'avg_price': 1.5, 'avg_profit': 0.23}
    res2 = {'session_id': 2, 'converged': True, 'steps': 600, 'avg_price': 1.9, 'avg_profit': 0.33}
    
    results = [res0, res1, res2]
    evaluator = Evaluator(c, results, env)
    
    stats = evaluator.get_statistics()
    assert stats['converged_sessions'] == 2
    
    evaluator.print_summary()
    best_id = evaluator.get_best_session_id()
    assert best_id == 2
    
    evaluator.plot_price_histogram("test_hist.png")
    if os.path.exists("test_hist.png"):
        os.remove("test_hist.png")
    print("Evaluator OK.\n")

def test_impulse_response(env):
    print("--- Impulse Response Test ---")
    c = Config()
    n_states = c.m_grid ** 2
    n_actions = c.m_grid
    q1 = np.random.rand(n_states, n_actions)
    q2 = np.random.rand(n_states, n_actions)
    
    target_result = {
        'q_matrices': [q1, q2],
        'final_state': 0,
        'converged': True,
        'avg_price': 1.5,
        'avg_profit': 0.2
    }
    
    analyzer = ImpulseResponseAnalyzer(c, env, target_result)
    analyzer.run_simulation()
    
    analyzer.plot_impulse_response("test_irf_price.png", "test_irf_profit.png")
    if os.path.exists("test_irf_price.png"):
        os.remove("test_irf_price.png")
        os.remove("test_irf_profit.png")
    print("Impulse Response Analyzer OK.\n")

def test_main_integration():
    print("--- Main Integration Test ---")
    
    class TestMainConfig(Config):
        max_steps = 2000
        n_sessions = 2
        convergence_window = 100
        l_buffer = 100
    
    c = TestMainConfig()
    main(test_config=c)
    
    # 確認
    subdirs = sorted([d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))])
    if subdirs:
        latest_dir = os.path.join("results", subdirs[-1])
        print(f"Checking output directory: {latest_dir}")
        if os.path.exists(os.path.join(latest_dir, "summary.json")):
            print("  OK: summary.json exists.")
        else:
            print("  FAIL: summary.json NOT found.")
            
    print("Main Integration Test Passed.\n")

if __name__ == "__main__":
    test_config()
    env = test_environment()
    test_agent(env)
    test_trainer(env)
    test_trainer_extended(env)
    test_reproducibility(env)    # 新規追加
    test_realistic_convergence(env) # 新規追加
    test_evaluator(env)
    test_impulse_response(env)
    test_main_integration()
    test_memory_usage()          # 新規追加
