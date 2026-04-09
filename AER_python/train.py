import numpy as np
import time
from collections import deque
from numba import jit

@jit(nopython=True)
def get_action_numba(state, q_values, m_grid, epsilon):
    """
    Numba用 行動選択 (epsilon-greedy)
    """
    if np.random.random() < epsilon:
        return np.random.randint(m_grid)
    else:
        # greedy
        # q_values: (m_grid,)
        max_q = -1e20 # 十分小さい値
        for i in range(m_grid):
            if q_values[i] > max_q:
                max_q = q_values[i]
        
        # 最大値を持つインデックスを探す (Ties breaking)
        # より単純な実装: もう一度ループして候補を集める
        # Numba の np.random.choice は配列を受け取れる
        candidates_arr = np.zeros(m_grid, dtype=np.int32)
        idx = 0
        for i in range(m_grid):
            if q_values[i] >= max_q - 1e-12:
                candidates_arr[idx] = i
                idx += 1
        
        # 0 ~ idx-1 の範囲からランダムに選ぶ
        if idx > 0:
            rand_idx = np.random.randint(idx)
            return candidates_arr[rand_idx]
        else:
            # 万が一候補が見つからない場合（浮動小数点誤差など）
            # 単純にargmaxを再計算して返す
            best_a = 0
            current_max = -1e20
            for i in range(m_grid):
                if q_values[i] > current_max:
                    current_max = q_values[i]
                    best_a = i
            return best_a

@jit(nopython=True)
def get_greedy_action_numba(state, q_values, m_grid):
    """
    Numba用 Greedy行動選択 (収束判定用: 同点時は最小インデックス)
    """
    max_q = -1e20
    best_action = 0
    
    for i in range(m_grid):
        if q_values[i] > max_q:
            max_q = q_values[i]
            best_action = i
        elif q_values[i] == max_q:
            # 最小インデックス優先なら何もしない (今のbest_actionの方が小さい)
            pass
            
    return best_action

@jit(nopython=True)
def run_simulation_jit(
    max_steps, 
    convergence_window, 
    m_grid, 
    n_states,
    alpha, 
    delta, 
    beta_decay, 
    initial_epsilon,
    initial_s,
    q1, q2,
    profits, # (n_states, 2)
    prices,  # (m_grid,)
    l_buffer,
    seed     # 追加: シード値
):
    """
    JITコンパイルされた学習ループ
    """
    # Numba内の乱数シードを設定
    np.random.seed(seed)
    
    # 変数初期化
    s = initial_s
    epsilon = initial_epsilon
    stable_count = 0
    converged = False
    t_converged = 0
    
    # 最適行動表 (n_states, 2)
    optimal_strategies = np.zeros((n_states, 2), dtype=np.int32)
    
    # 初期最適行動の計算
    for i in range(n_states):
        optimal_strategies[i, 0] = get_greedy_action_numba(i, q1[i], m_grid)
        optimal_strategies[i, 1] = get_greedy_action_numba(i, q2[i], m_grid)
        
    # 履歴バッファ (リングバッファとして実装)
    # shape: (l_buffer, 4) -> p1, p2, pi1, pi2
    hist_buffer = np.zeros((l_buffer, 4), dtype=np.float64)
    hist_ptr = 0
    hist_full = False
    
    # 学習ループ
    for t in range(max_steps):
        # 行動選択
        a1 = get_action_numba(s, q1[s], m_grid, epsilon)
        a2 = get_action_numba(s, q2[s], m_grid, epsilon)
        
        # 次状態と報酬
        s_next = a1 * m_grid + a2
        r1 = profits[s_next, 0]
        r2 = profits[s_next, 1]
        
        # バッファ記録
        hist_buffer[hist_ptr, 0] = prices[a1]
        hist_buffer[hist_ptr, 1] = prices[a2]
        hist_buffer[hist_ptr, 2] = r1
        hist_buffer[hist_ptr, 3] = r2
        hist_ptr = (hist_ptr + 1) % l_buffer
        if hist_ptr == 0:
            hist_full = True
            
        # Q値更新
        # Agent 1
        max_q1_next = -1e20
        for i in range(m_grid):
            if q1[s_next, i] > max_q1_next:
                max_q1_next = q1[s_next, i]
        td1 = r1 + delta * max_q1_next
        q1[s, a1] += alpha * (td1 - q1[s, a1])
        
        # Agent 2
        max_q2_next = -1e20
        for i in range(m_grid):
            if q2[s_next, i] > max_q2_next:
                max_q2_next = q2[s_next, i]
        td2 = r2 + delta * max_q2_next
        q2[s, a2] += alpha * (td2 - q2[s, a2])
        
        # 収束判定
        opt1 = get_greedy_action_numba(s, q1[s], m_grid)
        opt2 = get_greedy_action_numba(s, q2[s], m_grid)
        
        if opt1 == optimal_strategies[s, 0] and opt2 == optimal_strategies[s, 1]:
            stable_count += 1
        else:
            stable_count = 0
            optimal_strategies[s, 0] = opt1
            optimal_strategies[s, 1] = opt2
            
        if stable_count >= convergence_window:
            converged = True
            t_converged = t - convergence_window + 1
            # 終了時の状態を更新しておく
            s = s_next
            break
            
        # 次の状態へ
        s = s_next
        
        # 探索率減衰
        epsilon *= beta_decay
        
    return converged, t_converged, q1, q2, s, hist_buffer, hist_ptr, hist_full


class Trainer:
    """
    学習ループを管理するクラス (Numba高速化版)
    """
    def __init__(self, config, env, agent_1, agent_2):
        self.config = config
        self.env = env
        self.agents = [agent_1, agent_2]

    def train_session(self, session_id):
        # 1. 初期化
        np.random.seed(session_id)
        for agent in self.agents:
            agent.reset()
            
        # Numba関数に渡すデータの準備
        # Q行列はコピーして渡す (Numba内で更新される)
        q1 = self.agents[0].Q.copy()
        q2 = self.agents[1].Q.copy()
        
        # パラメータ
        m_grid = self.config.m_grid
        n_states = m_grid ** 2
        alpha = self.config.alpha
        delta = self.config.delta
        # config.beta_decay は指数部の係数なので、減衰率自体(beta)を渡す
        beta = self.agents[0].beta 
        
        s_init = self.env.get_initial_state()
        
        start_time = time.time()
        
        # 2. 高速シミュレーション実行
        converged, t_conv, q1_final, q2_final, s_final, hist_buf, hist_ptr, hist_full = run_simulation_jit(
            self.config.max_steps,
            self.config.convergence_window,
            m_grid,
            n_states,
            alpha,
            delta,
            beta,
            self.agents[0].epsilon,
            s_init,
            q1, q2,
            self.env.profits,
            self.env.prices,
            self.config.l_buffer,
            session_id # シードとして渡す
        )
        
        elapsed_time = time.time() - start_time
        
        # AgentのQ値を更新
        self.agents[0].Q = q1_final
        self.agents[1].Q = q2_final
        
        # 3. 結果集計
        result = {
            "session_id": session_id,
            "converged": converged,
            "steps": t_conv if converged else self.config.max_steps,
            "elapsed_time": elapsed_time
        }
        
        # 収束有無に関わらず、バッファ内のデータから統計量を計算する
        # (未収束の場合は直近l_buffer期間の平均となる)
        valid_data = []
        if not hist_full:
            # まだ一周していない場合: 0 ~ hist_ptr-1
            valid_data = hist_buf[:hist_ptr]
        else:
            # 一周している場合: 全データ有効
            valid_data = hist_buf
        
        if len(valid_data) > 0:
            # valid_data columns: p1, p2, pi1, pi2
            # 平均価格: (mean(p1) + mean(p2)) / 2
            avg_p = (np.mean(valid_data[:, 0]) + np.mean(valid_data[:, 1])) / 2.0
            # 平均利潤: (mean(pi1) + mean(pi2)) / 2
            avg_pi = (np.mean(valid_data[:, 2]) + np.mean(valid_data[:, 3])) / 2.0
            
            result["avg_price"] = avg_p
            result["avg_profit"] = avg_pi
            result["q_matrices"] = [q1_final.copy(), q2_final.copy()]
            result["final_state"] = s_final
        else:
            # データが全くない場合（あり得ないが念のため）
            result["avg_price"] = None
            result["avg_profit"] = None
            result["q_matrices"] = None
            result["final_state"] = None
            
        return result
