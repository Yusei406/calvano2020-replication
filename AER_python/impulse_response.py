import numpy as np
import matplotlib.pyplot as plt

class ImpulseResponseAnalyzer:
    """
    インパルス応答分析を行うクラス
    準コード「5. Impulse Response Analysis」に対応
    """
    def __init__(self, config, env, result_target):
        """
        result_target: ターゲットセッションの実行結果辞書 (q_matrices, final_state を含む)
        """
        self.config = config
        self.env = env
        self.q_matrices = result_target['q_matrices']
        self.initial_state = result_target['final_state']
        
        # シミュレーション範囲
        self.t_start = -5
        self.t_end = 20
        self.simulation_steps = self.t_end - self.t_start + 1 # 26 steps
        self.history = None

    def run_simulation(self):
        """
        インパルス応答シミュレーションを実行する
        t=0 で Agent 1 が裏切り (Static Best Response) を行う
        """
        # シミュレーション結果の保存用
        prices_1 = []
        prices_2 = []
        profits_1 = []
        profits_2 = []
        
        # 初期状態の設定
        current_state = self.initial_state
        
        # 前の状態から価格を復元する（t=0の裏切り計算用）
        m = self.config.m_grid
        
        # シミュレーションループ
        # t は -5 から 20 まで
        for t in range(self.t_start, self.t_end + 1):
            
            # --- 行動選択 ---
            if t == 0:
                # t=0: Deviation (裏切り)
                
                # Agent 1: 直前期の相手価格 p2_{-1} に対する最適反応
                prev_a2 = current_state % m
                p2_prev = self.env.prices[prev_a2]
                
                # 一期利潤を最大化する a1 を探索
                best_profit = -np.inf
                best_a1 = 0
                
                for a1_cand in range(m):
                    p1_cand = self.env.prices[a1_cand]
                    # 利潤計算 (Environmentのメソッドを利用)
                    profits = self.env._compute_profit_for_price_pair(p1_cand, p2_prev)
                    if profits[0] > best_profit:
                        best_profit = profits[0]
                        best_a1 = a1_cand
                        
                action1 = best_a1
                
                # Agent 2: Q値最大化行動 (通常通り)
                q_values_2 = self.q_matrices[1][current_state, :]
                max_q_2 = np.max(q_values_2)
                candidates_2 = np.where(q_values_2 == max_q_2)[0]
                action2 = np.random.choice(candidates_2) # 同点ならランダム
                
            else:
                # t != 0: 両者ともQ値最大化行動 (Greedy)
                # alpha=0, epsilon=0 なので、学習・探索なし
                
                # Agent 1
                q_values_1 = self.q_matrices[0][current_state, :]
                max_q_1 = np.max(q_values_1)
                candidates_1 = np.where(q_values_1 == max_q_1)[0]
                action1 = np.random.choice(candidates_1)
                
                # Agent 2
                q_values_2 = self.q_matrices[1][current_state, :]
                max_q_2 = np.max(q_values_2)
                candidates_2 = np.where(q_values_2 == max_q_2)[0]
                action2 = np.random.choice(candidates_2)

            # --- 環境ステップ ---
            next_state, rewards = self.env.step(current_state, [action1, action2])
            
            # 記録
            p1 = self.env.prices[action1]
            p2 = self.env.prices[action2]
            
            prices_1.append(p1)
            prices_2.append(p2)
            profits_1.append(rewards[0])
            profits_2.append(rewards[1])
            
            # 次の状態へ
            current_state = next_state

        self.history = {
            't': list(range(self.t_start, self.t_end + 1)),
            'p1': prices_1, 'p2': prices_2,
            'pi1': profits_1, 'pi2': profits_2
        }

    def plot_impulse_response(self, save_path_price="irf_price.png", save_path_profit="irf_profit.png"):
        """
        価格と利潤の推移グラフを作成・保存する
        """
        if self.history is None:
            print("Simulation not run yet.")
            return

        t = self.history['t']
        
        # --- 価格グラフ ---
        plt.figure(figsize=(10, 6))
        plt.plot(t, self.history['p1'], label='Firm 1 (Deviator)', marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.plot(t, self.history['p2'], label='Firm 2', marker='x', linestyle='--', linewidth=2, markersize=8)
        
        # 理論値ライン
        plt.axhline(self.config.p_nash, color='red', linestyle=':', label='Nash Price', linewidth=1.5)
        plt.axhline(self.config.p_monopoly, color='green', linestyle=':', label='Monopoly Price', linewidth=1.5)
        
        plt.xlabel('Period (t)', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.title('Impulse Response: Price Trajectory', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(self.t_start, self.t_end + 1, 5), fontsize=12) # 目盛り調整
        plt.yticks(fontsize=12)
        
        plt.savefig(save_path_price)
        print(f"Price IRF saved to {save_path_price}")
        plt.close()
        
        # --- 利潤グラフ ---
        plt.figure(figsize=(10, 6))
        plt.plot(t, self.history['pi1'], label='Firm 1 (Deviator)', marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.plot(t, self.history['pi2'], label='Firm 2', marker='x', linestyle='--', linewidth=2, markersize=8)
        
        # 理論値ライン
        # 対称ゲームなので企業1の値を代表として使用
        pi_nash = self.env.pi_nash[0]
        pi_mon = self.env.pi_monopoly[0]
        
        plt.axhline(pi_nash, color='red', linestyle=':', label='Nash Profit', linewidth=1.5)
        plt.axhline(pi_mon, color='green', linestyle=':', label='Monopoly Profit', linewidth=1.5)
        
        plt.xlabel('Period (t)', fontsize=14)
        plt.ylabel('Profit', fontsize=14)
        plt.title('Impulse Response: Profit Trajectory', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(self.t_start, self.t_end + 1, 5), fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.savefig(save_path_profit)
        print(f"Profit IRF saved to {save_path_profit}")
        plt.close()
