import numpy as np
import matplotlib.pyplot as plt
import os

class Evaluator:
    """
    実験結果の集計・分析を行うクラス
    準コード「4. Evaluation & Summary Statistics」に対応
    """
    def __init__(self, config, results, env):
        """
        config: Configオブジェクト
        results: 全セッションの結果リスト (辞書のリスト)
        env: Environmentオブジェクト (理論値参照用)
        """
        self.config = config
        self.results = results
        self.env = env
        
        # 結果の抽出・整理
        self._parse_results()

    def _parse_results(self):
        """
        全セッションの結果から必要なデータを抽出してメンバ変数に格納
        """
        self.converged_flags = []
        self.converged_steps = []
        self.avg_prices = []
        self.avg_profits = []
        self.converged_sessions = [] # 収束したセッションのIDやインデックス
        self.delta_indices = []      # 協力指数 Delta
        
        for i, res in enumerate(self.results):
            self.converged_flags.append(res['converged'])
            
            if res['converged']:
                self.converged_sessions.append(i)
                self.converged_steps.append(res['steps'])
                self.avg_prices.append(res['avg_price'])
                self.avg_profits.append(res['avg_profit'])
                
                # 協力指数 Delta = (pi_bar - pi_N) / (pi_M - pi_N)
                # 対称ゲームなので、理論値として企業1の値を参照する
                pi_nash_val = self.env.pi_nash[0]
                pi_mon_val = self.env.pi_monopoly[0]
                
                delta = (res['avg_profit'] - pi_nash_val) / \
                        (pi_mon_val - pi_nash_val)
                self.delta_indices.append(delta)

    def get_statistics(self):
        """
        統計量を辞書形式で返す (main.pyでの保存用)
        """
        n_total = len(self.results)
        n_conv = len(self.converged_sessions)
        rate_conv = n_conv / n_total if n_total > 0 else 0
        
        stats = {
            "total_sessions": n_total,
            "converged_sessions": n_conv,
            "convergence_rate": rate_conv,
            "nash_price": self.config.p_nash,
            "monopoly_price": self.config.p_monopoly,
            "nash_profit": self.env.pi_nash[0],
            "monopoly_profit": self.env.pi_monopoly[0]
        }
        
        if n_conv > 0:
            stats.update({
                "avg_steps_mean": np.mean(self.converged_steps),
                "avg_steps_std": np.std(self.converged_steps),
                "avg_price_mean": np.mean(self.avg_prices),
                "avg_price_std": np.std(self.avg_prices),
                "avg_profit_mean": np.mean(self.avg_profits),
                "avg_profit_std": np.std(self.avg_profits),
                "delta_mean": np.mean(self.delta_indices),
                "delta_std": np.std(self.delta_indices) if len(self.delta_indices) > 0 else 0
            })
        else:
            stats.update({
                "avg_steps_mean": None, "avg_steps_std": None,
                "avg_price_mean": None, "avg_price_std": None,
                "avg_profit_mean": None, "avg_profit_std": None,
                "delta_mean": None, "delta_std": None
            })
            
        return stats

    def print_summary(self):
        """
        統計量を計算して表示する
        """
        stats = self.get_statistics()
        
        print("=" * 50)
        print("4. Evaluation & Summary Statistics")
        print("=" * 50)
        print(f"Total Sessions: {stats['total_sessions']}")
        print(f"Converged Sessions: {stats['converged_sessions']} ({stats['convergence_rate']:.2%})")
        
        if stats['converged_sessions'] == 0:
            print("No sessions converged.")
            return
            
        print(f"Convergence Steps (T_conv): Mean = {stats['avg_steps_mean']:.2f}, Std = {stats['avg_steps_std']:.2f}")
        print(f"Average Price (p_bar): Mean = {stats['avg_price_mean']:.4f}, Std = {stats['avg_price_std']:.4f}")
        print(f"Average Profit (pi_bar): Mean = {stats['avg_profit_mean']:.4f}, Std = {stats['avg_profit_std']:.4f}")
        print(f"Cooperation Index (Delta): Mean = {stats['delta_mean']:.4f}")
        
        # 理論値との比較
        print("-" * 30)
        print(f"Nash Price: {stats['nash_price']:.4f}, Monopoly Price: {stats['monopoly_price']:.4f}")
        print(f"Nash Profit: {stats['nash_profit']:.4f}, Monopoly Profit: {stats['monopoly_profit']:.4f}")
        print("=" * 50)

    def plot_price_histogram(self, save_path="histogram_price.png"):
        """
        収束価格のヒストグラムを作成して保存する
        """
        if not self.avg_prices:
            print("No data to plot.")
            return

        plt.figure(figsize=(10, 6))
        
        # ヒストグラムの描画
        plt.hist(self.avg_prices, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Nash価格と独占価格のライン
        plt.axvline(self.config.p_nash, color='red', linestyle='--', linewidth=2, label='Nash Price')
        plt.axvline(self.config.p_monopoly, color='green', linestyle='--', linewidth=2, label='Monopoly Price')
        
        plt.xlabel('Average Price', fontsize=14)
        plt.ylabel('Frequency (Number of Sessions)', fontsize=14)
        plt.title('Distribution of Converged Prices', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 保存
        try:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        except Exception as e:
            print(f"Error saving histogram: {e}")
        finally:
            plt.close()

    def get_best_session_id(self):
        """
        最も協力指数(Delta)が高く、かつ収束状態が安定的(Fixed Point)なセッションIDを返す。
        Fixed Pointなセッションがない場合は、単純にDeltaが最大のセッションを返す。
        """
        if not self.delta_indices:
            return None

        pi_nash = self.env.pi_nash[0]
        pi_monopoly = self.env.pi_monopoly[0]
        
        candidates = []
        fixed_point_candidates = []
        
        m_grid = self.config.m_grid
        
        for r in self.results:
            if r['converged'] and r['avg_profit'] is not None:
                delta = (r['avg_profit'] - pi_nash) / (pi_monopoly - pi_nash)
                candidates.append((r['session_id'], delta))
                
                # Check for stability (Fixed Point check)
                q = r['q_matrices']
                # Ensure q is numpy array
                if isinstance(q, list):
                    q = np.array(q)

                s = r['final_state']
                
                # Simulate a few steps to check if actions change
                prices = []
                curr_s = int(s)
                
                # Check 10 steps
                for _ in range(10):
                    a1 = np.argmax(q[0, curr_s, :])
                    a2 = np.argmax(q[1, curr_s, :])
                    prices.append((a1, a2))
                    curr_s = a1 * m_grid + a2
                
                # If variance of actions is 0, it's a fixed point
                if np.std(prices) == 0:
                     fixed_point_candidates.append((r['session_id'], delta))

        # First, try to find best among fixed point sessions
        if fixed_point_candidates:
            # Sort by delta descending
            fixed_point_candidates.sort(key=lambda x: x[1], reverse=True)
            print(f"Found {len(fixed_point_candidates)} fixed point sessions. Selecting best Delta.")
            return fixed_point_candidates[0][0]
            
        # Fallback to general best delta
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            print("No fixed point sessions found. Selecting best Delta from all.")
            return candidates[0][0]
            
        return None
