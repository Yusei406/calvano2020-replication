import numpy as np

class Config:
    """
    実験パラメータ管理クラス
    Fortranコードの globals.f90 および A_InputParameters.txt に対応
    """
    # --- 経済環境パラメータ ---
    n_agents = 2             # 企業数 (n)
    quality = 2.0            # 製品品質 (a_i, alpha_i)
    marginal_cost = 1.0      # 限界費用 (c_i)
    mu = 0.25                # 需要の価格弾力性パラメータ (mu)
    a_0 = 0.0                # 外部選択肢品質 (a_0)

    # --- 価格グリッド設定 ---
    # 理論値としてのNash均衡価格と独占価格
    p_nash = 1.47293         # Nash均衡価格
    p_monopoly = 1.92498     # 独占価格
    
    m_grid = 15              # グリッド点数 (m)
    xi = 0.10                # 範囲拡張係数 (xi)
    
    # --- Q-learning パラメータ ---
    k_memory = 1             # 記憶長 (k) - 今回の実装では1で固定
    alpha = 0.15             # 学習率 (learning rate)
    delta = 0.95             # 割引因子 (delta)
    beta_decay = 0.1         # 探索減衰係数 (MExpl)
    
    # 探索率減衰のタイムスケール基準 (itersPerEpisode)
    # beta = exp(-beta_decay / iters_per_episode)
    iters_per_episode = 25000 

    # --- 実験設計パラメータ ---
    # 動作確認・卒論実験用に調整済みの設定
    # ※ 論文の完全再現（MExpl=0.005等）を行う場合は、
    #    max_stepsを 1.25*10^9、convergence_windowを 100,000 に増やす必要がある。
    
    convergence_window = 10000    # 収束判定窓 (W)
    max_steps = 10_000_000        # 最大反復回数 (収束しなかった場合の打ち切り)
    n_sessions = 100               # セッション数 (S)
    
    # 統計・分析用
    l_buffer = 100                # 平均価格計算用に保持する直近の期数

