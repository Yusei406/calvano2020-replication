import numpy as np

class Environment:
    """
    経済環境およびMDPの状態遷移を管理するクラス
    準コード「1. Environment」に対応
    """
    
    def __init__(self, config):
        """
        config: Configクラスのインスタンス
        """
        self.config = config
        
        # 1.1 価格グリッドの生成
        self.prices = self._compute_price_grid()
        
        # 1.1 利潤表の事前計算 (m^2 x 2)
        # 行: 状態 s = a1 * m + a2
        # 列: 企業インデックス (0 or 1)
        self.profits = self._compute_payoff_matrix()
        
        # 1.1 ナッシュ利潤と独占利潤の計算
        self.pi_nash = self._compute_profit_for_price_pair(config.p_nash, config.p_nash)
        self.pi_monopoly = self._compute_profit_for_price_pair(config.p_monopoly, config.p_monopoly)

        # MDP空間のサイズ定義
        self.n_actions = config.m_grid
        self.n_states = config.m_grid ** 2

    def _compute_price_grid(self):
        """
        価格グリッドを生成する
        範囲: [p_nash - xi*(p_mon - p_nash), p_mon + xi*(p_mon - p_nash)]
        """
        p_n = self.config.p_nash
        p_m = self.config.p_monopoly
        xi = self.config.xi
        m = self.config.m_grid
        
        lower = p_n - xi * (p_m - p_n)
        upper = p_m + xi * (p_m - p_n)
        
        # numpy.linspace は start, stop を含む等間隔な点を生成する（倍精度）
        return np.linspace(lower, upper, m)

    def _compute_payoff_matrix(self):
        """
        全ての行動ペア (a1, a2) について利潤を計算し、テーブルに格納する
        """
        m = self.config.m_grid
        pi_matrix = np.zeros((m * m, 2))
        
        for a1 in range(m):
            for a2 in range(m):
                # 実価格の取得
                p1 = self.prices[a1]
                p2 = self.prices[a2]
                
                # 利潤計算 (企業1, 企業2)
                pi_vals = self._compute_profit_for_price_pair(p1, p2)
                
                # 状態インデックス s = a1 * m + a2
                s = a1 * m + a2
                
                pi_matrix[s, 0] = pi_vals[0]
                pi_matrix[s, 1] = pi_vals[1]
                
        return pi_matrix

    def _compute_profit_for_price_pair(self, p1, p2):
        """
        任意の価格ペア (p1, p2) に対する両企業の利潤を計算する
        (Logit需要モデル)
        """
        # パラメータ展開
        quality = self.config.quality # alpha_i
        mu = self.config.mu
        a0 = self.config.a_0
        cost = self.config.marginal_cost
        
        # Logit需要の分母項: exp((alpha_j - p_j) / mu)
        exp1 = np.exp((quality - p1) / mu)
        exp2 = np.exp((quality - p2) / mu)
        exp0 = np.exp(a0 / mu)
        
        denom = exp1 + exp2 + exp0
        
        # 需要量 q_i
        q1 = exp1 / denom
        q2 = exp2 / denom
        
        # 利潤 pi_i = (p_i - c_i) * q_i
        profit1 = (p1 - cost) * q1
        profit2 = (p2 - cost) * q2
        
        return np.array([profit1, profit2])

    # --- 1.2 MDP / 相互作用 ---

    def get_initial_state(self):
        """
        初期状態 s0 をランダムに返す
        a_i,0 ~ Uniform{0, ..., m-1}
        s0 = a1,0 * m + a2,0
        """
        return np.random.randint(0, self.n_states)

    def step(self, current_state, actions):
        """
        行動 a_t = (a1, a2) を受け取り、次状態 s_{t+1} と利潤 pi_t を返す
        
        actions: (a1, a2) のタプルまたはリスト
        return: next_state, rewards (numpy array of shape (2,))
        """
        a1, a2 = actions
        m = self.config.m_grid
        
        # 次状態 s_{t+1} = a1 * m + a2
        next_state = a1 * m + a2
        
        # 利潤 pi_t = PI[s_{t+1}]
        # ※ 準コードにある通り、学習中は事前計算したテーブルから参照する
        rewards = self.profits[next_state]
        
        return next_state, rewards

