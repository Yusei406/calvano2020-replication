import numpy as np

class Agent:
    """
    Q学習エージェント
    準コード「2. Agent (Q-learning)」に対応
    """
    def __init__(self, config, environment, agent_id):
        """
        config: Configクラスのインスタンス
        environment: Environmentクラスのインスタンス (利潤表参照用)
        agent_id: 0 (Agent 1) or 1 (Agent 2)
        """
        self.config = config
        self.env = environment
        self.id = agent_id
        
        # 探索パラメータ
        self.epsilon = 1.0  # 初期値 eps_0
        
        # 減衰率 beta = exp(-MExpl / itersPerEpisode)
        # config.beta_decay が MExpl に対応
        self.beta = np.exp(-config.beta_decay / config.iters_per_episode)

        # Q行列の初期化 (n_states x n_actions)
        self.Q = self._initialize_Q()

    def reset(self):
        """
        セッションごとのリセット処理
        """
        self.epsilon = 1.0
        self.Q = self._initialize_Q()

    def _initialize_Q(self):
        """
        Q行列を初期化する
        初期値: 相手が一様ランダムに行動すると仮定したときの割引累積利潤の期待値
        """
        m = self.config.m_grid
        n_states = self.config.m_grid ** 2
        
        # 初期Q行列 (全要素ゼロで作成し、計算値で埋める)
        Q = np.zeros((n_states, m))
        
        # 全行動 a_self について、相手がランダムな場合の期待利潤を計算
        # Q値は初期状態 s に依存せず、自分の行動 a_self だけで決まる
        # (初期化の定義より、どの s に対しても同じ値を埋める)
        
        expected_profits = np.zeros(m)
        
        for a_self in range(m):
            sum_profit = 0.0
            for a_opp in range(m):
                # 利潤表から参照するための状態インデックスを計算
                # 利潤表 PI[s, i] は s = a1 * m + a2 で定義されている
                if self.id == 0:
                    # 自分が Agent 1 (a1)
                    s_temp = a_self * m + a_opp
                else:
                    # 自分が Agent 2 (a2)
                    s_temp = a_opp * m + a_self
                
                # 利潤を加算
                sum_profit += self.env.profits[s_temp, self.id]
            
            # 期待値計算 (相手の手は m 通り等確率)
            expected_profits[a_self] = sum_profit / m
        
        # 割引累積期待利潤 = E[pi] / (1 - delta)
        initial_q_values = expected_profits / (1.0 - self.config.delta)
        
        # 全状態 s に対して同じ値をコピー
        for s in range(n_states):
            Q[s, :] = initial_q_values
            
        return Q

    def get_action(self, state):
        """
        状態 s_t に基づき行動 a_{i,t} を選択する (epsilon-greedy)
        """
        # 確率 epsilon でランダム行動
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.config.m_grid)
        else:
            # 確率 1-epsilon でQ値最大化行動
            action = self.get_greedy_action(state)
            
        # 探索率の更新 (準コード: epsilon_t = epsilon_0 * beta^t)
        # 実装上は毎ステップ beta を掛けることで実現
        self.epsilon *= self.beta
        
        return action

    def get_greedy_action(self, state):
        """
        Q値が最大になる行動を選択する
        同点の場合はランダムに選択 (Ties breaking)
        """
        q_values = self.Q[state, :]
        max_q = np.max(q_values)
        
        # 最大値を持つインデックスを全て取得
        candidates = np.where(q_values == max_q)[0]
        
        # ランダムに1つ選ぶ
        action = np.random.choice(candidates)
        
        return action

    def update_Q(self, s, a, reward, s_next):
        """
        Q学習の更新ルールに従ってQ値を更新する
        
        Q(s, a) <- Q(s, a) + alpha * [ reward + delta * max_a' Q(s', a') - Q(s, a) ]
        """
        # 次状態での最大Q値 (max_a' Q(s', a'))
        max_q_next = np.max(self.Q[s_next, :])
        
        # TDターゲット
        td_target = reward + self.config.delta * max_q_next
        
        # 更新
        self.Q[s, a] += self.config.alpha * (td_target - self.Q[s, a])

