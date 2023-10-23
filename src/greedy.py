# greedy.py
# Greedy algorithm
# louisx1221@gmail.com
# 2023/10/11

import numpy as np

class Candidate():
    def __init__(self, state = None):
        self.seq = []
        self.seq_rew = []
        self.reward = 0.
        self.state = state

class Greedy():
    """
    Greedy algorithm
    Minimizing the objective function
    func:       objective function      (score, state_next = func(state, idx))
    state0:     initial state
    cand_num:   number of the candidates
    seq_len:    length of the sequence
    """

    def __init__(
        self,
        func,
        state0,
        cand_num = 10,
        seq_len = 5
    ):
        super().__init__()
        self.func = func
        self.state0 = state0
        self.state_len = len(state0)
        self.cand_num = cand_num
        self.seq_len = seq_len

    def search(self):
        # Initialize candidate.
        candidate = Candidate(self.state0)

        # Loop over words.
        for _ in range(self.seq_len):
            # Predict next token.
            score_next = np.zeros(self.cand_num)
            state_next = np.zeros([self.cand_num, self.state_len])
            for j in range(self.cand_num):
                if j in candidate.seq:
                    score_next[j] = np.inf
                else:
                    score_next[j], state_next[j] = self.func(candidate.state, j)

            # Find top predictions.
            idx_top = score_next.argsort()[0]
            score_top = score_next[idx_top]

            # Update candidate details.
            candidate.seq.append(idx_top)
            candidate.seq_rew.append(score_top)
            candidate.reward += score_top
            candidate.state = state_next[idx_top].copy()

        # Get top candidate.
        return candidate