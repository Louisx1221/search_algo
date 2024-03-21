# traversal.py
# Traversal algorithm
# louisx1221@gmail.com
# 2024/03/20

import numpy as np
import copy
import time

class Candidate():
    def __init__(self, state = None):
        self.seq = []
        self.seq_rew = []
        self.reward = 0.
        self.state = state

class Traversal():
    """
    Traversal algorithm
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
        self.cand_num = cand_num
        self.seq_len = seq_len

    def search(self):
        t0 = time.time()
        # Initialize sequence
        seq = [0] * self.seq_len
        score_best = np.inf
        seq_best = seq.copy()

        # Loop
        while sum(seq) < self.seq_len * (self.cand_num - 1):
            # Update sequence
            for i in range(self.seq_len):
                seq[i] += 1
                if seq[i] == self.cand_num:
                    seq[i] = 0
                else:
                    break

            # Check repeat
            flag_repeat = False
            for i in range(self.seq_len):
                for j in range(i + 1, self.seq_len):
                    if seq[i] == seq[j]:
                        flag_repeat = True
                        break
                if flag_repeat:
                    break
            if flag_repeat:
                continue

            # Objective function value
            score = 0.
            state_i = copy.copy(self.state0)
            for i in range(self.seq_len):
                score_i, state_i = self.func(state_i, seq[i])
                score += score_i

            # Best candidate
            if score < score_best:
                score_best = score
                seq_best = seq.copy()
        res = Candidate()
        res.seq = seq_best.copy()
        res.reward = score_best
        res.time = time.time() - t0
        return res