# traversal.py
# Traversal algorithm
# louisx1221@gmail.com
# 2024/03/21

import numpy as np
import copy
import time

class Candidate():
    def __init__(self, state = None):
        self.seq = []
        self.seq_rew = []
        self.reward = np.inf
        self.seq_sta = [state]
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
        cand = Candidate(self.state0)
        for _ in range(self.seq_len):
            cand = self.expand(cand)
        res = copy.deepcopy(cand)

        # Loop
        while True:
            cand = self.next(cand)
            if len(cand.seq) == 0:
                break
            if cand.reward < res.reward:
                res = copy.deepcopy(cand)

        res.time = time.time() - t0
        return res

    def expand(self, cand):
        act = 0
        while act in cand.seq:
            act += 1
            if act >= self.cand_num:
                break

        rew, sta = self.func(cand.seq_sta[-1], act)
        cand.seq += [act]
        cand.seq_rew += [rew]
        cand.seq_sta += [sta]
        if len(cand.seq) == self.seq_len:
            cand.reward = sum(cand.seq_rew)
        return cand

    def next(self, cand):
        act = cand.seq[-1]
        while act in cand.seq:
            act += 1
            if act >= self.cand_num:
                del cand.seq[-1], cand.seq_rew[-1], cand.seq_sta[-1]
                if len(cand.seq) == 0:
                    return cand
                act = cand.seq[-1]
        del cand.seq[-1], cand.seq_rew[-1], cand.seq_sta[-1]

        rew, sta = self.func(cand.seq_sta[-1], act)
        cand.seq += [act]
        cand.seq_rew += [rew]
        cand.seq_sta += [sta]
        if len(cand.seq) == self.seq_len:
            cand.reward = sum(cand.seq_rew)

        for _ in range(self.seq_len - len(cand.seq)):
            cand = self.expand(cand)
        return cand