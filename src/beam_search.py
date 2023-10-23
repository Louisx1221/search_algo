# beam_search.py
# Beam search
# louisx1221@gmail.com
# 2023/10/11

import numpy as np

class Candidate():
    def __init__(self, state = None):
        self.seq = []
        self.seq_rew = []
        self.reward = 0.
        self.state = state

class BeamSearch():
    """
    Beam search
    Minimizing the objective function
    func:       objective function      (score, state_next = func(state, idx))
    state0:     initial state
    cand_num:   number of the candidates
    seq_len:    length of the sequence
    width:      width of the beam
    """

    def __init__(
        self,
        func,
        state0,
        cand_num = 10,
        seq_len = 5,
        width = 3
    ):
        super().__init__()
        self.func = func
        self.state0 = state0
        self.state_len = len(state0)
        self.cand_num = cand_num
        self.seq_len = seq_len
        self.width = width

    def search(self):
        # Initialize candidates.
        candidates = [Candidate(self.state0)]

        # Loop over words.
        for _ in range(self.seq_len):
            candidates_new = []
            # Loop over candidates.
            for i in range(len(candidates)):
                # Candidate details.
                seq = candidates[i].seq
                seq_rew = candidates[i].seq_rew
                reward = candidates[i].reward

                # Predict next token.
                score_next = np.zeros(self.cand_num)
                state_next = np.zeros([self.cand_num, self.state_len])
                for j in range(self.cand_num):
                    if j in seq:
                        score_next[j] = np.inf
                    else:
                        score_next[j], state_next[j] = self.func(candidates[i].state, j)

                # Find top predictions.
                idx_top = score_next.argsort()[:self.width]
                score_top = score_next[idx_top]

                # Loop over top predictions.
                for j in range(self.width):
                    candidate = Candidate()

                    # Update candidate details.
                    candidate.seq = seq.copy()
                    candidate.seq.append(idx_top[j])
                    candidate.seq_rew = seq_rew.copy()
                    candidate.seq_rew.append(score_top[j])
                    candidate.reward = reward + score_top[j]
                    candidate.state = state_next[idx_top[j]].copy()

                    # Add to new candidates.
                    candidates_new.append(candidate)

            # Get top candidates.
            candidates_new.sort(key=lambda x: x.reward)
            candidates = candidates_new[:self.width].copy()

        # Get top candidate.
        return candidates[0] 