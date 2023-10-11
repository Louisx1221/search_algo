# beam_search.py
# Beam search
# louisx1221@gmail.com
# 2023/10/11

import numpy as np

class Candidate():
    def __init__(self, state = None):
        self.seq = []
        self.score = 0
        self.state = state

class BeamSearch():
    # Minimizing the objective function: score, state_next = func(state, idx)
    def __init__(
        self,
        func,
        state0,
        num_cand = 10,
        num_seq = 5,
        width = 3
    ):
        super().__init__()
        self.func = func
        self.state0 = state0
        self.num_cand = num_cand
        self.num_seq = num_seq
        self.width = width

    def search(self):
        # Initialize candidates.
        candidates = [Candidate(self.state0)]

        # Loop over words.
        for _ in range(self.num_seq):
            candidates_new = []
            # Loop over candidates.
            for i in range(len(candidates)):
                # Candidate details.
                seq = candidates[i].seq
                score = candidates[i].score

                # Predict next token.
                score_next = np.zeros(self.num_cand)
                state_next = np.zeros([self.num_cand, len(candidates[i].state)])
                for j in range(self.num_cand):
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
                    candidate.score = score + score_top[j]
                    candidate.state = state_next[idx_top[j]].copy()

                    # Add to new candidates.
                    candidates_new.append(candidate)

            # Get top candidates.
            candidates_new.sort(key=lambda x: x.score)
            candidates = candidates_new[:self.width].copy()

        # Get top candidate.
        return candidates[0] 