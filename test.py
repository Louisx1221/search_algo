# test.py

import numpy as np
import matplotlib.pyplot as plt

from src.beam_search import BeamSearch

def func(state, idx):
    state_next = np.array([np.cos(idx), np.sin(idx)])
    dist = np.linalg.norm(state_next - state)
    score = dist
    return score, state_next

if __name__ == '__main__':
    state0 = np.array([0.5, 0.5])

    search_algo = BeamSearch(func, state0, 100, 20, 5)

    res_best = search_algo.search()

    print(res_best.seq)
    print(res_best.score)

    x = [state0[0]]
    y = [state0[1]]
    for idx in res_best.seq:
        x.append(np.cos(idx))
        y.append(np.sin(idx))

    t0 = np.linspace(0, 360) * np.pi / 180.
    x0 = np.cos(t0)
    y0 = np.sin(t0)

    plt.figure()
    plt.plot(x0, y0)
    plt.plot(x, y)
    plt.axis("equal")
    plt.grid("on")

    plt.show()