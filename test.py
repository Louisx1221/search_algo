# test.py

import numpy as np
import matplotlib.pyplot as plt

from src.greedy import Greedy
from src.beam_search import BeamSearch

def func(state, idx):
    state_next = np.array([np.cos(idx), np.sin(idx)])
    dist = np.linalg.norm(state_next - state)
    score = dist
    return score, state_next

if __name__ == '__main__':
    state0 = np.zeros(2)
    state0[0] = np.random.random()
    state0[1] = np.random.random()

    # greedy
    search_algo = Greedy(func, state0, 100, 20)
    res_greedy = search_algo.search()
    print("Greedy:", res_greedy.score)
    print(res_greedy.seq)
    # beam search
    search_algo = BeamSearch(func, state0, 100, 20, 10)
    res_bs = search_algo.search()
    print("Beam search:", res_bs.score)
    print(res_bs.seq)

    # basic image
    t0 = np.linspace(0, 360) * np.pi / 180.
    x0 = np.cos(t0)
    y0 = np.sin(t0)

    x1 = [state0[0]]
    y1 = [state0[1]]
    for idx in res_greedy.seq:
        x1.append(np.cos(idx))
        y1.append(np.sin(idx))

    x2 = [state0[0]]
    y2 = [state0[1]]
    for idx in res_bs.seq:
        x2.append(np.cos(idx))
        y2.append(np.sin(idx))

    plt.figure()
    plt.plot(x0, y0, "--", color="k", label="Basic")
    plt.plot(x1, y1, label="Greedy")
    plt.plot(x2, y2, label="Beam search")
    plt.legend()
    plt.axis("equal")
    plt.grid("on")

    plt.show()