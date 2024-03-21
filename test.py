# test.py

import numpy as np
import matplotlib.pyplot as plt

from src.greedy import Greedy
from src.beam_search import BeamSearch
from src.mcts import MCTS
from src.traversal import Traversal

def func(state, idx):
    state_next = np.array([np.cos(idx**2), np.sin(idx)])
    dist = np.linalg.norm(state_next - state)
    score = dist
    return score, state_next

if __name__ == '__main__':
    state0 = np.zeros(2)
    state0[0] = np.random.random()
    state0[1] = np.random.random()

    num_total = 8
    num_seq = 8

    # Greedy
    search_algo = Greedy(func, state0, num_total, num_seq)
    res_greedy = search_algo.search()
    print("Greedy:", res_greedy.reward, res_greedy.time)
    print(res_greedy.seq)
    # Beam search
    search_algo = BeamSearch(func, state0, num_total, num_seq, 10)
    res_bs = search_algo.search()
    print("Beam search:", res_bs.reward, res_bs.time)
    print(res_bs.seq)
    # Monte Carlo Tree Search
    search_algo = MCTS(func, state0, num_total, num_seq, 1000, gamma=0.9, epsilon=0.1)
    res_mcts = search_algo.search()
    print("MCTS:", res_mcts.reward, res_mcts.time)
    print(res_mcts.seq)
    Traversal
    search_algo = Traversal(func, state0, num_total, num_seq)
    res_trav = search_algo.search()
    print("Traversal:", res_trav.reward, res_trav.time)
    print(res_trav.seq)

    # basic image
    t0 = np.linspace(0, 360) * np.pi / 180.
    x0 = np.cos(t0)
    y0 = np.sin(t0)

    x1 = [state0[0]]
    y1 = [state0[1]]
    for idx in res_greedy.seq:
        x1.append(np.cos(idx**2))
        y1.append(np.sin(idx))

    x2 = [state0[0]]
    y2 = [state0[1]]
    for idx in res_bs.seq:
        x2.append(np.cos(idx**2))
        y2.append(np.sin(idx))

    x3 = [state0[0]]
    y3 = [state0[1]]
    for idx in res_mcts.seq:
        x3.append(np.cos(idx**2))
        y3.append(np.sin(idx))

    x4 = [state0[0]]
    y4 = [state0[1]]
    for idx in res_trav.seq:
        x4.append(np.cos(idx**2))
        y4.append(np.sin(idx))

    plt.figure()
    plt.plot(x0, y0, "--", color="k", label="Basic")
    plt.plot(x1, y1, label="Greedy")
    plt.plot(x2, y2, label="Beam search")
    plt.plot(x3, y3, label="MCTS")
    plt.plot(x4, y4, label="Traversal")
    plt.legend()
    plt.axis("equal")
    plt.grid("on")

    plt.show()