# mcts.py
# Monte Carlo Tree Search
# louisx1221@gmail.com
# 2023/10/13
# [A Survey of Monte Carlo Tree Search Methods | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/6145622/)

import numpy as np
import random

class Node():
    def __init__(self, state = None):
        self.seq = []
        self.seq_rew = []
        self.reward = 0.
        self.state = state
        self.terminal = False
        self.parent = None
        self.children = []
        self.child_idx = []
        self.quality = 0.
        self.times = 0

class MCTS():
    """
    Monte Carlo Tree Search (upper confidence bounds for tree, UCT)
    Minimizing the objective function
    func:               objective function      (score, state_next = func(state, idx))
    state0:             initial state
    node_num:           number of the nodes
    seq_len:            length of the sequence
    computation_budget: computational budget (iteration constraint)
    gamma:              discount factor
    """

    def __init__(
        self,
        func,
        state0,
        node_num = 10,
        seq_len = 5,
        computation_budget = 100,
        gamma = 0.9,
        epsilon = 0.1
    ):
        super().__init__()
        self.func = func
        self.state0 = state0
        self.state_len = len(state0)
        self.node_num = node_num
        self.seq_len = seq_len
        self.computation_budget = computation_budget
        self.gamma = gamma
        self.epsilon = epsilon

    def search(self):
        # Create root node v0 with state s0
        node = Node(self.state0)
        for _ in range(self.seq_len):
            node = self.uct_search(node)

        node.reward = -node.reward
        return node

    def uct_search(self, node):
        for _ in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(node)

            # 2. Random run to add node and get reward
            reward = self.default_policy(expand_node)

            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)

        # N. Get the best next node
        best_next_node = self.best_child(node, False)

        return best_next_node

    def tree_policy(self, node):
        # Check if the current node is the leaf node
        while not node.terminal:
            if len(node.seq) + len(node.children) < self.node_num:
                # Return the new sub node
                return self.expand(node)
            else:
                node = self.best_child(node, True)
                if len(node.seq) >= self.seq_len:
                    node.terminal =True

        # Return the leaf node
        return node

    def expand(self, node):
        # Choose a in untried actions from A(s(v))
        act = random.choice(range(self.node_num))
        while (act in node.seq) or (act in node.child_idx):
            act = random.choice(range(self.node_num))

        # Add a new child v' to v
        sub_node = Node()
        reward, sub_node.state = self.func(node.state, act)
        sub_node.seq = node.seq + [act]
        sub_node.seq_rew = node.seq_rew + [reward]
        sub_node.reward = node.reward - reward
        sub_node.parent = node
        node.children.append(sub_node)
        node.child_idx.append(act)

        return sub_node

    def best_child(self, node, exploration):
        # UCB: argmax_{v' in children of v} Q(v') / N(v') + c * sqrt(2 * ln(N(v)) / N(v'))
        # c =   sqrt(1/2),  if exploration
        #       0,          else

        # Use the min float value
        score_best = -np.inf

        # Travel all sub nodes to find the best one
        for sub_node in node.children:
            score = sub_node.quality / sub_node.times
            # Ignore exploration for inference
            if exploration:
                score += np.sqrt(np.log(node.times) / sub_node.times)

            if score > score_best:
                sub_node_best = sub_node
                score_best = score

        return sub_node_best

    def default_policy(self, node):
        # Get the state of the game
        current_state = node.state.copy()
        current_seq = node.seq.copy()
        rewards = node.reward

        gamma = self.gamma
        epsilon = self.epsilon
        # Run until the game over
        while len(current_seq) < self.seq_len:
            # Epsilon-greedy
            if random.random() < epsilon:
                # Pick one random action to play and get next state
                act = random.choice(range(self.node_num))
                while act in current_seq:
                    act = random.choice(range(self.node_num))
            else:
                # Greedy
                reward_best = np.inf
                act_best = 0
                for act in range(self.node_num):
                    if act in current_seq:
                        continue
                    reward, _ = self.func(current_state, act)
                    if reward < reward_best:
                        reward_best = reward
                        act_best = act
                act = act_best

            reward, current_state = self.func(current_state, act)
            current_seq.append(act)
            # Discount reward
            gamma *= self.gamma
            rewards += -reward * gamma

        return rewards

    def backup(self, node, reward):
        # Update util the root node
        while not node == None:
            # Update the visit times
            node.times += 1

            # Update the quality value
            node.quality += reward
            
            # Change the node to the parent node
            node = node.parent