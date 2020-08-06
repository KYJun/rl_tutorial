#####################
## DQN project

## Editted by YoungJun Kim

## original code by 
# https://github.com/jaara/AI-blog/
# https://github.com/cmusjtuliuyuan/RainBow/

## SumTree structure for PER

## v1.0 (03/04/2018)
######################



from __future__ import absolute_import, division, print_function
import numpy as np

class SumTree:
    # write : how many have written (reset if exceed buffer), pointer for assigning data
    # count : total data length, represent total buffer size
    write = 0
    count = 0

    def __init__(self, capacity):
        # capacity: max size of the buffer
        # tree: sumtree for probability
        # data: actual data with index to search
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        # idx : idx for tree (child nodes)
        # change : how much to change in parent node
        # parent : parent node, sum of child nodes

        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            # repeat until reaches the top
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # from 0 node, search top-down, left-right for matching value under s

        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total_and_count(self):
        # self.tree[0]: total sum of p
        # self.count: how many data are stored
        return self.tree[0], self.count

    def add(self, p, data):
        # tree_idx = data_idx + total_length - 1
        idx = self.write + self.capacity - 1

        # assign data idx with data info
        self.data[self.write] = data

        # update tree and its parents node
        self.update(idx, p)

        # move write pointer (reset if exceed) and accumulate count
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):

        # how much to change from existing info
        change = p - self.tree[idx]

        # update tree with new p
        self.tree[idx] = p

        # change parent with the amount of change
        self._propagate(idx, change)

    def get(self, s):
        # retrieve where tree idx is below s (left-first)
        idx = self._retrieve(0, s)

        # search data idx
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



if __name__ == "__main__":
    s_tree = SumTree(4)
    #print(s_tree.tree)
    #print(s_tree.data)
    #print(s_tree.total_and_count())

    s_tree.add(1, 0)
    s_tree.add(0.6, 1)
    s_tree.add(0.4, 2)
    s_tree.add(0.5, 3)


    #print(s_tree.total_and_count())
    #print(s_tree.tree)
    #print(s_tree.data)