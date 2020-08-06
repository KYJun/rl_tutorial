#####################
## DQN project

## Editted by YoungJun Kim

## original code by 
# https://github.com/jaara/AI-blog/
# https://github.com/cmusjtuliuyuan/RainBow/

## Simple Experience Replay & Prioritized Experience Replay for mini-batch training

## v1.0 (03/04/2018)
######################

import numpy as np
import random
from sumtree import SumTree

class ReplayMemory:
    """Store and replay (sample) memories."""
    def __init__(self,
                max_size):
        """Setup memory.
        You should specify the maximum size o the memory. Once the
        memory fills up oldest values are removed.
        """
        self._max_size = max_size
        
        # set default memory
        self._memory = []


    def append(self, state, action, reward, next_state, done):
        """Add a list of samples to the replay memory."""
        num_sample = len(state)

        # if size exceeds buffer length, overwrite memor
        if len(self._memory) >= self._max_size:
            del(self._memory[0:num_sample])

        # add to the buffer
        for s, a, r, n_s, d in zip(state, action, reward, next_state, done):
            self._memory.append((s, a, r, n_s, d))


    def sample(self, batch_size, indexes=None):
        """Return samples from the memory.
        Returns
        --------
        (old_state_list, action_list, reward_list, new_state_list, is_terminal_list, frequency_list)
        """
        # sample from random distribution
        samples = random.sample(self._memory, min(batch_size, len(self._memory)))

        # zip samples
        zipped = list(zip(*samples))
        
        return zipped


class PriorityExperienceReplay:
    '''
    Almost copy from
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    '''
    def __init__(self,
                max_size,
                window_size,
                input_shape):

        # set default sumtree
        self.tree = SumTree(max_size)
        self._max_size = max_size

        # dimension for how to store state and next state 
        self._window_size = window_size
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]

        # hyperparmeters for priority probability
        self.e = 0.01
        self.a = 0.6

    def _getPriority(self, error):

        # set probability for given experience
        return (error + self.e) ** self.a

    def append(self, state, action, reward, next_state, done):
        
        # add experience to tree with probability computed
        for s, a, r, n_s, d in zip(state, action, reward, next_state, done):
            
            # when first appended, set maximum priority
            # 0.5 is the maximum error
            p = self._getPriority(0.5)
            
            self.tree.add(p, data=(s, a, r, n_s, d))

    def sample(self, batch_size, indexes=None):

        # set batch for data, index and priority
        data_batch = []
        idx_batch = []
        p_batch = []

        # split the tree into batch size
        segment = self.tree.total_and_count()[0] / batch_size

        # search for high priority
        # divide into multiple section in tree to search for diverse, yet high priority sampels
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)

        zipped = list(zip(*data_batch))
        zipped[0] = np.reshape(zipped[0], (-1, self._WIDTH, self._HEIGHT, self._window_size))
        zipped[3] = np.reshape(zipped[3], (-1, self._WIDTH, self._HEIGHT, self._window_size))

        sum_p, count = self.tree.total_and_count()
        return zipped, idx_batch, p_batch, sum_p, count

    def update(self, idx_list, error_list):

        # update priority according to td error from current network
        # repeat after every training step
        for idx, error in zip(idx_list, error_list):
            p = self._getPriority(error)
            self.tree.update(idx, p)

if __name__ == "__main__":
    memory = PriorityExperienceReplay(max_size=4, window_size=4, input_shape=[1,1])
    state = [[1, 2, 3, 4], [3, 2, 1, 5], [3, 4, 3, 4], [2, 3, 1, 4]]
    action = [1, 2, 3, 4]
    reward = [1, -2, -1, 2]
    next_state = [[1, 2, 4, 3], [3, 2, 5, 1], [3, 4, 4, 3], [2, 3, 4, 1]]
    done = [0, 0, 0, 1]
    memory.append(state, action, reward, next_state, done)
    batch, idx_list, p_batch, _, _ = memory.sample(2)
    memory.update(idx_list, [0.1, 0.2])
    batch, idx_list, p_batch, _, _ = memory.sample(2)
    print(batch, idx_list, p_batch)
    #print(np.array(memory.sample(2)[1]).shape)
    #print(memory.sample(2)[0][0].shape)
    #print(memory.sample(2)[1][0])
    #print(memory.sample(2)[2][0])
    #print(memory.sample(2)[3][0])

