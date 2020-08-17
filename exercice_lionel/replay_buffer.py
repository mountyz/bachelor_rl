import random
from operator import itemgetter
from collections import deque

import numpy as np

class ReplayBuffer:

    def __init__(self, capacity=int(1e4)):
        """Replay buffer, defined as a FIFO data strcture that
        contains (at most) the `size` most recent experiences.
        """
        self.capacity = capacity
        self._keys = ['ob', 'ac', 'rew', 'done', 'next_ob']
        self._data = {k: deque(maxlen=self.capacity) for k in self._keys}

    def store(self, transition):
        assert set(self._data.keys()) == set(transition.keys()), "non-matching keys"
        for key in self._keys:
            self._data[key].append(transition[key])

    def sample(self, batch_size):
        indexes = np.random.choice(np.arange(0, self.size), batch_size)
        return {'ob': np.array(list(self._data['ob']), dtype=np.float32)[indexes],
                'ac': np.array(list(self._data['ac']), dtype=np.longlong)[indexes],
                'rew': np.array(list(self._data['rew']), dtype=np.float32)[indexes],
                'next_ob': np.array(list(self._data['next_ob']), dtype=np.float32)[indexes],
                'done': np.array(list(self._data['done']), dtype=np.float32)[indexes]}

    @property
    def size(self):
        return len(self._data['ob'])  # arbitrarily picking `ob`

