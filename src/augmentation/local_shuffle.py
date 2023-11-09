import numpy as np
import torch


class LocalShuffleAug:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.prob = 0.5

    def __call__(
        self, seq: np.ndarray
    ) -> np.ndarray:        
        seq_window_size = seq.shape[1]
        for i in range(seq_window_size // self.window_size):
            if np.random.rand() < self.prob:
                np.random.shuffle(seq[i*self.window_size:(i+1)*self.window_size])
        return seq