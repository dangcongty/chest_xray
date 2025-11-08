import numpy as np
from torch.utils.tensorboard import SummaryWriter


class CtTracker:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.reset()

    def reset(self, mode=None):
        keys = ['global_total', 'local']
        if mode in [None, 'train']:
            self.logs_train = {k: [] for k in keys}
        if mode in [None, 'val']:
            self.logs_val = {k: [] for k in keys}

    def append(self, gl, ll, mode='train'):
        logs = self.logs_train if mode == 'train' else self.logs_val
        logs['global_total'].append(gl.item())
        logs['local'].append(ll.item())

    def add_scalar(self, e, mode='train'):
        logs = self.logs_train if mode == 'train' else self.logs_val
        for k, v in logs.items():
            self.writer.add_scalar(f'{mode}/{k}', np.mean(v), e)
        self.reset(mode)
