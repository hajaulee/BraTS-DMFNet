import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import pkload
from .rand import *
from .transforms import *


class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', for_train=False, transforms=''):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path + 'data_f32.pkl')
        # print(x.shape, y.shape)#(240, 240, 155, 4) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print(x.shape, y.shape)  # (240, 240, 155, 4) (240, 240, 155)
        return x, y

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
