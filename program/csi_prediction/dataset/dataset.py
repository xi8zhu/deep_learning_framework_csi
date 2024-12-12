import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py
import numpy as np
class my_dataset(Dataset):
    def __init__(self, cfg):
        super(my_dataset, self).__init__()
        
        self.dataroot = cfg.dataroot
        # self.dataroot_x = cfg.dataroot_x
        # self.dataroot_y = cfg.dataroot_y
        data_mat_x = h5py.File(cfg.dataroot_x, 'r')
        data_mat_y = h5py.File(cfg.dataroot_y, 'r')

        self.data = data_mat_x['result_a']
        self.target = data_mat_y['result_21']

        # the last dimension of .mat is sample length
        x_length =  self.data.shape[-1]
        y_length =  self.target.shape[-1]
        if x_length == y_length:
            self.sample_length = x_length
        else:
            raise ValueError("the length of the data and target is different!")
        if not cfg.total_data:
            self.sample_length = int(self.sample_length / 10)

    def __len__(self):
        return self.sample_length

    def __getitem__(self, idx):
        sample_x = self.data[..., idx]
        sample_y = self.target[..., idx]
        data_x = torch.from_numpy(sample_x).float()
        data_y = torch.from_numpy(sample_y).float()
        data_y = data_y[..., 0]
        x = data_x.permute(4, 3, 2, 1, 0)
        y = data_y.permute(3, 2, 1, 0)
        return x, y