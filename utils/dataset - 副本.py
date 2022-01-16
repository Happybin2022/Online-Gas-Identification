import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class SelfDataset(Dataset):
    def __init__(self, xlsxfile, stride=10):
        self.file = pd.read_csv(xlsxfile, header=None).values
        self.stride = stride
    def __len__(self):
        return int(len(self.file) / self.stride)
    def __getitem__(self, idx):
        x_data  = torch.Tensor(self.file[idx*self.stride: (idx+1)*self.stride, [2, 4, 6, 8]], device="cuda0")
        y_data = torch.Tensor(self.file[idx*self.stride: (idx+1)*self.stride, [0, 1]], device="cuda0")
        return x_data, y_data

def SelfDataset_1(xlsxfile, stride=10):
    datas = pd.read_csv(xlsxfile, header=None).values
    stride = stride
    (time, sensor) = np.shape(datas)
    length = int(time / stride)
    x_data = np.zeros([length, stride, 4])
    y_data = np.zeros([length, stride, 2])
    for idx in range(length):
        x_data[idx, :, :]  = datas[idx*stride: (idx+1)*stride, [2, 4, 6, 8]]
        y_data[idx, :, :] = datas[idx*stride: (idx+1)*stride, [0, 1]]
    x_data = torch.tensor(x_data, dtype=torch.float32, device="cuda:0")
    y_data = torch.tensor(y_data, dtype=torch.float32, device="cuda:0")
    return x_data, y_data