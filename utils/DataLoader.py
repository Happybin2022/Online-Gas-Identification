import torch
import pandas as pd
import numpy as np

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

class Loader():
    def __init__(self, param):
        self.param = param
        self.raw_data_array = {'data': None}
        self.x_tra, self.y_tra = None, None
        self.x_val, self.y_val = None, None
        self.x_tes, self.y_tes = None, None

    def __get_local_data__(self):
        ''' Fetch raw local data '''
        self.raw_data_array['data'] = pd.read_csv(self.param.xlsxfile, header=None).values[: , 1:]

    def __split_on_time__(self, start, end):
        tempx, tempy = [], []
        for i in range(start, end - 1):
            tempx.append(self.x[i - self.param.seq_len:i, :, :])
            tempy.append(self.y[i - self.param.seq_len:i+1, int(self.param.segment_len/2), :])
        return tempx, tempy

    def __normalize__(self, data):
        data_sens = (data[:, 2:] - np.mean(data[:, 2: ], axis=0)) / (np.std(data[:, 2: ], axis=0) + 1e-20)
        data_conv = (data[:, 0:2] - np.min(data[:, 0:2], axis=0)) / (np.max(data[:, 0:2], axis=0) - np.min(data[:, 0:2], axis=0)+ 1e-20)
        data_ = np.column_stack((data_conv, data_sens))
        return data_

    def __clean_data_by_entities__(self):
        ''' process the raw data '''
        x = self.raw_data_array['data']
        # x = x[10:, :]

        ''' 浓度校正，延后12s '''
        # delay = 12
        [a, b] = np.shape(x)
        # print("Total Time length: %d" %(a))
        # x_ = np.zeros((a, b))
        # x_[:, 2:] = x[:, 2:]
        # x_[:(a-delay), :2] = x[delay:, :2]

        ''' 规范化 '''
        if self.param.norm:
            x = self.__normalize__(x)


        ''' 加窗并选择数据 '''
        length = a // self.param.segment_len
        x = x[:length * self.param.segment_len, [0, 1, 2, 4, 6, 8]]
        self.y = np.reshape(x[:, 0:2], (length, self.param.segment_len, self.param.y_dim))
        self.x = np.reshape(x[:, 2: ], (length, self.param.segment_len, self.param.segment_dim))

        ''' 分割数据集 '''
        self.x_tra, self.y_tra = self.__split_on_time__(self.param.seq_len, int(self.param.seq_len + (length - self.param.seq_len) * 0.6))
        self.x_val, self.y_val = self.__split_on_time__(int(self.param.seq_len + (length - self.param.seq_len) * 0.6), int(self.param.seq_len + (length - self.param.seq_len) * 0.8))
        self.x_tes, self.y_tes = self.__split_on_time__(int(self.param.seq_len + (length - self.param.seq_len) * 0.8), int(self.param.seq_len + (length - self.param.seq_len) * 1.0))
        print("Train      length: %d" %(len(self.x_tra)))
        print("Validation length: %d" %(len(self.x_val)))
        print("Test       length: %d" %(len(self.x_tes)))
    
    def fetch_data(self):
        self.__get_local_data__()
        self.__clean_data_by_entities__()
        self.x_tra = torch.tensor(self.x_tra, dtype=torch.float32, device="cuda:0")
        self.y_tra = torch.tensor(self.y_tra, dtype=torch.float32, device="cuda:0")
        self.x_val = torch.tensor(self.x_val, dtype=torch.float32, device="cuda:0")
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32, device="cuda:0")
        self.x_tes = torch.tensor(self.x_tes, dtype=torch.float32, device="cuda:0")
        self.y_tes = torch.tensor(self.y_tes, dtype=torch.float32, device="cuda:0")

        return self.x_tra, self.y_tra, self.x_val, self.y_val, self.x_tes, self.y_tes

if __name__ == "__main__":
    pass
