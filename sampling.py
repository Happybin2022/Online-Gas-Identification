import pandas as pd
import numpy as np
from sklearn import preprocessing 

def sampling1(data="./dataset/ethylene_CO.csv"):
    f = pd.read_csv(data, header=None).values
    n_datas = []
    for data in f:
        if round(data[0], 2) * 100 % 100 == 0:
            n_datas.append(data)
    n_datas = pd.DataFrame(n_datas)
    n_datas.to_csv("./dataset/ethylene_CO_1Hz.csv", header=None, index=False)

def sampling2(data="./dataset/ethylene_methane.csv"):
    f = pd.read_csv(data, header=None).values
    n_datas = []
    for data in f:
        if round(data[0], 2) * 100 % 100 == 0:
            n_datas.append(data)
    n_datas = pd.DataFrame(n_datas)
    n_datas.to_csv("./dataset/ethylene_methane_1Hz.csv", header=None, index=False)

def processing(data="./dataset/ethylene_CO_1Hz.csv"):
    f = pd.read_csv(data, header=None).values
    n_datas = []
    for id, data in enumerate(f):
        data[0] = data[0] - 10
        if data[0] <= 10:
            pass
        else:
            n_datas.append(list(data))
    return n_datas

def split(datas):
    length = len(datas)
    datas = np.array(datas)
    ''' 浓度校正，延后12s '''
    delay = 12
    [a, b] = np.shape(datas)
    f = np.zeros((a, b))
    f[:, 3:] = datas[:, 3:]
    f[:(a-delay), :3] = datas[delay:, :3]
    datas = f

    ''' 分割训练、验证和测试集 '''
    train_datas = datas[:int(0.6*length), :]
    val_datas   = datas[int(0.6*length):int(0.8*length), :]
    test_datas  = datas[int(0.8*length):, :]
    
    ''' 归一化 '''
    Train_scaler, val_scaler, test_scaler = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()
    X_train_minmax = Train_scaler.fit_transform(train_datas[:, 1:])
    X_val_minmax   = val_scaler.fit_transform(val_datas[:, 1:])
    X_test_minmax  = test_scaler.fit_transform(test_datas[:, 1:])

    ''' 保存训练、验证和测试集 '''
    train, val, test = pd.DataFrame(X_train_minmax), pd.DataFrame(X_val_minmax), pd.DataFrame(X_test_minmax)
    train.to_csv("./dataset/ethylene_CO_1Hz_train.csv", header=None, index=False)
    val.to_csv("./dataset/ethylene_CO_1Hz_val.csv"    , header=None, index=False)
    test.to_csv("./dataset/ethylene_CO_1Hz_test.csv"  , header=None, index=False)

if __name__ == "__main__":
    # sampling1()
    # sampling2()
    n_datas = processing()
    split(n_datas)
