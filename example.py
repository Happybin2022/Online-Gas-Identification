import torch
import torch.nn as nn
import numpy as np
from utils.config import ModelParam
from utils.DataLoader import Loader, FastTensorDataLoader
import matplotlib.pyplot as plt

def Evonet_example():
    param = ModelParam()
    
    criterion = nn.MSELoss()  
    from model.Evonet import ClusterStateRecognition, Evonet_TSC
    GNN = ClusterStateRecognition(param)
    tes_prob, tes_patterns = GNN.predict("tes")
    tes_batch = tes_prob.size()[0]
    tes_patterns = torch.repeat_interleave(tes_patterns.unsqueeze(0), repeats=tes_batch, dim=0)
    DataLoader = Loader(param)
    _, _, _, _, tes_x, tes_y = DataLoader.fetch_data()
    tes_loader = FastTensorDataLoader(tes_x, tes_y, tes_prob, tes_patterns, batch_size=1000000, shuffle=False)
    net = Evonet_TSC(param).cuda()
    net.load_state_dict(torch.load("./model/Evonet_example.pth"))
    net.eval()

    predict, real = [], []
    with torch.no_grad():
        for (x_batch, y_batch, prob_batch, patterns) in tes_loader:
            outputs, _ = net(prob_batch, y_batch, patterns)
            # y_batch_ = torch.reshape(y_batch[:, 2:, :], (-1, param.n_event))
            y_batch_ = torch.reshape(y_batch[:, 1:-1, ], (-1, param.n_event))
            loss = criterion(outputs, y_batch_)
            outputs_ = torch.reshape(outputs, (-1, param.seq_len-1, param.n_event))
            predict.append(outputs_[:, -1, :])
            real.append(y_batch[:, -2, :])
    predict, real = torch.stack(predict).to("cpu"), torch.stack(real).to("cpu")
    predict, real = torch.reshape(predict, (-1, 2)), torch.reshape(real, (-1, 2))
    predict, real = predict.numpy(), real.numpy()
    length = len(predict)
    time = np.arange(0, length*param.segment_len, param.segment_len)[:, np.newaxis]
    
    plt.subplot(2, 1, 1)
    plt.plot(time ,predict[:, 0], c="wheat")
    plt.plot(time ,real[:, 0], c="orange")
    plt.title("Evonet: {:.4f}".format(loss.item()))
    plt.subplot(2, 1, 2)
    plt.plot(time ,predict[:, 1], c="lightblue")
    plt.plot(time ,real[:, 1], c="deepskyblue")
    plt.show()

def Clstm_example():
    param = ModelParam()
    DataLoader = Loader(param)
    _, _, _, _, tes_x, tes_y = DataLoader.fetch_data()
    tes_loader = FastTensorDataLoader(tes_x, tes_y, batch_size=1000000, shuffle=False)
    criterion = nn.MSELoss() 

    from model.CLSTM import CLSTM
    net = CLSTM().cuda()
    net.load_state_dict(torch.load("./model/Clstm_example.pth"))
    net.eval()

    predict, real = [], []
    with torch.no_grad():
        for (curves, labels) in tes_loader:
            outputs = net(curves)
            loss = criterion(outputs, labels[:, -1, :])
            predict.append(outputs)
            real.append(labels[:, -1, :])
    predict, real = torch.stack(predict).to("cpu"), torch.stack(real).to("cpu")
    predict, real = torch.reshape(predict, (-1, 2)), torch.reshape(real, (-1, 2))
    predict, real = predict.numpy(), real.numpy()
    length = len(predict)
    time = np.arange(0, length*param.segment_len, param.segment_len)[:, np.newaxis]
    
    plt.subplot(2, 1, 1)
    plt.plot(time ,predict[:, 0], c="wheat")
    plt.plot(time ,real[:, 0], c="orange")
    plt.title("CLstm: {:.4f}".format(loss.item()))
    plt.subplot(2, 1, 2)
    plt.plot(time ,predict[:, 1], c="lightblue")
    plt.plot(time ,real[:, 1], c="deepskyblue")
    plt.show()

def MCNA_example():
    param = ModelParam()
    DataLoader = Loader(param)
    _, _, _, _, tes_x, tes_y = DataLoader.fetch_data()
    tes_loader = FastTensorDataLoader(tes_x, tes_y, batch_size=1000000, shuffle=False)
    criterion = nn.MSELoss()

    from model.MCNA import MultiScaleCNN_S1_A1
    net = MultiScaleCNN_S1_A1(param, 2).cuda()
    net.load_state_dict(torch.load("./model/MCNA_example.pth"))
    net.eval()

    predict, real = [], []
    with torch.no_grad():
        for (curves, labels) in tes_loader:
            outputs = net(curves)
            loss = criterion(outputs, labels[:, :-1, :])
            predict.append(outputs)
            real.append(labels[:, :-1, :])
    predict, real = torch.stack(predict).to("cpu"), torch.stack(real).to("cpu")
    predict, real = torch.reshape(predict, (-1, 2)), torch.reshape(real, (-1, 2))
    predict, real = predict.numpy(), real.numpy()
    length = len(predict[::param.seq_len])
    time = np.arange(0, length*param.segment_len, param.segment_len)[:, np.newaxis]
    
    print(loss.item())
    plt.subplot(2, 1, 1)
    plt.plot(time ,predict[::param.seq_len, 0], c="wheat")
    plt.plot(time ,real[::param.seq_len, 0], c="orange")
    plt.title("MCNA: {:.4f}".format(loss.item()))
    plt.subplot(2, 1, 2)
    plt.plot(time ,predict[::param.seq_len, 1], c="lightblue")
    plt.plot(time ,real[::param.seq_len, 1], c="deepskyblue")
    
    plt.show()
    
if __name__ == "__main__":
    # Clstm_example()
    Evonet_example()
    # MCNA_example()
    pass