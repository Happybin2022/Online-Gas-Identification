import torch
import time
import numpy as np

### 单次网络训练
def train(net, train_loader, optimizer, criterion):
    net.train()
    sum_loss=0.
    i = 0
    for (curves, labels) in train_loader:
        optimizer.zero_grad()
        outputs = net(curves)
        loss = criterion(outputs, labels[:, :-1, :])
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        i = i + 1
    loss = sum_loss / i
    # print("train_accuracy = %f%%  loss = %f" %(correct.cpu().numpy()/total*100, loss))
    return loss      

def val(net, val_loader, criterion):
    net.eval()
    sum_loss=0.
    i = 0
    with torch.no_grad():
        for (curves, labels) in val_loader:
            outputs = net(curves)
            loss = criterion(outputs, labels[:, :-1, :])
            sum_loss += loss.item()
            i = i + 1
        # print("test_accuracy = %f%%" % (float(correct.cpu().numpy()) / total *100))
        loss = sum_loss / i
        return loss


### 单次网络测试
def test(net, test_loader, criterion):
    net.eval()
    sum_loss=0.
    i = 0
    with torch.no_grad():
        for (curves, labels) in test_loader:
            outputs = net(curves)
            loss = criterion(outputs, labels[:, :-1, :])
            sum_loss += loss.item()
            i = i + 1
        # print("test_accuracy = %f%%" % (float(correct.cpu().numpy()) / total *100))
        loss = sum_loss / i
        return loss

