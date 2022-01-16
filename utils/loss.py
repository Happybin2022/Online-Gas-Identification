import torch.nn as nn
import torch


class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()
        
    def forward(self, y_pre, y_tru):
        return torch.mean(torch.pow((y_pre - y_tru), 2))
