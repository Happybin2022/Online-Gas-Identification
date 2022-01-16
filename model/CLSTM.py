import torch.nn as nn

class CLSTM(nn.Module):
    def __init__(self, feature_dim=32):
        super(CLSTM, self).__init__()
        self.feature_dim = feature_dim
        self.cnn = CNN(out_channels = feature_dim)
        self.lstm = Lstm(feature_dim, feature_dim * 2, 2)
    def forward(self, x):
        [batch_size, seq, segment_len, segment_dim] = x.size()
        out = x.reshape(-1, segment_len, segment_dim).unsqueeze(1)
        out = self.cnn(out)
        out = out.squeeze(2).squeeze(2).reshape(batch_size, seq, self.feature_dim)
        out = self.lstm(out)
        return out

class CNN(nn.Module):
    def __init__(self, out_channels):
        super(CNN, self).__init__()
        self.layer_1 = BasicBlock(in_channels = 1, out_channels = int(out_channels/4))
        # self.layer_2 = BasicBlock(in_channels = int(out_channels/4), out_channels = int(out_channels/4))
        self.layer_3 = BasicBlock(in_channels = int(out_channels/4), out_channels = int(out_channels/2))
        # self.layer_4 = BasicBlock(in_channels = int(out_channels/2), out_channels = int(out_channels/2))
        self.layer_5 = BasicBlock(in_channels = int(out_channels/2), out_channels = int(out_channels/1))
        # self.layer_6 = BasicBlock(in_channels = int(out_channels/1), out_channels = int(out_channels/1))
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Global_average_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        out = self.layer_1(x)
        # out = self.layer_2(out)
        # out = self.pooling(out)
        out = self.layer_3(out)
        # out = self.layer_4(out)
        # out = self.pooling(out)
        out = self.layer_5(out)
        # out = self.layer_6(out)
        out = self.Global_average_pooling(out)
        return out 

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d( in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.relu  = nn.ReLU()
    def forward(self, x):
        if self.in_channels == self.out_channels:
            resident = x
        else:
            resident = self.conv3(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out + resident)
        return out

class Lstm(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):
        super(Lstm, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, dropout=0.2, batch_first=True)
        self.fc_1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc_2 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc_3 = nn.Linear(int(hidden_dim/4), 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc_1(out[:, -1, :])
        out = self.dropout(out)
        # out = self.relu(out)
        out = self.fc_2(out)
        out = self.dropout(out)
        # out = self.relu(out)
        out = self.fc_3(out)
        return out 

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    net = CLSTM()
    print(net)
    print(get_parameter_number(net))



