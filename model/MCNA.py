import torch
import math
import torch.nn as nn

class BasicBlock_1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock_1d, self).__init__()
        self.conv1_1d = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=(1, 3), stride=1 ,padding=(0, 1), bias=False)
        self.conv2_1d = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), stride=1 ,padding=(0, 1), bias=False)
        self.gelu  = nn.GELU()
    def forward(self, x):
        out = self.conv1_1d(x)
        out = self.gelu(out)
        out = self.conv2_1d(out)
        return out

class MakeLayer_1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MakeLayer_1d, self).__init__()
        self.BasicBlock_1d = BasicBlock_1d(in_channels, out_channels)
        self.gelu  = nn.GELU()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
    def forward(self, x):
        out = self.BasicBlock_1d(x) + self.shortcut(x)
        out = self.gelu(out)
        return out

class Block_1d(nn.Module):
    def __init__(self, scale):
        super(Block_1d, self).__init__()
        self.layer_1 = MakeLayer_1d(in_channels = 1, out_channels = 8)
        self.layer_3 = MakeLayer_1d(in_channels = 8, out_channels = 16)
        self.layer_5 = MakeLayer_1d(in_channels = 16, out_channels = 32)
        self.pooling = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.ScalePooling = self.ScalePoolingLayer(scale)
        self.GlobalAveragePooling = nn.AdaptiveAvgPool2d(1)
    def ScalePoolingLayer(self, Scale):
        layer = []
        for i in range(int(math.log(Scale, 2))):
            layer.append(self.pooling)
        return nn.Sequential(*layer) 
    def forward(self, x):
        out = self.ScalePooling(x)
        out = self.layer_1(out)
        out = self.pooling(out)
        out = self.layer_3(out)
        out = self.pooling(out)
        out = self.layer_5(out)
        out = self.GlobalAveragePooling(out)
        return out

class MutiScale_Block_S1(nn.Module):
    def __init__(self):
        super(MutiScale_Block_S1, self).__init__()
        self.Scale_1 = Block_1d(1)
    def forward(self, x):
        out = self.Scale_1(x).squeeze(2).squeeze(2).unsqueeze(1)
        return out

class Attention(nn.Module):
    def __init__(self, model_dim, dropout = 0.6):
        super(Attention, self).__init__()
        self.model_dim = model_dim
        self.linear_k =nn.Linear(model_dim, model_dim)
        self.linear_v =nn.Linear(model_dim, model_dim)
        self.linear_q =nn.Linear(model_dim, model_dim)
        self.linear_in = nn.Linear(model_dim, model_dim * 4)
        # self.linear_mid = nn.Linear(model_dim * 4, model_dim * 4)
        self.linear_out = nn.Linear(model_dim * 4, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.gelu  = nn.GELU()
    def forward(self, input):    
        ''' 自注意力模块 '''
        residual = input
        input = self.layer_norm(input)
        key   = self.linear_k(input)
        value = self.linear_v(input)
        query = self.linear_q(input)
        attention = torch.matmul(query, key.transpose(-1,-2)) / self.model_dim ** 0.5
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        output = residual + output # output = self.layer_norm(residual + output)

        ''' 前馈神经网络 '''
        residual = output
        output = self.layer_norm_1(output)
        output = self.linear_in(output)
        output = self.gelu(output)
        output = self.dropout(output)
        # output = self.linear_mid(output)
        # output = self.gelu(output)
        # output = self.dropout(output)
        output = self.linear_out(output)
        output = self.dropout(output)
        output = output + residual
        # output = self.layer_norm_1(output)
        return output

class MultiScaleCNN_S1_A1(nn.Module):
    def __init__(self, param, num_classes=2):
        super(MultiScaleCNN_S1_A1, self).__init__()
        self.param = param
        self.Block_1d_sensor_1 = MutiScale_Block_S1()
        self.Block_1d_sensor_2 = MutiScale_Block_S1()
        self.Block_1d_sensor_3 = MutiScale_Block_S1()
        self.Block_1d_sensor_4 = MutiScale_Block_S1()
        self.attention_1 = Attention(32)
        self.fc_1 = nn.Linear(4 * 32, num_classes)
    def forward(self, x):
        x = x.reshape(-1, 1, self.param.segment_len, self.param.segment_dim)
        out_1 = self.Block_1d_sensor_1(x[:, :, :, 0].unsqueeze(2))
        out_2 = self.Block_1d_sensor_2(x[:, :, :, 1].unsqueeze(2))
        out_3 = self.Block_1d_sensor_3(x[:, :, :, 2].unsqueeze(2))
        out_4 = self.Block_1d_sensor_4(x[:, :, :, 3].unsqueeze(2))
        out = torch.cat([out_1, out_2, out_3, out_4], 1)
        out = self.attention_1(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = out.reshape(-1, self.param.seq_len, 2)
        return out

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    print(get_parameter_number(MultiScaleCNN_S1_A1(num_classes=10)))



            

    
