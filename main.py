import torch, time
import torch.nn as nn
from utils.DataLoader import Loader, FastTensorDataLoader
from utils.config import ModelParam

def Evonet_example():
    # 输入常数
    param = ModelParam()

    from model.Evonet import ClusterStateRecognition, Evonet_TSC
    # 获取单个图表示
    GNN = ClusterStateRecognition(param)
    tra_prob, tra_patterns = GNN.predict("tra")
    val_prob, val_patterns = GNN.predict("val")
    tes_prob, tes_patterns = GNN.predict("tes")
    tra_batch = tra_prob.size()[0]
    val_batch = val_prob.size()[0]
    tes_batch = tes_prob.size()[0]
    tra_patterns = torch.repeat_interleave(tra_patterns.unsqueeze(0), repeats=tra_batch, dim=0)
    val_patterns = torch.repeat_interleave(val_patterns.unsqueeze(0), repeats=val_batch, dim=0)
    tes_patterns = torch.repeat_interleave(tes_patterns.unsqueeze(0), repeats=tes_batch, dim=0)
    print(tra_prob.shape, tra_patterns.shape, val_prob.shape, val_patterns.shape)
    
    # 载入训练集和测试集
    DataLoader = Loader(param)
    tra_x, tra_y, val_x, val_y, tes_x, tes_y = DataLoader.fetch_data()
    tra_loader = FastTensorDataLoader(tra_x, tra_y, tra_prob, tra_patterns, batch_size=param.batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(val_x, val_y, val_prob, val_patterns, batch_size=1000000, shuffle=False)
    tes_loader = FastTensorDataLoader(tes_x, tes_y, tes_prob, tes_patterns, batch_size=1000000, shuffle=False)
    
    net = Evonet_TSC(param).cuda()
    
    # 定义网络损失函数，使用交叉熵损失函数
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.235, 1.235, 1.235, 1.235, 0.06]).cuda())
    criterion = nn.MSELoss()
    # 定义网络的优化器，使用SGD优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=param.learning_rate)
    # optimizer = torch.optim.SGD(net.parameters(), lr=param.learning_rate, momentum=0.9)
    # 自动更新学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, 
                                                           threshold=10e-4, threshold_mode='rel',  min_lr=0.000001)

    from utils.train_test_evonet import train, val, test
    for epoch in range(1, param.TrainTimes+1):
        loss_tra = train(net, tra_loader, optimizer, criterion, param)
        loss_val= val(net, val_loader, criterion, param)
        # scheduler.step(acc_1)
        print("Epoch: %d |Train loss: %.4f Validation loss: %.4f" %(epoch, loss_tra, loss_val) )

        ''' 统计训练的结果 '''
        # if epoch % 10 == 0:
        # print("Epoch: %d |Train loss: %.5f Learn rate: %.2f, Val loss: %.5f" %(epoch, loss_tra, optimizer.param_groups[0]['lr'], loss_val) )
        # if loss_tra < loss_min:
            # loss_min = loss_tra
    path = "./model/Evonet.pth"
    torch.save(net.state_dict(), path)
    loss_tes = test(net, tes_loader, criterion, param)
    print("Test loss: %.6f" %(loss_tes))
    # return Epoch_plot, train_acc, val_acc, loss_tes

def Clstm_example():
    # 输入常数
    param = ModelParam()

    # 载入训练集和测试集
    DataLoader = Loader(param)
    tra_x, tra_y, val_x, val_y, tes_x, tes_y = DataLoader.fetch_data()
    tra_loader = FastTensorDataLoader(tra_x, tra_y, batch_size=param.batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(val_x, val_y, batch_size=1000000, shuffle=False)
    tes_loader = FastTensorDataLoader(tes_x, tes_y, batch_size=1000000, shuffle=False)

    from model.CLSTM import CLSTM 
    net = CLSTM(32).cuda()
    
    # 定义网络损失函数，使用交叉熵损失函数
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.235, 1.235, 1.235, 1.235, 0.06]).cuda())
    criterion = nn.MSELoss()
    # 定义网络的优化器，使用SGD优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=param.learning_rate)
    # optimizer = torch.optim.SGD(net.parameters(), lr=param.learning_rate, momentum=0.9)
    # 自动更新学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, 
                                                           threshold=10e-4, threshold_mode='rel',  min_lr=0.000001)
    from utils.train_test_clstm import train, val, test
    for epoch in range(1, param.TrainTimes+1):
        loss_tra = train(net, tra_loader, optimizer, criterion)
        loss_val= val(net, val_loader, criterion)
        # scheduler.step(acc_1)
        print("Epoch: %d |Train loss: %.4f Validation loss: %.4f" %(epoch, loss_tra, loss_val) )

        ''' 统计训练的结果 '''
        # if epoch % 10 == 0:
        # print("Epoch: %d |Train loss: %.5f Learn rate: %.2f, Val loss: %.5f" %(epoch, loss_tra, optimizer.param_groups[0]['lr'], loss_val) )
        # if loss_tra < loss_min:
            # loss_min = loss_tra
    path = "./model/Clstm_example.pth"
    torch.save(net.state_dict(), path)
    loss_tes = test(net, tes_loader, criterion)
    print("Test loss: %.4f" %(loss_tes))
    # return Epoch_plot, train_acc, val_acc, loss_tes

def MCNA_example():
    # 输入常数
    param = ModelParam()

    # 载入训练集和测试集
    DataLoader = Loader(param)
    tra_x, tra_y, val_x, val_y, tes_x, tes_y = DataLoader.fetch_data()
    tra_loader = FastTensorDataLoader(tra_x, tra_y, batch_size=param.batch_size, shuffle=True)
    val_loader = FastTensorDataLoader(val_x, val_y, batch_size=1000000, shuffle=False)
    tes_loader = FastTensorDataLoader(tes_x, tes_y, batch_size=1000000, shuffle=False)    
    
    from model.MCNA import MultiScaleCNN_S1_A1
    net = MultiScaleCNN_S1_A1(param, 2).cuda()
    
    # 定义网络损失函数，使用交叉熵损失函数
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.235, 1.235, 1.235, 1.235, 0.06]).cuda())
    criterion = nn.MSELoss()
    # 定义网络的优化器，使用SGD优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=param.learning_rate)
    # optimizer = torch.optim.SGD(net.parameters(), lr=param.learning_rate, momentum=0.9)
    # 自动更新学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, 
                                                           threshold=10e-4, threshold_mode='rel',  min_lr=0.000001)
    from utils.train_test_mcna import train, val, test
    for epoch in range(1, param.TrainTimes+1):
        loss_tra = train(net, tra_loader, optimizer, criterion)
        loss_val= val(net, val_loader, criterion)
        # scheduler.step(acc_1)
        print("Epoch: %d |Train loss: %.4f Validation loss: %.4f" %(epoch, loss_tra, loss_val) )

        ''' 统计训练的结果 '''
        # if epoch % 10 == 0:
        # print("Epoch: %d |Train loss: %.5f Learn rate: %.2f, Val loss: %.5f" %(epoch, loss_tra, optimizer.param_groups[0]['lr'], loss_val) )
        # if loss_tra < loss_min:
            # loss_min = loss_tra
    path = "./model/MCNA_example.pth"
    torch.save(net.state_dict(), path)
    loss_tes = test(net, tes_loader, criterion)
    print("Test loss: %.4f" %(loss_tes))
    # return Epoch_plot, train_acc, val_acc, loss_tes

if __name__ == "__main__":
    begin = time.time()
    Evonet_example()
    # Clstm_example()
    # MCNA_example()
     
    print("time: %.2f" %(time.time()-begin))
