import torch, os
from torch.utils.data import DataLoader

def Test_with_data(model, path, dataset):
    

if __name__ == "__main__":
    dataset_list = []
    net_path = "E:/TrainInformation/Train_CNN_1d_Description_60_1_val_49/CNN_1d_100.pth"
    for i in os.listdir("E:\Gas_sensor\Description"):                                   
        dataset_list.append(i)    

    for dataset in dataset_list:
        print("dataset: {} | Accuarry: {:.2f}%".format(dataset, Test_with_data("CNN_1d", net_path, dataset)))


