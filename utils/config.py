import os

class ModelParam(object):
    # basic
    model_save_path = "./model"

    # dataset
    xlsxfile = './dataset/ethylene_CO_1Hz_smooth.csv'
    seq_len = 10
    segment_len = 5
    segment_dim = 4
    y_dim = 2
    norm = True
    
    # state recognition
    n_state = 30
    n_event = 2
    node_dim = 2 * segment_len * segment_dim
    graph_dim = 256
    covariance_type = 'diag'
    
    # model
    TrainTimes    = 100
    learning_rate = 0.001
    batch_size    = 500