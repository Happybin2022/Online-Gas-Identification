import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os, joblib, torch
from sklearn.mixture import GaussianMixture

class ClusterStateRecognition():
    def __init__(self, param):
        self.param = param
        self.model_obj = GaussianMixture(n_components=param.n_state, covariance_type=param.covariance_type)
        raw_data = pd.read_csv(param.xlsxfile, header=None).values[: , 1:]
        self.length, _ = np.shape(raw_data)
        self.__clean_data_by_entities__(raw_data)
        print(np.shape(self.x_fit))
        self.fit(np.array(self.x_fit))

    def __normalize__(self, data):
        data_sens = (data[:, 2:] - np.mean(data[:, 2: ], axis=0)) / (np.std(data[:, 2: ], axis=0) + 1e-20)
        data_conv = (data[:, 0:2] - np.min(data[:, 0:2], axis=0)) / (np.max(data[:, 0:2], axis=0) - np.min(data[:, 0:2], axis=0)+ 1e-20)
        data_ = np.column_stack((data_conv, data_sens))
        return data_

    def __clean_data_by_entities__(self, x):
        [a, b] = np.shape(x)

        if self.param.norm:
            x = self.__normalize__(x)

        length = a // self.param.segment_len
        x = x[:length * self.param.segment_len, [0, 1, 2, 4, 6, 8]]
        self.y = np.reshape(x[:, 0:2], (length, self.param.segment_len, self.param.y_dim))
        self.x = np.reshape(x[:, 2: ], (length, self.param.segment_len, self.param.segment_dim))
        self.x_fit, self.y_fit = self.__split_on_time__(self.param.seq_len, int(self.param.seq_len + (length - self.param.seq_len) * 0.6)) # length)
        self.x_tra, self.y_tra = self.__split_on_time__(self.param.seq_len, int(self.param.seq_len + (length - self.param.seq_len) * 0.6))
        self.x_val, self.y_val = self.__split_on_time__(int(self.param.seq_len + (length - self.param.seq_len) * 0.6), int(self.param.seq_len + (length - self.param.seq_len) * 0.8))
        self.x_tes, self.y_tes = self.__split_on_time__(int(self.param.seq_len + (length - self.param.seq_len) * 0.8), int(self.param.seq_len + (length - self.param.seq_len) * 1.0))

    def __split_on_time__(self, start, end):
        tempx, tempy = [], []
        for i in range(start, end - 1):
            tempx.append(self.x[i - self.param.seq_len:i, :, :])
            tempy.append(self.y[i - self.param.seq_len:i, int(self.param.segment_len/2), :])
        return tempx, tempy

    def fit(self, ts):
        _, _, segment_len, segment_dim = ts.shape
        ts_ = np.reshape(ts, [-1, segment_len, segment_dim])
        ts_ = self.__getfeatures__(ts_)
        try:
            self.model_obj.fit(ts_)
            self.store(self.param.model_save_path)
        except Exception as e:
            raise e

    def predict(self, part):
        if part == "tra":
            x = np.array(self.x_tra)
        elif part == "val":
            x = np.array(self.x_val)
        else:
            x = np.array(self.x_tes)
        
        ts = np.reshape(x, [-1, x.shape[-2], x.shape[-1]])
        ts = self.__getfeatures__(ts)
        self.restore(self.param.model_save_path)
        tprob = self.model_obj.predict_proba(ts)
        tpatterns = np.concatenate([self.model_obj.means_, self.model_obj.covariances_], axis=1)
        xprob = np.reshape(tprob, (-1, x.shape[1], self.param.n_state))
        
        tpatterns = torch.tensor(tpatterns, dtype=torch.float32, device="cuda:0")
        xprob = torch.tensor(xprob, dtype=torch.float32, device="cuda:0")
        return xprob, tpatterns

    def __getfeatures__(self, x):
        segment_len = x.shape[1]    # [segment size * segment len * metric dim]
        segment_dim = x.shape[2]

        # means = np.reshape(np.mean(x, axis=1), [-1, segment_dim])
        # stds = np.reshape(np.std(x, axis=1), [-1, segment_dim])
        # features = np.concatenate([means, stds], axis=1)

        features = np.reshape(x, [-1, segment_len * segment_dim])

        return features

    def store(self, path, **kwargs):
        save_model_name = "gmm_{}.state_model".format(self.param.n_state)
        joblib.dump(self.model_obj, os.path.join(path, save_model_name))

    def restore(self, path, **kwargs):
        save_model_name = "gmm_{}.state_model".format(self.param.n_state)
        self.model_obj = joblib.load(os.path.join(path, save_model_name))

class Evonet_TSC(nn.Module):
    def __init__(self, param):
        super(Evonet_TSC, self).__init__()
        self.model = Evonet(param)
        self.param = param

        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(self.param.graph_dim, 512)
        self.fc2 = nn.Linear(512, self.param.n_event)

    def forward(self, a, y, p):
        graph_logits, node_logits, attention_logits = self.model.get_embedding(a, y, p)
        attention_score = F.softmax(attention_logits, dim=-1)

        # output
        patterns = torch.reshape(graph_logits, (-1, self.param.graph_dim))
        out_logits = self.fc1(patterns)
        out_logits = self.activation(out_logits)
        out_logits = self.fc2(out_logits)
        out_logits = self.activation(out_logits)
        return out_logits, attention_score

class Evonet(nn.Module):
    def __init__(self, param):
        super(Evonet, self).__init__()
        self.timesteps = param.seq_len
        self.n_event = param.n_event
        self.n_nodes = param.n_state
        self.n_features = param.node_dim
        self.graph_dim = param.graph_dim
        
        self.activation = nn.Tanh()
        self.Hin = nn.Linear(self.n_features, self.n_features)
        self.Hout = nn.Linear(self.n_features, self.n_features)
        self.A = nn.Linear(self.n_features+self.graph_dim, 1, bias=False)
        self.node_lstm = Lstm_unit(self.n_features+self.graph_dim, self.n_features, self.n_features)
        self.graph_lstm = Lstm_unit(self.n_features+self.n_event, self.graph_dim, self.graph_dim)
    
    def get_embedding(self, state_sequence, event_sequence, initial_node_embedding):
        # 输入：self.a, y_clf, self.p
        self.batch_size = state_sequence.size()[0]

        # time major
        state_sequence = torch.transpose(state_sequence, 0, 1)  # [seq_length * batch_size * n_features]
        event_sequence = torch.transpose(event_sequence, 0, 1)  # [seq_len+1 * batch_size * n_event]

        # inital
        cur_node_emb = torch.ones(self.batch_size, self.n_nodes, self.n_features, device="cuda:0") * initial_node_embedding
        cur_node_mem = torch.ones(self.batch_size, self.n_nodes, self.n_features, device="cuda:0") * initial_node_embedding
        cur_graph_emb = torch.zeros(self.batch_size, self.graph_dim, device="cuda:0")
        cur_graph_mem = torch.zeros(self.batch_size, self.graph_dim, device="cuda:0")

        graph_embs, node_embs, attention_logits = [], [], []

        for i in range(self.timesteps-1):
            cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a = self.Cell(state_sequence[i], state_sequence[i+1], event_sequence[i+1], cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem)
            graph_embs.append(cur_graph_emb)
            node_embs.append(cur_node_emb)
            attention_logits.append(cur_a)
        graph_embs = torch.stack(graph_embs)
        node_embs = torch.stack(node_embs)
        attention_logits = torch.stack(attention_logits)

        # transpose [batch_size, timestep, *]
        graph_embs = torch.transpose(graph_embs, 0, 1)
        node_embs = torch.transpose(node_embs, 0, 1)
        attention_logits = torch.reshape(torch.transpose(attention_logits, 0, 1), (-1, self.timesteps-1))

        return graph_embs, node_embs, attention_logits

    def Cell(self, send_nodes, receive_nodes, event, prev_graph_emb, prev_graph_mem, prev_node_emb, prev_node_mem):
        
        # intermediate node representation
        H_nodes = self.MessagePassing(send_nodes, receive_nodes, prev_node_emb)
        
        # temporal modeling
        cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a = self.TemporalModeling(event, H_nodes, prev_graph_emb, prev_graph_mem, prev_node_emb, prev_node_mem)
        
        return cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a
    
    def MessagePassing(self, send_nodes, receive_nodes, prev_node_embedding):

        # transition matrix
        Min = torch.matmul(send_nodes.reshape(-1, self.n_nodes, 1), receive_nodes.reshape(-1, 1, self.n_nodes))
        Mout = torch.matmul(receive_nodes.reshape(-1, self.n_nodes, 1), send_nodes.reshape(-1, 1, self.n_nodes))

        # middle embedding
        Hin_ = torch.reshape(torch.matmul(Min, prev_node_embedding), (-1, self.n_features))
        Hin = self.activation(self.Hin(Hin_))
        Hout_ = torch.reshape(torch.matmul(Mout, prev_node_embedding), (-1, self.n_features))
        Hout = self.activation(self.Hout(Hout_))

        # pooling
        # shape: [-1, n_nodes, n_features]
        H, _ = torch.max(torch.reshape(torch.cat([Hin, Hout], dim=-1), (-1, self.n_nodes, self.n_features, 2)), dim=-1)
        return H

    def TemporalModeling(self, event, middle_node_emb, prev_graph_emb, prev_graph_mem, prev_node_emb, prev_node_mem):
        
        # attention score
        cur_a = self.A(torch.cat([torch.mean(middle_node_emb, dim=1), prev_graph_emb], dim=-1))
        g_ = torch.ones(self.batch_size, self.n_nodes, self.graph_dim, device="cuda:0") * torch.unsqueeze(cur_a * prev_graph_emb, axis=1)
        h_input = torch.cat([g_, middle_node_emb], dim=-1)
        h_input = torch.reshape(h_input, (-1, self.n_features+self.graph_dim))
        prev_node_emb = torch.reshape(prev_node_emb, (-1, self.n_features))
        prev_node_mem = torch.reshape(prev_node_mem, (-1, self.n_features))
        cur_node_emb, cur_node_mem = self.node_lstm(h_input, prev_node_emb, prev_node_mem)
        cur_node_emb = torch.reshape(cur_node_emb, [-1, self.n_nodes, self.n_features])
        cur_node_mem = torch.reshape(cur_node_mem, [-1, self.n_nodes, self.n_features])
        
        # next graph embedding
        h_ = cur_a * torch.mean(cur_node_emb, dim=1)
        g_input = torch.cat([h_, event], dim=-1)
        cur_graph_emb, cur_graph_mem  = self.graph_lstm(g_input, prev_graph_emb, prev_graph_mem)

        return cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a

class Lstm_unit(nn.Module):
    def __init__(self, input_feature_1, input_feature_2, output_feature):
        super(Lstm_unit, self).__init__()
        self.Wi = nn.Linear(input_feature_1, output_feature)
        self.Vi = nn.Linear(input_feature_2, output_feature)
        self.Wf = nn.Linear(input_feature_1, output_feature)
        self.Vf = nn.Linear(input_feature_2, output_feature)
        self.Wo = nn.Linear(input_feature_1, output_feature)
        self.Vo = nn.Linear(input_feature_2, output_feature)
        self.Wc = nn.Linear(input_feature_1, output_feature)
        self.Vc = nn.Linear(input_feature_2, output_feature)
        self.activation_s = nn.Sigmoid()
        self.activation_t = nn.Tanh()
    def forward(self, input, prev_hidden, prev_memory):
        I = self.activation_s(self.Wi(input) + self.Vi(prev_hidden))
        F = self.activation_s(self.Wf(input) + self.Vf(prev_hidden))
        O = self.activation_s(self.Wo(input) + self.Vo(prev_hidden))
        C = self.activation_s(self.Wi(input) + self.Vi(F * prev_hidden))
        # output
        Ct = F * prev_memory + I * C
        # current information
        current_memory = Ct
        current_hidden = O * self.activation_t(Ct)
        return current_memory, current_hidden