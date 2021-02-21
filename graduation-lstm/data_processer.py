import json
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from dataloader import DataSet
class DataProcesser():

    def data_processer(self):
        configs = json.load(open('config.json', 'r'))

        ## 载入数据
        data = pd.read_excel('features.xlsx')
        all_data = data.head(500)

        ## 数据预处理
        scaler = MinMaxScaler(feature_range=(0, 1))     # 将数据压缩至0-1之间
        sel_col = configs['data']['columns']
        all_data = all_data[sel_col]

        # 将数据集每个特征都压缩到0-1之间
        for col in sel_col[1:]:
            all_data[col] = scaler.fit_transform(all_data[col].values.reshape(-1, 1))

        # print(all_data)

        all_data = torch.tensor(np.array(all_data))

        length = len(all_data)
        tr_val_slip = int(configs['data']['train_test_split'] * length)       # 将前80%的数据作为训练集

        x = torch.zeros(length - configs['data']['sequence_length'], configs['data']['sequence_length'], configs['model']['layers']['input_dim'])
        y = torch.zeros(length - configs['data']['sequence_length'], configs['data']['sequence_length'], configs['model']['layers']['output_dim'])

        for i in range(0, length - configs['data']['sequence_length'] - 1):
            x[i] = all_data[i: i + configs['data']['sequence_length']]
            y[i] = all_data[i + 1: i + configs['data']['sequence_length'] + 1][:, 0].reshape(-1, 1)

        train_x = x[0: tr_val_slip]
        train_y = y[0: tr_val_slip]
        valid_x = x[tr_val_slip:]
        valid_y = y[tr_val_slip:]

        train_set = DataSet(train_x, train_y)
        valid_set = DataSet(valid_x, valid_y)
        train_loader = DataLoader(train_set, batch_size=configs['training']['batch_size'], shuffle=False)
        valid_loader = DataLoader(valid_set, batch_size=configs['training']['batch_size'], shuffle=False)

        return train_loader, valid_loader