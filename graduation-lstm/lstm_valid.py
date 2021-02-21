import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from data_processer import DataProcesser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = json.load(open('config.json', 'r'))


lstm = torch.load('lstm_model.pkl')

data_processer = DataProcesser()
train_loader, valid_loader = data_processer.data_processer()


y_valid_pred = lstm(valid_loader.dataset.x.to(device)).cpu()
y_valid_pred = torch.squeeze(y_valid_pred, dim=2)
y_valid_pred = y_valid_pred[:, configs['model']['layers']['input_dim']-1].detach().numpy()

y_train = torch.squeeze(train_loader.dataset.y, dim=2)
y_train = y_train[:, configs['model']['layers']['input_dim']-1 ].detach().numpy()

y_valid = torch.squeeze(valid_loader.dataset.y, dim=2)
y_valid = y_valid[:, configs['model']['layers']['input_dim']-1 ].detach().numpy()

y = np.append(y_train, y_valid)

plt.plot(y, label='observation')
plt.plot(np.arange(len(y)-len(y_valid_pred), len(y)), y_valid_pred, label='prediction')
plt.legend()
plt.grid(True)
plt.axis('tight')
plt.setp(plt.gca().get_xticklabels(), rotation=50)
plt.show()