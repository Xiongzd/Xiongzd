import json
import torch
import time
import matplotlib
import torch.nn as nn
from LSTM import LSTM
from data_processer import DataProcesser
from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = json.load(open('config.json', 'r'))

# model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers)

# k = torch.rand(8, 20, 13).device
# with SummaryWriter(comment='lstm') as w:
#     w.add_graph(model_demo, (k,))

data_processer = DataProcesser()
train_loader, valid_loader = data_processer.data_processer()

lstm = LSTM(input_dim=configs['model']['layers']['input_dim'], hidden_dim=configs['model']['layers']['neurons'], out_dim=configs['model']['layers']['output_dim'],
                                   num_layers=configs['model']['layers']['layers']).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-4)

tol_time = time.time()

train_losses = []
valid_losses = []
for epoch in range(configs['training']['epochs']):
    epoch_start_time = time.time()
    train_loss = 0
    train_acc = 0.
    val_loss = 0
    val_acc = 0.

    lstm.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        y_pred = lstm(inputs)
        loss = loss_function(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss)

    lstm.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            val_pred = lstm(inputs)
            loss = loss_function(val_pred, labels)
            val_loss += loss.item()
        valid_losses.append(val_loss)

        if epoch % 100 == 0 and epoch != 0:
            print('[%03d/%03d] Tra_Loss: %3.6f||val_Loss: %3.6f' % (epoch, configs['training']['epochs'], train_loss, val_loss))
print("训练以及验证的总运行总时间：", time.time() - tol_time)

if configs['model']['save_model']:
    torch.save(lstm, 'lstm_model.pkl')

figsize = 8,6
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.08, right=0.94, bottom=0.06, top=0.94)
plt.grid(True)
plt.plot(train_losses, label='train_loss')
plt.plot(valid_losses, label='valid_loss')
plt.legend()
plt.axis('tight')
plt.setp(plt.gca().get_xticklabels(), rotation=50)
plt.show()


# y_valid_pred = lstm(valid_loader.dataset.x.to(device)).cpu()
# y_valid_pred = torch.squeeze(y_valid_pred, dim=2)
# y_valid_pred = y_valid_pred[:, configs['model']['layser']['input_dim']-1].detach().numpy()
#
# y = torch.squeeze(valid_loader.dataset.y, dim=2)
# y = y[:, configs['model']['layser']['input_dim']-1 ].detach().numpy()
#
#
# plt.plot(y, label='observation')
# plt.plot(np.arange(len(y)-len(y_valid_pred), len(y)), y_valid_pred, label='prediction')
# plt.legend()
# plt.grid(True)
# plt.axis('tight')
# plt.setp(plt.gca().get_xticklabels(), rotation=50)
# plt.show()