import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from xyznet.data import DataLoaders
from xyznet.model import XyzConv
from xyznet.utils import evaluate, plot_act_pred, Metrics

# %%

#data_file = 'data/processed_train_type_0.csv'
#data_file = 'data/processed_train_type_1.csv'
#data_file = 'data/processed_train_type_2.csv'
#data_file = 'data/processed_train_type_3.csv'
#data_file = 'data/processed_train_type_4.csv'
#data_file = 'data/processed_train_type_5.csv'
#data_file = 'data/processed_train_type_6.csv'
#data_file = 'data/processed_train_type_7.csv'
data_file = 'data/processed_train_type_all.csv'
batch_size = 2 ** 9
data_loaders = DataLoaders(data_file)

# %%

train_loader, val_loader = data_loaders.get_data_loaders(batch_size=batch_size,
                                                         train_frac=0.9,
                                                         val_frac=0.1)

mini_loaders = data_loaders.get_data_loaders(batch_size=batch_size,
                                             train_frac=0.005,
                                             val_frac=0.005)
mini_train_loader, mini_val_loader = mini_loaders

inner_channels = 32
model = XyzConv(inner_channels)
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

checkin = int(50000 / batch_size)
if checkin == 0:
    checkin = 1
check_metric = int(1000000 / batch_size)
if check_metric > len(train_loader):
    check_metric = len(train_loader) - 1
model.train()
pred_list = []
act_list = []
train_metrics = Metrics('train')
val_metrics = Metrics('val')
trainloss = Metrics('train_loss')
for epoch in range(100):
    print('epoch: {:0.0f}, total samples trained on:'
          ' {:0.0f}'.format(epoch, epoch*len(train_loader)*batch_size))
    ti = time.time()
    for i, (elem_vec, relative_pos, target) in enumerate(train_loader):
        elem_vec = Variable(elem_vec.cuda(non_blocking=True))
        relative_pos = Variable(relative_pos.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        output = model(elem_vec, relative_pos)
        loss = criterion(output, target)
        pred_list += output.detach().cpu().numpy().tolist()
        act_list += target.detach().cpu().numpy().ravel().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tf = time.time()
        trainloss.update_loss(loss.detach().cpu().numpy().sum())
        if i % checkin == 0:
            if i == 0:
                continue
            print('epoch:', epoch, '  molecules/sec ~', batch_size / (tf - ti))
            trainloss.show_loss()
            plt.figure(figsize=(5, 5))
            plot_act_pred(np.array(act_list), np.array(pred_list))
            plt.show()
            model.train()
            pred_list = []
            act_list = []
        if i % check_metric == 0:
            if i == 0:
                continue
            print('computing inference results...')
            y_act_train, y_pred_train = evaluate(model, mini_train_loader)
            y_act_val, y_pred_val = evaluate(model, mini_val_loader)
            train_metrics.update(y_act_train, y_pred_train)
            val_metrics.update(y_act_val, y_pred_val)
            train_metrics.show()
            val_metrics.show()
            plt.figure(figsize=(5, 5))
            plot_act_pred(y_act_train, y_pred_train, label='train')
            plot_act_pred(y_act_val, y_pred_val, label='val')
            plt.legend()
            plt.show()
            model.train()
        ti = time.time()
#        if i == 15000:
#            break
    scheduler.step()

