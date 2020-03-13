import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.autograd import Variable
from transnet.data import DataLoaders
from transnet.model import CrabNet
from transnet.utils import (evaluate, plot_act_pred, Metrics, plot_roc_curve,
                            count_parameters)


plt.rcParams.update({'font.size': 16})

# %%

data_file='MLvector.csv'
data_loaders = DataLoaders(data_file)

# %%
n_nbrs = 5
input_dims = 200*(n_nbrs+1)+19
# input_dims = 200*(n_nbrs+1)
# input_dims = 19
model = CrabNet(input_dims,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=64,
                dropout=0.1)
model.cuda()
pos_weight = torch.Tensor([1, 0.0001]).cuda()
criterion = nn.BCEWithLogitsLoss(weight=pos_weight)
print(f'total parameters: {count_parameters(model)}')

batch_size = 1024

train_loader, val_loader = data_loaders.get_data_loaders(batch_size=batch_size,
                                                         train_frac=0.1,
                                                         val_frac=0.1)

mini_loaders = data_loaders.get_data_loaders(batch_size=batch_size,
                                              train_frac=0.005,
                                              val_frac=0.005)
mini_train_loader, mini_val_loader = mini_loaders

# %%

# optimizer = torch.optim.AdamW(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)

checkin = int(50000 / batch_size)
if checkin == 0:
    checkin = 1
check_metric = 10
sigmoid = m = nn.Sigmoid()
best_mae = 20
model.train()
pred_list = []
act_list = []
train_metrics = Metrics('train')
val_metrics = Metrics('val')
trainloss = Metrics('train_loss')
for epoch in range(10000):
    print('epoch: {:0.0f}, total samples trained on:'
          ' {:0.0f}'.format(epoch, epoch*len(train_loader)*batch_size))
    ti = time.time()
    for i, model_args in enumerate(train_loader):
        site_vec, target, mol_id = model_args
        site_vec = Variable(site_vec.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        output, _ = model(site_vec, mol_id)
        loss = criterion(output, target)
        pred_list += sigmoid(output).detach().cpu().numpy().tolist()
        act_list += target.detach().cpu().numpy().ravel().tolist()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tf = time.time()
        trainloss.update_loss(loss.detach().cpu().numpy().sum())

    if epoch % check_metric == 0:
        plt.figure(figsize=(6, 6))
        # if i == 0:
        #     continue
        print(f'train performance...{loss}')
        check_pred_list = []
        check_act_list = []
        for i, model_args in enumerate(train_loader):
            site_vec, target, mol_id = model_args
            site_vec = Variable(site_vec.cuda(non_blocking=True))
            target = Variable(target.cuda(non_blocking=True))
            output, _ = model(site_vec, mol_id)

            check_pred_list += sigmoid(output).detach().cpu().numpy()[:, 0].tolist()
            check_act_list += target.detach().cpu().numpy()[:, 0].ravel().tolist()
        score = roc_auc_score(check_act_list, check_pred_list)
        pred_labels = [0 if val < 0.5 else 1 for val in check_pred_list]
        prfs = precision_recall_fscore_support(check_act_list, pred_labels)
        print(f'pr: {prfs[0][0]:0.4f}, rc: {prfs[1][0]:0.4f}, fsc: {prfs[2][0]:0.4f}')
        print(f'pr: {prfs[0][1]:0.4f}, rc: {prfs[1][1]:0.4f}, fsc: {prfs[2][1]:0.4f}')
        plot_roc_curve(check_act_list, check_pred_list, 'train')
        print(f'score: {score}')

        print('val performance...')
        check_pred_list = []
        check_act_list = []
        for i, model_args in enumerate(val_loader):
            site_vec, target, mol_id = model_args
            site_vec = Variable(site_vec.cuda(non_blocking=True))
            target = Variable(target.cuda(non_blocking=True))
            output, _ = model(site_vec, mol_id)

            check_pred_list += sigmoid(output).detach().cpu().numpy()[:, 0].tolist()
            check_act_list += target.detach().cpu().numpy()[:, 0].ravel().tolist()
        score = roc_auc_score(check_act_list, check_pred_list)
        pred_labels = [0 if val < 0.5 else 1 for val in check_pred_list]
        prfs = precision_recall_fscore_support(check_act_list, pred_labels)
        print(f'pr: {prfs[0][0]:0.4f}, rc: {prfs[1][0]:0.4f}, fsc: {prfs[2][0]:0.4f}')
        print(f'pr: {prfs[0][1]:0.4f}, rc: {prfs[1][1]:0.4f}, fsc: {prfs[2][1]:0.4f}')
        plot_roc_curve(check_act_list, check_pred_list, 'val')
        plt.show()
        print(f'score: {score}')
            # y_act_train, y_pred_train = evaluate(model, mini_train_loader)
            # y_act_val, y_pred_val = evaluate(model, mini_val_loader)
            # train_metrics.update(y_act_train, y_pred_train)
            # val_metrics.update(y_act_val, y_pred_val)
            # train_metrics.show()
            # val_metrics.show()
            # plt.figure(figsize=(5, 5))
            # plot_act_pred(y_act_train, y_pred_train, label='train')
            # plot_act_pred(y_act_val, y_pred_val, label='val')
            # plt.legend()
            # plt.show()
            # if val_metrics.mae < best_mae:
            #     torch.save(model.state_dict(),
            #                'trained_models/best_v3_'+str(inner_channels)+'.pth')
            #     best_mae = val_metrics.mae
        model.train()
        ti = time.time()
#        if i == 15000:
#            break
    # scheduler.step()

