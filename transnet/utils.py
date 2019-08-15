import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate(model, data_loader, inference=False):
    y_act = []
    y_pred = []
    data_ids = []
    model.eval()
    for i, (model_args) in enumerate(data_loader):
        elem_vec, relative_pos, data_id, target = model_args
        elem_vec = Variable(elem_vec.cuda(non_blocking=True))
        relative_pos = Variable(relative_pos.cuda(non_blocking=True))
        output, ids = model(elem_vec, relative_pos, data_id)
        y = torch.Tensor(target)
        y_pred += output.detach().cpu().numpy().tolist()
        y_act += y.detach().numpy().tolist()
        data_ids += ids.detach().numpy().tolist()
        if inference:
            if i % int(len(data_loader)/10) == 0:
                percent = i/len(data_loader) * 100
                print('{:0.3f}% complete'.format(percent))
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
    if inference:
        return y_act, y_pred, data_ids
    return y_act, y_pred


def plot_act_pred(y_act, y_pred, label=None):
    max_max = max([y_act.max(), y_pred.max()])
    min_min = max([y_act.min(), y_pred.min()])
    plt.plot(y_act, y_pred, 'o', alpha=0.3, mfc='grey', label=label)
    plt.plot([max_max, min_min], [max_max, min_min], 'k-', alpha=0.7)
    plt.xlim(min_min, max_max)
    plt.ylim(min_min, max_max)
    plt.tick_params(right=True, top=True, direction='in')


class Metrics():
    def __init__(self, name='train'):
        self.name = name
        self.n = 0
        self.r2_list = []
        self.mae_list = []
        self.r2 = 0
        self.r2_sum = 0
        self.r2_avg = 0
        self.mae = 0
        self.mae_sum = 0
        self.mae_avg = 0

    def update(self, y_act, y_pred):
        self.n += 1
        self.r2 = r2_score(y_act, y_pred)
        self.r2_sum += self.r2
        self.r2_avg = self.r2_sum / self.n
        self.mae = mean_absolute_error(y_act, y_pred)
        self.mae_sum += self.mae
        self.mae_avg = self.mae_sum / self.n
        self.r2_list.append(self.r2)
        self.mae_list.append(self.mae)
        if len(self.r2_list) > 1000:
            self.r2_list = self.r2_list[-500:]
            self.mae_list = self.mae_list[-500:]

    def show(self):
        r2_str = ' {:<9} {:0.3f} ({:0.3f})\n'.format('r2:',
                                                     self.r2,
                                                     self.r2_avg)
        mae_str = ' {:<9} {:0.3f} ({:0.3f})\n'.format('mae:',
                                                      self.mae,
                                                      self.mae_avg)
        print(self.name + r2_str + self.name + mae_str)

    def update_loss(self, loss):
        self.n += 1
        self.dmae = loss - self.mae
        self.mae = loss
        self.mae_sum += self.mae
        self.mae_avg = self.mae_sum / self.n

    def show_loss(self):
        loss_text = ' {:<5} loss (delta): {:0.3f} ({:0.3f}),' \
                       '   Average: {:0.3f}'
        mae_str = loss_text.format('mae:', self.mae, self.dmae, self.mae_avg)
        print(self.name + mae_str)

    def plot(self):
        plt.figure(figsize=(5, 5))
        plt.plot(np.arange(0, self.n, 1), self.mae_list, '-o')
        plt.xlabel('updates')
        plt.ylabel('mae')
        plt.tick_params(right=True, top=True, direction='in')
        plt.show()


class EvalTrain():
    def __init__(self, df_train_prediction, train_file='raw_data/train.csv'):
        self.df_train = pd.read_csv(train_file)
        self.df_train.index = self.df_train.iloc[:, 0].astype(int)
        self.df_train_prediction = df_train_prediction
        self.coupling_types = ['3JHC',
                               '2JHC',
                               '1JHC',
                               '3JHH',
                               '2JHH',
                               '3JHN',
                               '2JHN',
                               '1JHN']
        self.all_lmae = {}

    def calc_mlmae(self):
        for coupling_type in self.coupling_types:
            y_act = self.df_train[self.df_train['type'] == coupling_type]
            y_act = y_act['scalar_coupling_constant']
            y_pred = self.df_train_prediction.loc[y_act.index, :]
            y_pred = y_pred['scalar_coupling_constant']
            mae = mean_absolute_error(y_act, y_pred)
            lmae = np.log10(mae)
            self.all_lmae[coupling_type] = lmae
            plt.plot(y_act, y_pred, 'o', label=coupling_type)
        plt.legend()
        plt.show()
        mean_lmae = np.array(list(self.all_lmae.values())).mean()
        return mean_lmae


