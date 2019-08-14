import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate(model, data_loader):
    y_act = []
    y_pred = []
    for i, (elem_vec, relative_pos, target) in enumerate(data_loader):
        elem_vec = Variable(elem_vec.cuda(non_blocking=True))
        relative_pos = Variable(relative_pos.cuda(non_blocking=True))
        model.eval()
        output = model(elem_vec, relative_pos)
        y = torch.Tensor(target)
        y_pred += output.detach().cpu().numpy().tolist()
        y_act += y.detach().cpu().numpy().tolist()
    y_act = np.array(y_act)
    y_pred = np.array(y_pred)
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
        self.r2_list = []
        self.r2 = 0
        self.r2_sum = 0
        self.r2_avg = 0
        self.mae = 0
        self.mae_sum = 0
        self.mae_avg = 0

    def save(self):
        self.r2_list.append(self.r2)
        self.mae_list.append(self.mae)

    def update(self, y_act, y_pred):
        self.n += 1
        self.r2 = r2_score(y_act, y_pred)
        self.r2_sum += self.r2
        self.r2_avg = self.r2_sum / self.n
        self.mae = mean_absolute_error(y_act, y_pred)
        self.mae_sum += self.mae
        self.mae_avg = self.mae_sum / self.n

    def show(self):
        r2_str = ' {: <9} {: 0.3f} ({: 0.3f})\n'.format('r2:',
                                                        self.r2,
                                                        self.r2_avg)
        mae_str = ' {: <9} {: 0.3f} ({: 0.3f})\n'.format('mae:',
                                                         self.mae,
                                                         self.mae_avg)
        print(self.name + r2_str + self.name + mae_str)

    def update_loss(self, loss):
        self.n += 1
        self.mae = loss
        self.mae_sum += self.mae
        self.mae_avg = self.mae_sum / self.n

    def show_loss(self):
        mae_str = ' {: <9} {: 0.3f} ({: 0.3f})\n'.format('mae:',
                                                         self.mae,
                                                         self.mae_avg)
        print(self.name + mae_str)

    def plot():
        print('r2')
