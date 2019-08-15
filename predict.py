import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from transnet.data import DataLoaders
from transnet.model import TransNet
from transnet.utils import evaluate, EvalTrain

train = False

if train:
    data_file = 'data/processed_train.npz'
else:
    data_file = 'data/processed_test.npz'

trained_folder = 'trained_models/'
model_path = 'best_v2_32.pth'
inner_channels = 32
#inner_channels = int(model_path[4:-4])

batch_size = 2 ** 10
data_loaders = DataLoaders(data_file)

prediction_loader = data_loaders.get_data_loaders(batch_size=batch_size,
                                                  inference=True)

model = TransNet(inner_channels)
model.load_state_dict(torch.load(trained_folder + model_path))
model.cuda()

y_act, y_pred, data_ids = evaluate(model, prediction_loader, inference=True)

prediction = pd.DataFrame()
prediction['id'] = data_ids
prediction['scalar_coupling_constant'] = y_pred
prediction.index = prediction['id'].astype(int)

if train:
    mae = np.log(mean_absolute_error(y_act, y_pred))
    r2 = r2_score(y_act, y_pred)
    print(mae, r2)

    eval_train = EvalTrain(prediction, train_file='data/raw_data/train.csv')
    mean = eval_train.calc_mlmae()
    print('expected test score: {:0.3f}'.format(mean))

    y_a = pd.DataFrame(y_act).sample(n=10000)
    y_p = pd.DataFrame(y_pred).loc[y_a.index]

    plt.plot(y_a, y_p, 'o', alpha=0.5)


else:
    df_test = pd.read_csv('data/raw_data/test.csv')
    df_test.index = df_test['id']
    submission = pd.concat([df_test['id'],
                           prediction['scalar_coupling_constant']], axis=1)

    submission.to_csv('data/submission2.gz',
                      index=False,
                      header=True,
                      compression='gzip')
