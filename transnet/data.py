import numpy as np
import functools
import torch
from torch.utils.data import Dataset, DataLoader
# from numba import jit
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_props = pd.read_csv('data/element_properties/mat2vec.csv', index_col=0)
index_array = zip(df_props.index, df_props.values)
elem_dict = {index: array for index, array in index_array}

class DataLoaders():
    def __init__(self, data_file):
        self.data = pd.read_csv(f'data/{data_file}').sample(frac=1).values
        zero_label = self.data[:, 1] == -1
        self.data[zero_label, 1] = 0
        self.scaler = StandardScaler()

    def get_data_loaders(self,
                         batch_size=1,
                         train_frac=0.15,
                         val_frac=0.01,
                         inference=False):
        '''
        input the dataset, get train test split
        '''
        if inference:
            prediction_dataset = XyzData(self.data)
            prediction_loader = DataLoader(prediction_dataset,
                                           batch_size=batch_size,
                                           pin_memory=True,
                                           shuffle=False)
            return prediction_loader

        train_size = int(self.data.shape[0] * train_frac)
        val_size = int(self.data.shape[0] * val_frac)
        self.data[:train_size, 2:] = self.scaler.fit_transform(self.data[:train_size, 2:])
        self.data[train_size:, 2:] = self.scaler.transform(self.data[train_size:, 2:])
        train_dataset = XyzData(self.data[:train_size, :])
        val_dataset = XyzData(self.data[-val_size:, :])
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                pin_memory=True,
                                shuffle=True)
        return train_loader, val_loader



class XyzData(Dataset):
    """
    this is kaai's xyz description of a molecule.
    """

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return self.data.shape[0]

    @functools.lru_cache(maxsize=None)  # Cache loaded molecules
    def __getitem__(self, idx):
        # sample molecule from data (mol_id)
        mol_id = self.data[idx, 0]
        # set target values to zero, (will put a "1" where label goes)
        target = np.zeros(2)
        # self.data[idx, 1] contains 1 or 0 depending on centro/non-centro
        target[self.data[idx, 1]] = 1
        # read in the molecule's xyz file (mol)
        mol = pd.read_csv(f'data/xyzs/{mol_id}.xyz',
                          delim_whitespace=True, skiprows=[0, 1], header=None)

        # get molecule properties (mol_vec)
        mol_vec = self.data[idx, 2:].astype(float)
        # print(mol_vec)
        # define site properties (elem_vec)
        elem = mol[0].map(elem_dict)
        elem_vec = np.concatenate(elem).reshape(len(elem), -1)

        # get site neighbor properties (nbrs_vec)
        xyz = mol.values[:, 1:].astype(float)
        mat1 = np.expand_dims(xyz, 0)
        mat2 = np.expand_dims(xyz, 1)
        dist_mat = np.sqrt(((mat2 - mat1)**2).sum(-1))
        n_nbrs = 5
        n_sites = 12
        nbrs_idx = [np.argsort(dist)[0:n_nbrs+1] for dist in dist_mat]
        nbrs_idx = np.array(nbrs_idx)
        # if len(nbrs_idx) < n_nbrs:
        #     print('ahhhhhhhh')
        #     print(len(nbrs_idx), len(n_nbrs))
        row_expansion = np.expand_dims(np.arange(len(nbrs_idx)), -1)
        nbrs_dist = dist_mat[row_expansion, nbrs_idx]
        mol_mat = np.ones(shape=(n_sites, 200*(n_nbrs+1)+len(mol_vec))) * -10
        for i in range(xyz.shape[0]):
            # print(i)
            # print(mol_mat.shape[0])
            if i == mol_mat.shape[0]:
                break
            # get the properties for all closest sites (including current site)
            site_vec = np.concatenate(elem_vec[nbrs_idx[i]]).ravel()
            dist_off = [[dist]*200 for dist in nbrs_dist[i]]
            dist_off = np.array(dist_off)
            dist_off = np.sqrt(dist_off) / 50
            dist_off = dist_off.ravel()

            site_vec = site_vec + dist_off
            # print(site_vec.shape)
            site_vec = np.concatenate([site_vec, mol_vec])
            # print(site_vec.shape)
            # print(mol_mat.shape)
            mol_mat[i, :] = site_vec
        mol_mat = torch.Tensor(mol_mat / n_sites * 10)
        site_vec = torch.Tensor(site_vec)
        target = torch.Tensor(target)

        return (mol_mat, target, mol_id)


# %%
if __name__ == '__main__':
    dataloaders = DataLoaders(data_file='MLvector.csv')
    train_loader, val_loader = dataloaders.get_data_loaders()
    for i, data in enumerate(train_loader):
        break
        print(data[0].shape)



