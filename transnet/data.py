import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numba import jit


class DataLoaders():
    def __init__(self, data_file):
        self.data = np.load(data_file)['arr_0']

    def get_data_loaders(self,
                         batch_size=1,
                         train_frac=0.025,
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


@jit(nopython=True)
def get_site_mean(sites, n_elems):
    shaped = sites[n_elems:].reshape(3, n_elems)
    dim1 = shaped[0].mean()
    dim2 = shaped[1].mean()
    dim3 = shaped[2].mean()
    center_coords = np.array([dim1, dim2, dim3])
    return center_coords


@jit(nopython=True)
def get_angle(v1, v2):
    if v1.sum() == 0:
        angle = 0
    elif v2.sum() == 0:
        angle = 0
    else:
        cos_angle = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        angle = np.arccos(cos_angle)
    return cos_angle, angle


@jit(nopython=True)
def rel_loops(sites, n_elems, idx1, idx2):
    cntr_loc = get_site_mean(sites, n_elems)
    relative_pos = np.ones(shape=(6, n_elems))
    site1 = sites[n_elems+3*idx1:n_elems+3*idx1+3]
    site2 = sites[n_elems+3*idx2:n_elems+3*idx2+3]
    site_loc = (site1 + site2)/2
    for j in range(n_elems):
        site_j = sites[n_elems+3*j:n_elems+3*j+3]
        nbr_dist = np.linalg.norm(site_j - site_loc)
        cntr_dist = np.linalg.norm(site_j - cntr_loc)
        v1 = site1 - site_j
        v2 = site2 - site_j
        v3 = cntr_loc - site_j
        v4 = site_loc - site_j
        cos_angle_site, angle_site = get_angle(v1, v2)
        cos_angle_cntr, angle_cntr = get_angle(v3, v4)
        relative_pos[0, j] = nbr_dist
        relative_pos[1, j] = cos_angle_site
        relative_pos[2, j] = angle_site
        relative_pos[3, j] = cntr_dist
        relative_pos[4, j] = cos_angle_cntr
        relative_pos[5, j] = angle_cntr
    return relative_pos


@jit(nopython=True)
def get_sites(molecule):
    target = molecule[-1]
    data_id = molecule[-2]
    coupling_type = int(molecule[-3])
    elems = molecule[:29]
    sites = molecule[29:4 * 29]
    idx1 = int(molecule[-5])
    idx2 = int(molecule[-4])

    # remove the filler 'dummy' elements
    n_elems = 29
    elems = elems[0:n_elems]
    sites = sites[0:n_elems * 3]
    sites = np.concatenate((elems, sites))

    # get one-hot encodings
    one_hot_elems = np.zeros(shape=(n_elems, 5))
    for i, elem in enumerate(elems):
        one_hot_elems[i, int(elem)-1] = 1

    # get coupling pairs
    coupling_pairs = np.zeros(shape=(n_elems, 1))
    coupling_pairs[idx1, :] = 1
    coupling_pairs[idx2, :] = 1
    coupling_types = np.zeros(shape=(n_elems, 8))
    coupling_types[idx1, coupling_type] = 1
    coupling_types[idx2, coupling_type] = 1
    elem_props = (one_hot_elems, coupling_pairs, coupling_types)
    elem_vec = np.concatenate(elem_props, axis=1)

    # get positions relative to center between indices
    relative_pos = np.ones(shape=(n_elems, 2))
    relative_pos = rel_loops(sites,
                             n_elems,
                             idx1,
                             idx2)
    return elem_vec, relative_pos, data_id, target


class XyzData(Dataset):
    """
    this is kaai's dataset
    """
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        molecule = self.data[idx, :]
        # get position information using numba functions
        elem_vec, relative_pos, data_id, target = get_sites(molecule)
        # sort molecules by proximity to center
        sort_index = relative_pos.argsort()[0]
        relative_pos = relative_pos.transpose()[sort_index]
        elem_vec = elem_vec[sort_index]
        # convert to tensors
        elem_vec = torch.Tensor(elem_vec)
        relative_pos = torch.Tensor(relative_pos)
        target = torch.Tensor([float(target)])
        return (elem_vec, relative_pos, data_id, target)
