# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 07:16:29 2019

@author: Kaai
"""
ti = time.time()
for i in range(10000):
    molecule = self.data[i, :]
    # get position information using numba functions
    elem_vec, relative_positions, target = get_sites(molecule)
    # sort molecules by proximity to center
    sort_index = relative_positions.argsort()[0]
    relative_positions = relative_positions.transpose()[sort_index]
    elem_vec = elem_vec[sort_index]
    # convert to tensors
    elem_vec = torch.Tensor(elem_vec)
    relative_positions = torch.Tensor(relative_positions)
    target = torch.Tensor([float(target)])
    
tf = time.time()
print(10000/(tf-ti))