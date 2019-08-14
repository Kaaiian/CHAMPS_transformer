import numpy as np
from numba import jit


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
    relative_positions = np.ones(shape=(2, n_elems))
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
        relative_positions[0, j] = nbr_dist
        relative_positions[1, j] = cos_angle_site
        relative_positions[2, j] = angle_site
        relative_positions[3, j] = cntr_dist
        relative_positions[4, j] = cos_angle_cntr
        relative_positions[5, j] = angle_cntr
    return relative_positions


@jit(nopython=True)
def get_sites(molecule):
    target = molecule[-1]
    elems = molecule[:29]
    sites = molecule[29:4 * 29]
    idx1 = int(molecule[-4])
    idx2 = int(molecule[-3])

    # remove the filler 'dummy' elements
    n_elems = elems.argmin()
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
    elem_vec = np.concatenate((one_hot_elems, coupling_pairs), axis=1)

    # get positions relative to center between indices
    relative_positions = rel_loops(sites,
                                   n_elems,
                                   idx1,
                                   idx2)

    return elem_vec, relative_positions, target

ti = time.time()
for i in range(30000):
    molecule = self.data[idx, :]

    elem_vec, relative_positions, target = get_sites(molecule)

    sort_index = relative_positions.argsort()[0]
    relative_positions = relative_positions.transpose()[sort_index]
    elem_vec = elem_vec[sort_index]

    elem_vec = torch.Tensor(elem_vec)
    relative_positions = torch.Tensor(relative_positions)
    target = torch.Tensor([float(target)])
tf = time.time()

print(tf - ti)