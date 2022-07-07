import gc

import numpy as np
import torch
from tqdm import tqdm

def check_nan(*tensors):
    for t in tensors:
        if t.isnan().sum() > 0:
            print('Found NaN:', t.shape)
            return True
    return False


def check_nan_np(*tensors):
    for t in tensors:
        if np.isnan(t).sum() > 0:
            print('Found NaN:', t.shape)
            return True
    return False


def convert_to_dense(mu, nu, spmatrix, symmetrize=False):
    '''
    Convert mu, nu indices and matrix values into dense matrix.

    :param mu:
    :param nu:
    :param spmatrix:
    :return:
    '''
    if isinstance(spmatrix, torch.Tensor):
        idx = torch.cat([mu[:, 0:1], nu[:, 0:1]], dim=1)
        imax = torch.max(mu[:, 0:1]) + 1
        dense = torch.sparse.FloatTensor(idx.t(), spmatrix,
                                         torch.Size([imax, imax])).to_dense()
        if symmetrize:
            dense = 0.5 * dense + 0.5 * dense.t()
    else:  # numpy
        imax = np.max(mu[:, 0:1]) + 1

        dense = np.zeros((imax, imax), dtype=np.float32)
        dense[mu[:, 0], nu[:, 0]] = spmatrix
        if symmetrize:
            dense = 0.5 * dense + 0.5 * dense.T

    return dense


def tensor_meta_data(tensor):
    element_count = 1;
    for dim in tensor.size():
        element_count = element_count * dim
    size_in_bytes = element_count * tensor.element_size()
    dtype = str(tensor.dtype).replace("torch.", "")
    size = str(tensor.size()).replace("torch.Size(", "").replace(")", "")
    return f"{size_in_bytes / 1000000:5.1f}MB" + \
           f" {dtype}{size} {type(tensor).__name__} {tensor.device}"


def print_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(tensor_meta_data(obj))


def get_number_orbitals(database):
    basis_def = database.basisdef
    n_orbitals = np.zeros(basis_def.shape[0], dtype=np.int)

    for i in range(basis_def.shape[0]):
        n_orbitals[i] = int(np.count_nonzero(basis_def[i, :, 2]))

    return n_orbitals


def get_average_energies(database, n_orbitals):
    atype_entries = n_orbitals.shape[0]

    mean_orb_energies = np.zeros((atype_entries, np.max(n_orbitals)))
    atype_count = np.zeros(atype_entries)

    for row_id in tqdm(range(len(database))):
        row = database[row_id]
        ham = row['hamiltonian']

        energies = np.diag(ham)

        # Get atom types
        atypes = row["_atomic_numbers"]
        pos = 0
        for atom in atypes:
            inc = n_orbitals[atom]
            mean_orb_energies[atom, :inc] += energies[pos:pos + inc]
            atype_count[atom] += 1
            pos += inc

    for i in range(atype_entries):
        count = atype_count[i]
        if count > 0:
            mean_orb_energies[i] /= count

    return mean_orb_energies


class Basissets:
    DEF2SVP = [[[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 1, 0, 0],
  [1, 0, 2, 0, 0],
  [2, 0, 2, 1, -1],
  [3, 0, 2, 1, 0],
  [4, 0, 2, 1, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 1, 0, 0],
  [1, 0, 2, 0, 0],
  [2, 0, 3, 0, 0],
  [3, 0, 2, 1, -1],
  [4, 0, 2, 1, 0],
  [5, 0, 2, 1, 1],
  [6, 0, 3, 1, -1],
  [7, 0, 3, 1, 0],
  [8, 0, 3, 1, 1],
  [9, 0, 3, 2, -2],
  [10, 0, 3, 2, -1],
  [11, 0, 3, 2, 0],
  [12, 0, 3, 2, 1],
  [13, 0, 3, 2, 2]],
 [[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]],
 [[0, 0, 1, 0, 0],
  [1, 0, 2, 0, 0],
  [2, 0, 3, 0, 0],
  [3, 0, 2, 1, -1],
  [4, 0, 2, 1, 0],
  [5, 0, 2, 1, 1],
  [6, 0, 3, 1, -1],
  [7, 0, 3, 1, 0],
  [8, 0, 3, 1, 1],
  [9, 0, 3, 2, -2],
  [10, 0, 3, 2, -1],
  [11, 0, 3, 2, 0],
  [12, 0, 3, 2, 1],
  [13, 0, 3, 2, 2]]]