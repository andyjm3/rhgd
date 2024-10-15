import os
import random
import torch
from torch import nn
import numpy as np
from torch.autograd import Function as F
from torch.utils import data
from geoopt import linalg
import argparse

from geoopt import Euclidean, Stiefel, ManifoldParameter
from manifolds import EuclideanMod
from utils import autograd

from scipy.io import loadmat

from optimizer import RHGDstep

import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def vec(X):
    """Reshape a symmetric matrix into a vector by extracting its upper-triangular part"""
    d = X.shape[-1]
    return X[..., torch.triu_indices(d, d)[0], torch.triu_indices(d, d)[1]]

# https://github.com/rheasukthanker/SPDNetNAS/blob/main/spd/functional.py#L493
class Re_op():
    """ Relu function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).float()

def add_id_matrix(P, alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    P = P + alpha * P.trace() * torch.eye(
        P.shape[-1], dtype=P.dtype, device=P.device)
    return P

def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape
    U, S = torch.zeros_like(P, device=P.device), torch.zeros(batch_size,
                                                       channels,
                                                       n,
                                                       dtype=P.dtype,
                                                       device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if (eig_mode == 'eig'):
                s, U[i, j] = torch.eig(P[i, j], True)
                S[i, j] = s[:, 0]
            elif (eig_mode == 'svd'):
                U[i, j], S[i, j], _ = torch.svd(add_id_matrix(P[i, j], 1e-5))
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  #batch size,channel depth,dimension
    Q = torch.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):  #can vectorize
        for j in range(channels):  #can vectorize
            Q[i, j] = P[i, j].diag()
    return Q

def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''

    #print("Correct back prop")
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0
    L[L == np.inf] = 0
    L[torch.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp



class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)






RADAR_CLASSES = 3
pval = 0.25  # validation percentage
ptest = 0.25  # test percentage


class DatasetRadar(data.Dataset):

    def __init__(self, path, names):
        self._path = path
        self._names = names

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        x = np.load(self._path + self._names[item])
        x = np.concatenate((x.real[:, None], x.imag[:, None]), axis=1).T
        x = torch.from_numpy(x)
        y = int(self._names[item].split('.')[0].split('_')[-1])
        y = torch.from_numpy(np.array(y))
        return x.float(), y.long()


class DataLoaderRadar:

    def __init__(self, data_path, pval, batch_size):
        for filenames in os.walk(data_path):
            names = sorted(filenames[2])
        random.Random(0).shuffle(names)
        N_val = int(pval * len(names))
        N_test = int(ptest * len(names))
        N_train = len(names) - N_test - N_val
        train_set = DatasetRadar(
            data_path, names[N_val + N_test:int(N_train) + N_test + N_val])
        test_set = DatasetRadar(data_path, names[:N_test])
        val_set = DatasetRadar(data_path, names[N_test:N_test + N_val])
        self._train_generator = data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=True)
        self._test_generator = data.DataLoader(test_set,
                                               batch_size=batch_size,
                                               shuffle=False)
        self._val_generator = data.DataLoader(val_set,
                                              batch_size=batch_size,
                                              shuffle=False)



def SPDnet(hparams, params, A):
    # A needs to be [batch, d, d]
    W1 = hparams[0]
    W2 = hparams[1]
    gamma = params[0]
    # 1st layer
    Z1 = W1.transpose(-1, -2) @ A @ W1
    A1 = ReEig().apply(Z1.unsqueeze(1))
    # 2nd layer
    Z2 = W2.transpose(-1,-2) @ A1.squeeze(1) @ W2
    Z = linalg.sym_logm(Z2)
    Z = vec(Z).squeeze()
    return Z


def FFnet(hparams, params, A):
    W1 = hparams[0]
    W2 = hparams[1]
    gamma = params[0]
    # 1st layer
    Z1 = vec(A) @ W1
    A1 = nn.functional.relu(Z1)
    # 2nd layer
    Z2 = A1 @ W2
    return Z2


def loss_lower_mfd(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = SPDnet(hparams, params, data_X)
    loss = nn.functional.cross_entropy(pred @ gamma, data_y) + 0.5 * lam * torch.norm(gamma) ** 2
    return loss

def loss_upper_mfd(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = SPDnet(hparams, params, data_X)
    loss = nn.functional.cross_entropy(pred @ gamma, data_y)
    return loss



def loss_lower_euc(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = FFnet(hparams, params, data_X)
    loss = nn.functional.cross_entropy(pred @ gamma, data_y) + 0.5 * lam * torch.norm(gamma) ** 2
    return loss


def loss_upper_euc(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = FFnet(hparams, params, data_X)
    loss = nn.functional.cross_entropy(pred @ gamma, data_y)
    return loss





def compute_acc(network, hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = network(hparams, params, data_X)
    logit = pred @ gamma
    y_pred = torch.argmax(logit, dim=1)
    acc = (y_pred == data_y).sum() / data_y.shape[0]
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta_x', type=float, default=0.1)
    parser.add_argument('--eta_y', type=float, default=0.1)
    parser.add_argument('--lower_iter', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--hygrad_opt', type=str, default='ad', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--ns_gamma', type=float, default=0.05)
    parser.add_argument('--ns_iter', type=int, default=30)
    parser.add_argument('--cg_gamma', type=float, default=0.)
    parser.add_argument('--cg_iter', type=int, default=50)
    parser.add_argument('--compute_hg_error', type=bool, default=False)
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.seed)
    print(device)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    lam = 0.1
    euc = False
    runs = 10

    # data_path = 'data/radar/'
    # lam = 0.1
    #
    # data_loader = DataLoaderRadar(data_path, pval, args.batch_size)
    # train_loader = data_loader._train_generator
    # valid_loader = data_loader._val_generator
    # test_loader = data_loader._test_generator

    data = loadmat('data/ETH_data.mat')

    X = data['data']
    y = data['labels']
    y = y - 1

    num_class = 8

    data_X = np.zeros((X.shape[-1], 100, 100))
    eps = 1e-3
    for s in range(X.shape[-1]):
        cov_mat = np.cov(X[...,s]) + eps * np.eye(100)
        data_X[s,...] = cov_mat

    data_X = torch.from_numpy(data_X).float()
    data_y = torch.from_numpy(y).long()



    acc_all_runs = []
    loss_all_runs = []
    runtime_all_runs = []
    for run in range(runs):
        train_X = []
        test_X = []
        train_y = []
        test_y = []
        for c in range(num_class):
            data_sub = data_X[y == c]
            idx_all =  torch.randperm(data_sub.shape[0])
            train_idx = idx_all[:int(idx_all.shape[0]/2)]
            test_idx = idx_all[int(idx_all.shape[0]/2):]
            train_X.append(data_sub[train_idx])
            test_X.append(data_sub[test_idx])

            train_y.extend([c]*train_idx.shape[0])
            test_y.extend([c]*test_idx.shape[0])

        train_X = torch.cat(train_X, dim=0)
        test_X = torch.cat(test_X, dim=0)
        train_y = torch.tensor(train_y, dtype=torch.long)
        test_y = torch.tensor(test_y, dtype=torch.long)

        idx_train = torch.randperm(train_y.shape[0])
        train_X = (train_X[idx_train]).to(device)
        train_y = train_y[idx_train].to(device)
        idx_test = torch.randperm(test_y.shape[0])
        test_X = test_X[idx_test].to(device)
        test_y = test_y[idx_test].to(device)


        euclidean = EuclideanMod(ndim=2)
        stiefel = Stiefel(canonical=False)

        d = 100
        d1 = 20
        d2 = 5

        if euc:
            print('initialization with euclidean network')
            hparams = [ManifoldParameter(torch.randn(int(d * (d + 1) / 2), d1, device=device), manifold=euclidean),
                       ManifoldParameter(torch.randn(d1, int(d2 * (d2 + 1) / 2), device=device), manifold=euclidean)]
            params = [
                ManifoldParameter(euclidean.random(int(d2 * (d2 + 1) / 2), num_class, device=device), manifold=euclidean)]
            loss_upper = loss_upper_euc
            loss_lower = loss_lower_euc
            network = FFnet
        else:
            print('initialization with manifold network')
            hparams = [ManifoldParameter(torch.eye(d, d1, device=device), manifold=stiefel),
                       ManifoldParameter(torch.eye(d1, d2, device=device), manifold=stiefel)]
            params = [ManifoldParameter(euclidean.random(int(d2 * (d2 + 1) / 2), num_class, device=device), manifold=euclidean)]
            mfd_params = [param.manifold for param in params]
            loss_upper = loss_upper_mfd
            loss_lower = loss_lower_mfd
            network = SPDnet


        data_lower = [train_X, train_y]
        data_upper = [test_X, test_y]

        epochs_all = [0]
        loss_u_all = [loss_upper(hparams, params, data_upper).item()]
        runtime = [0]
        acc_all = [compute_acc(network, hparams, params, data_upper).item()]

        for ep in range(1, args.epoch+1):

            hparams, params, loss_u, hgradnorm, step_time, hg_error = RHGDstep(loss_lower, loss_upper, hparams, params,
                                                                               args,
                                                                               data=[data_lower, data_upper],
                                                                               )
            with torch.no_grad():
                val_acc = compute_acc(network, hparams, params, data_upper)

            loss_u_all.append(loss_u)
            runtime.append(step_time)
            epochs_all.append(ep)
            acc_all.append(val_acc.item())

            print(f"Epoch {ep}: "
                  f"loss upper: {loss_u:.4e}, "
                  f"Val acc: {val_acc*100:.2f}")

        acc_all_runs.append(torch.tensor(acc_all).unsqueeze(0))
        loss_all_runs.append(torch.tensor(loss_u_all).unsqueeze(0))
        runtime_all_runs.append(torch.tensor(runtime).unsqueeze(0))


    stats = {'runtime': torch.cat(runtime_all_runs, dim=0), 'loss_upper': torch.cat(loss_all_runs, dim=0),
             'accuracy': torch.cat(acc_all_runs, dim=0)}

    filename = 'hyrep_spd_' + args.hygrad_opt + '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)








