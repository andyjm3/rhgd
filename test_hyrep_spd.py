import torch
from torch import nn
import geoopt
import numpy as np

from geoopt import linalg, ManifoldParameter
from geoopt import SymmetricPositiveDefinite
from geoopt import Stiefel, Euclidean
import time

from utils import autograd, compute_hypergrad2, compute_jvp, batch_egrad2rgrad
from utils import ts_conjugate_gradient
from manifolds import EuclideanMod
import argparse

from optimizer import RHGDstep, RSHGDstep


import pickle

## Euclidean ndim important
## ns approximation sensitive to conditioning of the Hessian and need to take very long step

def vec(X):
    """Reshape a symmetric matrix into a vector by extracting its upper-triangular part"""
    d = X.shape[-1]
    return X[..., torch.triu_indices(d, d)[0], torch.triu_indices(d, d)[1]]
    # return X.view(-1, d*d)


# class SPDShallow(nn.Module):
#     def __init__(self, in_size, hid_size):
#         super().__init__()
#         self.in_size = in_size
#         self.hid_size = hid_size
#         self.out_size = int(r * (r + 1) / 2)
#         self.stiefel = Stiefel()
#         self.spd = SymmetricPositiveDefinite()
#         self.gamma = nn.Parameter(torch.empty(self.out_size, ))
#         self.W = geoopt.ManifoldParameter(
#             torch.empty(in_size, hid_size), manifold=self.stiefel
#         )
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.gamma.set_(torch.randn(self.out_size, ))
#         self.W.set_(self.stiefel.random(self.in_size, self.hid_size))
#
#     def forward(self, A):
#         """
#         :param A: [n_data, d,d]
#         :return:
#         """
#         Z = torch.einsum("ij,bjk->bik", self.W.transpose(-1, -2), torch.einsum("bij,jk->bik", A, self.W))
#         Z = linalg.sym_logm(Z)
#         Z = torch.einsum("bk,k->b", vec(Z), self.gamma)
#         return Z


def shallowSPDnet(hparams, params, A):
    W = hparams[0]
    gamma = params[0]
    Z = W.transpose(-1, -2) @ A @ W
    Z = linalg.sym_logm(Z)
    Z = vec(Z)
    return Z


# def true_hypergrad(loss_lower, loss_upper, hparams, params):
#     """
#     Compute true hypergradient in closed form (for hessian inverse)
#     :return:
#     """
#     gamma = params[0]
#
#     egradfx = autograd(loss_upper(hparams, params), hparams)
#     rgradfx = batch_egrad2rgrad(hparams, egradfx)
#
#     predg = shallowSPDnet(hparams, params, data_lower)
#     predf = shallowSPDnet(hparams, params, data_upper)
#     egradf = (predf.transpose(-1, -2) @ (predf @ gamma - yval)) / data_upper.shape[0]
#     ehessg = (predg.transpose(-1, -2) @ predg) / data_lower.shape[0] + lam * torch.eye(predg.shape[1], device=device)
#     hessinvgrad = [torch.linalg.solve(ehessg, egradf)]
#     GxyHessinv = compute_jvp(loss_lower, hparams, params, hessinvgrad)
#
#     gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, GxyHessinv)]
#
#     return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]


# def true_hessinv(loss_lower, hparams, params, data_lower, tangents):
#     # gamma = params[0]
#     data_X, data_y = data_lower
#     predg = shallowSPDnet(hparams, params, data_X)
#     # predf = shallowSPDnet(hparams, params, data_upper)
#     # egradf = (predf.transpose(-1, -2) @ (predf @ gamma - yval)) / data_upper.shape[0]
#     ehessg = (predg.transpose(-1, -2) @ predg) / Atr.shape[0] + lam * torch.eye(predg.shape[1], device=device)
#     hessinvgrad = [torch.linalg.solve(ehessg, tangents[0])]
#     return hessinvgrad
#
#
# def loss_lower(hparams, params):
#     gamma = params[0]
#     pred = shallowSPDnet(hparams, params, Atr)
#     loss = 0.5 * torch.norm(pred @ gamma - ytr) ** 2 / Atr.shape[0] + 0.5 * lam * torch.norm(gamma) ** 2
#     return loss
#
#
# def loss_upper(hparams, params):
#     gamma = params[0]
#     pred = shallowSPDnet(hparams, params, Aval)
#     loss = 0.5 * torch.norm(pred @ gamma - yval) ** 2 / Aval.shape[0]
#     return loss




def loss_lower(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = shallowSPDnet(hparams, params, data_X)
    loss = 0.5 * torch.norm(pred @ gamma - data_y) ** 2 / data_X.shape[0] + 0.5 * lam * torch.norm(gamma) ** 2
    return loss


def loss_upper(hparams, params, data):
    data_X, data_y = data
    gamma = params[0]
    pred = shallowSPDnet(hparams, params, data_X)
    loss = 0.5 * torch.norm(pred @ gamma - data_y) ** 2 / data_X.shape[0]
    return loss



def true_hessinv(loss_lower, hparams, params, data_lower, tangents):
    # gamma = params[0]
    data_X, data_y = data_lower
    predg = shallowSPDnet(hparams, params, data_X)
    # predf = shallowSPDnet(hparams, params, data_upper)
    # egradf = (predf.transpose(-1, -2) @ (predf @ gamma - yval)) / data_upper.shape[0]
    ehessg = (predg.transpose(-1, -2) @ predg) / data_X.shape[0] + lam * torch.eye(predg.shape[1], device=device)
    hessinvgrad = [torch.linalg.solve(ehessg, tangents[0])]
    return hessinvgrad


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eta_x', type=float, default=0.05)
    parser.add_argument('--eta_y', type=float, default=0.05)
    parser.add_argument('--lower_iter', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--hygrad_opt', type=str, default='hinv', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--ns_gamma', type=float, default=0.05)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--cg_gamma', type=float, default=0.)
    parser.add_argument('--cg_iter', type=int, default=50)
    parser.add_argument('--compute_hg_error', type=bool, default=True)
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=9)
    args = parser.parse_args()

    # set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.seed)
    print(device)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


    # init
    mfd = SymmetricPositiveDefinite()
    N = 200
    d = 50
    r = 10
    lam = 0.1

    if args.stochastic:
        args.eta_x = 0.01
        args.eta_y = 0.01

    # generate A matrices
    A = mfd.random(N, d, d, device=device)  # N dxd SPD matrices
    # CN = 10
    # D = torch.diag_embed((CN - 1) * torch.rand(N,d, device=device) + 1)
    # U = Stiefel().random(N,d,d, device=device)
    # A = U.transpose(-1,-2) @ D @ U
    Wstar = Stiefel().random(d, r, device=device)
    gammastar = torch.randn(int(r * (r + 1) / 2), device=device)
    # gammastar = torch.randn(r*r, device=device)

    y = Wstar.transpose(-1, -2) @ A @ Wstar
    y = linalg.sym_logm(y)
    y = vec(y) @ gammastar + torch.randn(N, device=device)

    Atr = A[:int(N/2)]
    Aval = A[int(N/2):]
    ytr = y[:int(N/2)]
    yval = y[int(N/2):]

    euclidean = EuclideanMod(ndim=1)
    stiefel = Stiefel(canonical=False)

    params = [geoopt.ManifoldParameter(euclidean.random(int(r * (r + 1) / 2), device=device), manifold=euclidean)]
    # hparams = [geoopt.ManifoldParameter(stiefel.random(d, r, device=device), manifold=stiefel)]
    hparams = [geoopt.ManifoldParameter(torch.eye(d, r, device=device), manifold=stiefel)]
    mfd_params = [param.manifold for param in params]

    # for ep in range(epoch):
    #     step_start_time = time.time()
    #
    #     # lower level update (depending on whether we use ad)
    #     for ii in range(S):
    #         if hg_opt == 'ad':
    #             grad = autograd(loss_lower(hparams, params), params, create_graph=True)
    #             rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
    #             params = [mfd.retr(param, - eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
    #             # print(f"Loss {loss_lower(hparams, params):.4f}")
    #             # egrad = autograd(loss_upper(hparams, params), hparams)
    #         else:
    #             grad = autograd(loss_lower(hparams, params), params)
    #             with torch.no_grad():
    #                 for param, egrad in zip(params, grad):
    #                     rgrad = param.manifold.egrad2rgrad(param, egrad)
    #                     new_param = param.manifold.retr(param, -eta_y * rgrad)
    #                     param.copy_(new_param)
    #
    #     # compute hypergrad estimate
    #     # hypergrad = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='cg')
    #     #
    #     # true_hg = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv)
    #     # # assertion = [(hp.manifold._check_vector_on_tangent(hp,hg)) for hp, hg in zip(hparams, hypergrad)]
    #     #
    #     # ns_hg = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='ns',
    #     #                           ns_gamma=0.01, ns_iter=1000)
    #     # true_hg = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv)
    #     hypergrad = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='cg')
    #
    #     # deactivate the computational path
    #     if hg_opt == 'ad':
    #         # params = [param.detach().clone().requires_grad_(True) for param in params]
    #         params = [geoopt.ManifoldParameter(param.detach().clone(), manifold=mfd) for mfd, param in zip(mfd_params, params)]
    #
    #     true_hg = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv)
    #     # print(hypergrad[0] - true_hg[0])
    #
    #     with torch.no_grad():
    #         for hparam, hg in zip(hparams, hypergrad):
    #             new_hparam = hparam.manifold.retr(hparam, - eta_x * hg)
    #             hparam.copy_(new_hparam)
    #
    #         print(f"Epoch {ep}: "
    #               f"loss upper: {loss_upper(hparams, params).item():.4f}, "
    #               f"hypergrad norm: {hparams[0].manifold.inner(hparams[0], hypergrad[0]).item():.2f}")
    #
    #     step_time = time.time() - step_start_time

    # RHGD(loss_lower, loss_upper, hparams, params, args, true_hessinv=true_hessinv)

    data_lower = [Atr, ytr]
    data_upper = [Aval, yval]

    # true_hg = compute_hypergrad2(loss_lower_stoc, loss_upper_stoc, hparams, params, option='hinv', true_hessinv=true_hessinv,
    #                              data_lower=data_lower, data_upper=data_upper,
    #                              cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
    #                              ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)
    #
    # hypergrad = compute_hypergrad2(loss_lower_stoc, loss_upper_stoc, hparams, params,
    #                                data_lower=data_lower, data_upper=data_upper,
    #                                option=args.hygrad_opt, true_hessinv=true_hessinv,
    #                                cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
    #                                ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

    # initial run
    for ii in range(args.lower_iter):
        if args.hygrad_opt == 'ad':
            grad = autograd(loss_lower(hparams, params, data_lower), params, create_graph=True)
            rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
            params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
        else:
            grad = autograd(loss_lower(hparams, params, data_lower), params)
            with torch.no_grad():
                for param, egrad in zip(params, grad):
                    rgrad = param.manifold.egrad2rgrad(param, egrad)
                    new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                    param.copy_(new_param)

    params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd, p in zip(mfd_params, params)]

    true_hg = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                 data_lower=data_lower, data_upper=data_upper,
                                 option='hinv', true_hessinv=true_hessinv,
                                 cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                 ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

    hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                   data_lower=data_lower, data_upper=data_upper,
                                   option=args.hygrad_opt, true_hessinv=true_hessinv,
                                   cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                   ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

    with torch.no_grad():
        hgradnorm = 0
        for hparam, hg in zip(hparams, hypergrad):
            hgradnorm += hparam.manifold.inner(hparam, hg).item() / len(hparams)
    hg_error = 0
    if not (args.hygrad_opt == 'hinv'):
        hg_error = [torch.sqrt(hp.manifold.inner(hp, hg - t_hg)).item() for hg, t_hg, hp in
                    zip(hypergrad, true_hg, hparams)]
        hg_error = torch.Tensor(hg_error).sum().item()
    epochs_all = [0]
    loss_u_all = [loss_upper(hparams, params, data_upper).item()]
    hg_norm_all = [hgradnorm]
    runtime = [0]
    hg_error_all = [hg_error]
    for ep in range(args.epoch):

        if args.stochastic:

            ind_lower = torch.randint(low=0, high=int(N / 2), size=(args.batch_size,))
            ind_upper = torch.randint(low=0, high=int(N / 2), size=(args.batch_size,))
            data_lower_sub = [data_lower[0][ind_lower], data_lower[1][ind_lower]]
            data_upper_sub = [data_upper[0][ind_upper], data_upper[1][ind_upper]]

            hparams, params, loss_u, hgradnorm, step_time = RSHGDstep(loss_lower, loss_upper, hparams, params,
                                                                       args, data=[data_lower_sub, data_upper_sub],
                                                                       true_hessinv=true_hessinv,
                                                                       )
        else:
            hparams, params, loss_u, hgradnorm, step_time, hg_error = RHGDstep(loss_lower, loss_upper, hparams, params, args,
                                                                               data=[data_lower, data_upper],
                                                                               true_hessinv=true_hessinv,
                                                                               )

        loss_u_all.append(loss_u)
        runtime.append(step_time)
        hg_error_all.append(hg_error)
        hg_norm_all.append(hgradnorm)
        epochs_all.append(ep)
        print(f"Epoch {ep}: "
              f"loss upper: {loss_u:.4e}, "
              f"hypergrad norm: {hgradnorm:.4e},"
              f"hg error: {hg_error:.4e}")

    stats = {'epochs': np.array(epochs_all), 'runtime': np.array(runtime), "loss_upper": np.array(loss_u_all),
             "hg_error": np.array(hg_error_all), 'hg_norm': np.array(hg_norm_all)}

    flag = 'full'
    if args.stochastic:
        flag = "sto"

    filename = 'shallow_hyrep' + '_' + args.hygrad_opt + '_' + flag +  '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

