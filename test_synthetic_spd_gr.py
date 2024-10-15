
import argparse
import torch

from geoopt import ManifoldParameter, Stiefel
from manifolds import SymmetricPositiveDefiniteMod, Grassmann
from geoopt.linalg import sym_invm, sym_sqrtm, sym_inv_sqrtm2
from scipy.linalg import solve_continuous_lyapunov
import numpy as np
import time
from utils import autograd, compute_hypergrad2
import os
from optimizer import RHGDstep
import pickle
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def true_hessinv(loss_lower, hparams, params, data_lower, tangents):
    W = hparams[0]
    M = params[0]
    Minvhalf, Mhalf = sym_inv_sqrtm2(M)
    G = (tangents[0].T + tangents[0]) # there is a 2 multiplied
    # assert torch.allclose()
    A = (X.T @ X)
    B = (W @ (Y.T @ Y) @ W.T) + lam* torch.eye(d, device=W.device)
    lhs = Mhalf @ A @ Mhalf + Minvhalf @ B @ Minvhalf
    lhs = 0.5*(lhs + lhs.T)
    rhs = Minvhalf @ G @ Minvhalf
    U = solve_continuous_lyapunov(lhs.cpu().detach().numpy(), rhs.cpu().detach().numpy())
    U = torch.from_numpy(U).float().to(A.device)
    U = (U + U.transpose(-1,-2))/2
    U = Mhalf @ U @ Mhalf
    assert torch.allclose(U @ A @ M + M @ A @ U + U @ sym_invm(M) @ B + B @ sym_invm(M) @ U, G, atol=1e-5)
    return [U]


def loss_lower(hparams, params, data=None):
    W = hparams[0]
    M = params[0]
    Minv = sym_invm(M)
    loss = (M * (X.T @ X)).sum() + (Minv * (W @ (Y.T @ Y) @ W.T)).sum() + lam * torch.trace(Minv)
    return loss


def loss_upper(hparams, params, data=None):
    W = hparams[0]
    M = params[0]
    # Mhalf = sym_sqrtm(M)
    # loss = -0.5 * torch.trace(W.T @ (Mhalf @ (X.T @ X) @ Mhalf) @ W)
    loss = -torch.trace(M @ X.T @ Y @ W.T)
    # loss = torch.norm(X @ M - Y @ W.T, p='fro')**2
    return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eta_x', type=float, default=0.5)
    parser.add_argument('--eta_y', type=float, default=0.5)
    parser.add_argument('--lower_iter', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hygrad_opt', type=str, default='ns', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--ns_gamma', type=float, default=0.1)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--compute_hg_error', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.seed)
    print(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    n = 100
    d = 50
    r = 20
    lam = 0.01

    # base setting
    setting = 'base'
    setting = 'liter50'
    setting = 'ns_aba'
    args.eta_x = 0.5
    args.eta_y = 0.5
    args.lower_iter = 20
    args.ns_gamma = 1.
    args.ns_iter = 50
    args.cg_iter = 50
    args.cg_gamma = 0.
    if setting == 'liter50':
        args.lower_iter = 50
    elif setting == 'ns_aba':
        args.hygrad_opt = 'ns'
        args.lower_iter = 50
        args.ns_iter = 200
        args.ns_gamma = 0.5


    X = torch.randn(n, d, device=device)
    X = X/torch.norm(X)
    Y = torch.randn(n, r, device=device)
    Y = Y/torch.norm(Y)

    spd = SymmetricPositiveDefiniteMod()
    st = Stiefel(canonical=False)
    hparams = [ManifoldParameter(st.random(d,r, device=device), manifold=st)]
    params = [ManifoldParameter(torch.eye(d, device=device), manifold=spd)]
    mfd_params = [spd]

    # for ep in range(args.epoch):
    #     step_start_time = time.time()
    #
    #     # lower level update (depending on whether we use ad)
    #     for ii in range(200):
    #         if args.hygrad_opt == 'ad':
    #             grad = autograd(loss_lower(hparams, params), params, create_graph=True)
    #             rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
    #             params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
    #             # print(f"Loss {loss_lower(hparams, params):.4f}")
    #             # egrad = autograd(loss_upper(hparams, params), hparams)
    #         else:
    #             grad = autograd(loss_lower(hparams, params), params)
    #             with torch.no_grad():
    #                 for param, egrad in zip(params, grad):
    #                     rgrad = param.manifold.egrad2rgrad(param, egrad)
    #                     new_param = param.manifold.retr(param, -args.eta_y * rgrad)
    #                     param.copy_(new_param)
    #         # with torch.no_grad():
    #         #     print(loss_lower(hparams,params))
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
    #     hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params, option='ns', ns_gamma=0.2, ns_iter=500)
    #
    #     # deactivate the computational path
    #     if args.hygrad_opt == 'ad':
    #         # params = [param.detach().clone().requires_grad_(True) for param in params]
    #         params = [ManifoldParameter(param.detach().clone(), manifold=mfd) for mfd, param in zip(mfd_params, params)]
    #
    #     true_hg = compute_hypergrad2(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv)
    #     print(torch.norm(hypergrad[0] - true_hg[0]))
    #
    #     with torch.no_grad():
    #         for hparam, hg in zip(hparams, hypergrad):
    #             new_hparam = hparam.manifold.retr(hparam, - args.eta_x * hg)
    #             hparam.copy_(new_hparam)
    #
    #         print(f"Epoch {ep}: "
    #               f"loss upper: {loss_upper(hparams, params).item():.4f}, "
    #               f"hypergrad norm: {hparams[0].manifold.inner(hparams[0], hypergrad[0]).item():.4f}")
    #
    #     step_time = time.time() - step_start_time


    # initial run
    for ii in range(args.lower_iter):
        if args.hygrad_opt == 'ad':
            grad = autograd(loss_lower(hparams, params, None), params, create_graph=True)
            rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
            params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
        else:
            grad = autograd(loss_lower(hparams, params, None), params)
            with torch.no_grad():
                for param, egrad in zip(params, grad):
                    rgrad = param.manifold.egrad2rgrad(param, egrad)
                    new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                    param.copy_(new_param)

    params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd, p in zip(mfd_params, params)]

    true_hg = compute_hypergrad2(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv,
                                 cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                 ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

    hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                  option=args.hygrad_opt, true_hessinv=true_hessinv,
                                   cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                  ns_gamma=args.ns_gamma , ns_iter=args.ns_iter)

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
    loss_u_all = [loss_upper(hparams, params).item()]
    hg_norm_all = [hgradnorm]
    runtime = [0]
    hg_error_all = [hg_error]
    print(f"Epoch {0}: "
          f"loss upper: {loss_u_all[-1]:.4f}, "
          f"hypergrad norm: {hgradnorm:.4e},"
          f"hg error: {hg_error_all[-1]:.4e}")

    # main run
    for ep in range(1,args.epoch+1):

        hparams, params, loss_u, hgradnorm, step_time, hg_error = RHGDstep(loss_lower, loss_upper, hparams, params, args,
                                                                  data=None, true_hessinv=true_hessinv)

        loss_u_all.append(loss_u)
        runtime.append(step_time)
        hg_error_all.append(hg_error)
        hg_norm_all.append(hgradnorm)
        epochs_all.append(ep)

        print(f"Epoch {ep}: "
              f"loss upper: {loss_u:.4f}, "
              f"hypergrad norm: {hgradnorm:.4e},"
              f"hg error: {hg_error:.4e}")

    res_folder = '/results/'

    stats = {'epochs': np.array(epochs_all), 'runtime': np.array(runtime), "loss_upper": np.array(loss_u_all),
             "hg_error": np.array(hg_error_all), 'hg_norm': np.array(hg_norm_all)}

    if setting == 'ns_aba':
        filename = 'syn_' + args.hygrad_opt + '_' + str(args.ns_iter) + '_' + str(args.ns_gamma) + '.pickle'
    else:
        filename = 'syn_' + args.hygrad_opt + '_' + setting + '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)








