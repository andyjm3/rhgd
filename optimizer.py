# a simple implementation of Riemannian hypergradient descent for bilevel optimization problem
import torch
import time
from utils import autograd, compute_jvp, batch_egrad2rgrad, compute_hypergrad2, compute_hypergrad_stoc
import geoopt
from utils import get_subset

from higher.optim import DifferentiableOptimizer, _add, _GroupedGradsType


import torch.optim.optimizer
from geoopt import ManifoldParameter, ManifoldTensor, Euclidean
from geoopt.optim.mixin import OptimMixin






def RHGDstep(loss_lower, loss_upper, hparams, params, args, data=None, true_hessinv=None):
    """
    A single step of Riemannian hypergradient descent

    :param loss_lower: loss for lower problem (with input hparams, params and data1)
    :param loss_upper: loss for upper problem (with input hparams, params and data2)
    :param data: tuple of data1, data2 if different, else just tuple of size one, or can be none if loss_lower/loss_upper
                 does not depend on data
    :param hparams: list of hyper-parameters (x), in form of geoopt,ManifoldParameter
    :param params: list of parameters (y), in form of geoopt.ManifoldParameter
    :param true_hessinv: a function that calls to return the true hessian inverse
    :param args: arguments including:
        :param eta_x: stepsize for x
        :param eta_y: stepsize for y
        :param lower_iter: number of iterations for the lower problem update
        :param epoch: number of epochs for the upper problem update
        :param hygrad_opt: hypergradient options: {hinv, ad, cg, ns}
    :return:
    """

    assert (isinstance(data, tuple) or isinstance(data, list) or data is None)
    if data is not None:
        if len(data) == 1:
            data_lower = data[0]
            data_upper = data[0]
        elif len(data) == 2:
            data_lower = data[0]
            data_upper = data[1]
    else:
        data_lower = data_upper = None

    def compute_hgradnorm():
        hgradnorm = 0
        for mfd, hparam, hg in zip(mfd_hparams, hparams, hypergrad):
            hgradnorm += mfd.inner(hparam, hg).item() / len(hparams)
        return hgradnorm

    mfd_params = [param.manifold for param in params]
    mfd_hparams = [hparam.manifold for hparam in hparams]
    # loss_u_all = [loss_upper(hparams, params).item()]
    # hgradnorm_all = [compute_hgradnorm()]
    # time_all = [0]
    # loss_u_all = []
    # hgradnorm_all = []
    # time_all = []
    #
    step_start_time = time.time()

    # lower level update (depending on whether we use ad)
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

        # with torch.no_grad():
        #     print(loss_lower(hparams, params, data_lower))

    # compute hypergrad estimate
    hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                   data_lower=data_lower, data_upper=data_upper,
                                  option=args.hygrad_opt, true_hessinv=true_hessinv,
                                   cg_iter=args.cg_iter, cg_gamma=args.cg_gamma,
                                  ns_gamma=args.ns_gamma , ns_iter=args.ns_iter)

    params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for mfd,p in zip(mfd_params,params)]

    with torch.no_grad():
        if args.compute_hg_error and (true_hessinv is not None) and (not args.hygrad_opt == 'hinv'):
            with torch.enable_grad():
                true_hg = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                             data_lower=data_lower, data_upper=data_upper,
                                             option='hinv', true_hessinv=true_hessinv)
            hg_error = [torch.sqrt(hp.manifold.inner(hp, hg-t_hg)).item() for hg, t_hg, hp in zip(hypergrad, true_hg, hparams)]
            hg_error = torch.Tensor(hg_error).sum().item()
        else:
            hg_error = 0

    with torch.no_grad():
        for hparam, hg in zip(hparams, hypergrad):
            new_hparam = hparam.manifold.retr(hparam, - args.eta_x * hg)
            hparam.copy_(new_hparam)

        loss_u = loss_upper(hparams, params, data_upper).item()
        hgradnorm = compute_hgradnorm()


    step_time = time.time() - step_start_time

    # deactivate the computational path
    hparams = [geoopt.ManifoldParameter(hparam.detach().clone(), manifold=mfd) for mfd, hparam in
               zip(mfd_hparams, hparams)]
    params = [geoopt.ManifoldParameter(param.detach().clone(), manifold=mfd) for mfd, param in zip(mfd_params, params)]

    return hparams, params, loss_u, hgradnorm, step_time, hg_error



def RSHGDstep(loss_lower, loss_upper, hparams, params, args, data=None, true_hessinv=None):
    """
    A single step of Riemannian stochastic hypergradient descent

    :param loss_lower: loss for lower problem (with input hparams, params and data1)
    :param loss_upper: loss for upper problem (with input hparams, params and data2)
    :param data: tuple of data1, data2 if different, else just tuple of size one, or can be none if loss_lower/loss_upper
                 does not depend on data
    :param hparams: list of hyper-parameters (x), in form of geoopt,ManifoldParameter
    :param params: list of parameters (y), in form of geoopt.ManifoldParameter
    :param true_hessinv: a function that calls to return the true hessian inverse
    :param args: arguments including:
        :param eta_x: stepsize for x
        :param eta_y: stepsize for y
        :param lower_iter: number of iterations for the lower problem update
        :param epoch: number of epochs for the upper problem update
        :param hygrad_opt: hypergradient options: {hinv, ad, cg, ns}
    :return:
    """
    assert (isinstance(data, tuple) or isinstance(data, list) or data is None)
    if data is not None:
        if len(data) == 1:
            data_lower = data[0]
            data_upper = data[0]
        elif len(data) == 2:
            data_lower = data[0]
            data_upper = data[1]
    else:
        data_lower = data_upper = None

    def compute_hgradnorm():
        hgradnorm = 0
        for mfd, hparam, hg in zip(mfd_hparams, hparams, hypergrad):
            hgradnorm += mfd.inner(hparam, hg).item() / len(hparams)
        return hgradnorm

    mfd_params = [param.manifold for param in params]
    mfd_hparams = [hparam.manifold for hparam in hparams]
    # loss_u_all = [loss_upper(hparams, params).item()]
    # hgradnorm_all = [compute_hgradnorm()]
    # time_all = [0]
    # loss_u_all = []
    # hgradnorm_all = []
    # time_all = []
    step_start_time = time.time()

    # n_lower = data_lower.shape[0]
    # n_upper = data_upper.shape[0]

    # lower level update (depending on whether we use ad)
    for ii in range(args.lower_iter):
        if args.hygrad_opt == 'ad':
            grad = autograd(loss_lower(hparams, params, data_lower), params, create_graph=True)
            rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
            params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
        else:
            grad = autograd(loss_lower(hparams, params, get_subset(data_lower, 0.333, True)), params)
            with torch.no_grad():
                for param, egrad in zip(params, grad):
                    rgrad = param.manifold.egrad2rgrad(param, egrad)
                    new_param = param.manifold.retr(param, -args.eta_y * rgrad)
                    param.copy_(new_param)

    # compute hypergrad estimate
    hypergrad = compute_hypergrad_stoc(loss_lower, loss_upper, hparams, params,
                                   data_lower=get_subset(data_lower, 0.333, False),
                                   data_upper=data_upper,
                                  option=args.hygrad_opt, true_hessinv=true_hessinv,
                                  ns_gamma=args.ns_gamma , ns_iter=args.ns_iter)


    # if args.hygrad_opt == 'ad':


    # true_hg = compute_hypergrad(loss_lower, loss_upper, hparams, params, option='hinv', true_hessinv=true_hessinv)
    # print(hypergrad[0] - true_hg[0])

    with torch.no_grad():
        for hparam, hg in zip(hparams, hypergrad):
            new_hparam = hparam.manifold.retr(hparam, - args.eta_x * hg)
            hparam.copy_(new_hparam)

        loss_u = loss_upper(hparams, params, data_upper).item()
        hgradnorm = compute_hgradnorm()

    step_time = time.time() - step_start_time

    # deactivate the computational path
    hparams = [geoopt.ManifoldParameter(hparam.detach().clone(), manifold=mfd) for mfd, hparam in
               zip(mfd_hparams, hparams)]
    params = [geoopt.ManifoldParameter(param.detach().clone(), manifold=mfd) for mfd, param in zip(mfd_params, params)]

    return hparams, params, loss_u, hgradnorm, step_time




