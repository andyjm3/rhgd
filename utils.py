import torch
from geoopt.tensor import ManifoldParameter, ManifoldTensor
import math

# deal with list of parameters
def dot(tensors_one, tensors_two):
    """List of tensors in tensors_one, tensors_two"""
    ret = tensors_one[0].new_zeros((1, ), requires_grad=True)
    for t1, t2 in zip(tensors_one, tensors_two):
        ret = ret + torch.sum(t1 * t2)
    return ret

def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]


def norm(tensors):
    return math.sqrt(sum([torch.sum(tensor ** 2).item() for tensor in tensors]))

@torch.no_grad()
def conjugate_gradient(_hvp, b, maxiter=100, tol=1e-3, lam=0.0, use_cache=0, negcur=False, eps=-1e-7):
    """
    Minimize 0.5 x^T H^T H x - b^T H x, where H is symmetric
    Args:
        _hvp (function): hessian vector product, only takes a sequence of tensors as input
        b (sequence of tensors): b
        maxiter (int): number of iterations
        lam (float): regularization constant to avoid singularity of hessian. lam can be positive, zero or negative
    (Q = H^T H)
    """
    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)

        outputs = [xx + lam * yy for xx, yy in zip(outputs, inputs)]

        return outputs

    with torch.enable_grad():
        Hb = hvp(b)

    # zero initialization
    xxs = [hb.new_zeros(hb.size()) for hb in Hb]
    ggs = [- hb.clone().detach() for hb in Hb]
    dds = [- hb.clone().detach() for hb in Hb]

    i = 0

    while True:
        i += 1

        with torch.enable_grad():
            Qdds = hvp(hvp(dds))
            # Qdds = hvp(dds)

        # print(dot(ggs, ggs))
        # print(norm(ggs))

        # if dot(ggs, ggs) < tol:
        if norm(ggs) < tol:
            break

        # one step steepest descent
        alpha = - dot(dds, ggs) / dot(dds, Qdds)
        xxs = [xx + alpha * dd for xx, dd in zip(xxs, dds)]

        # update gradient
        ggs = [gg + alpha * Qdd for gg, Qdd in zip(ggs, Qdds)]

        # compute the next conjugate direction
        beta = dot(ggs, Qdds) / dot(dds, Qdds)
        dds = [gg - beta * dd for gg, dd in zip(ggs, dds)]

        if maxiter is not None and i >= maxiter:
            break

    # print("# of conjugate steps {:d}".format(i))

    return xxs

@torch.no_grad()
def ts_conjugate_gradient(_hvp, b, base, v0=None, maxiter=200, tol=1e-10, lam=0.0, verbose=1):
    """
    Solve H[v] = b where H is a tangent space operator at base points,
    Solved via (Hreg^2)[v] = Hreg[b], where Hreg = H + lam id
    All are list of tensors!

    :param _hvp: H vector product (function that takes a list[Tensor] and output a list[Tensor])
    :param b: List[tangent vector]
    :param v0: initialization (default to be zero)
    :param base: base point (list[geoopt.ManifoldParameter])
    :param maxiter: maximum number of iteration
    :param tol: tol for residual norm
    :param lam: regularization strength to ensure p.d.
    :return:
    """

    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)
        outputs = [xx + lam * yy for xx, yy in zip(outputs, inputs)]
        return outputs

    def sumls(ls):
        out = 0
        for ll in ls:
            out += ll
        return out

    with torch.enable_grad():
        Hb = hvp(b)

    # init
    v = v0
    if v0 is None:
        v = [hb.new_zeros(hb.size()) for hb in Hb]
    r = [hb.clone().detach() for hb in Hb]
    p = [hb.clone().detach() for hb in Hb]

    rnormprev = [xx.manifold.inner(xx, rr, rr) for xx, rr in zip(base, r)]

    it = 0
    while True:
        with torch.enable_grad():
            HHp = hvp(hvp(p))
        alpha = [rn/xx.manifold.inner(xx, pp, hhpp) for rn, xx, pp, hhpp in zip(rnormprev, base, p, HHp)]
        v = [vv + aa * pp for vv, aa, pp in zip(v, alpha, p)]
        r = [rr - aa * hhpp for rr, aa, hhpp in zip(r, alpha, HHp)]
        rnorm = [xx.manifold.inner(xx, rr, rr) for xx, rr in zip(base, r)]
        if sumls(rnorm) < tol:
            if verbose:
                print(f"CG tol reached, break at iter {it}.")
            break
        elif it >= maxiter:
            if verbose:
                print(f"CG tol not reached! Break at max iteration with residual {sumls(rnorm):.4f}.")
            break
        beta = [rn/rnprev for rn, rnprev in zip(rnorm, rnormprev)]
        p = [rr + bb * pp for rr, bb, pp in zip(r, beta, p)]
        rnormprev = rnorm
        it += 1

    return v


def batch_egrad2rgrad(params, egrad):
    return [param.manifold.egrad2rgrad(param, eg) for param, eg in zip(params, egrad)]


def compute_jvp(loss, hparams, params, tangents):
    """
    Compute the cross derivative of loss(hparams, params), i.e., G_xy [tangents] where x is hparams, y is params
    :param loss:
    :param inputs: List[Tensors] of size hparams
    :param tangents: List[Tensors] of size params
    :return:
    """
    assert len(params) == len(tangents)
    def function(params):
        grad = autograd(loss(hparams, [params]), hparams, create_graph=True) # list of size hparams
        return tuple([hparam.manifold.egrad2rgrad(hparam, gg) for hparam, gg in zip(hparams, grad)])

    _, gradxy = torch.autograd.functional.jvp(function, tuple(params), tuple(tangents))

    return gradxy


# def compute_hypergrad(loss_lower, loss_upper, hparams, params, option='cg',
#                       true_hessinv=None,
#                       ns_iter=200, ns_gamma=0.01):
#     """hparams is x, params is y, loss_lower is g, loss_upper is f
#
#     # :param eta_lower: the stepsize for updating inner variables
#     # :param S: number of iterations for updating inner variables (set to be large for ad option)
#     :param option: the option for estimating hypergradient, ['cg', 'ns', 'hinv', 'ad']
#     :param true_hessinv: the function call to compute true hessian inverse (only valid when option = 'hinv'
#     :param ns_iter: number of iterations for estimating hypergradient using truncated neumann series
#     :param ns_gamma: gamma for truncated neumann series (need to be sufficiently small to ensure gamma*hess has norm bounded by 1)
#
#     """
#
#     if option == 'cg':
#         egradfy = autograd(loss_upper(hparams, params), params)
#         egradfx = autograd(loss_upper(hparams, params), hparams)
#         rgradfy = batch_egrad2rgrad(params, egradfy)
#         rgradfx = batch_egrad2rgrad(hparams, egradfx)
#
#         # hessinv grad
#         def rhess_prod(u):
#             egrad = autograd(loss_lower(hparams, params), params, create_graph=True)
#             ehess = autograd(dot(egrad, u), params)
#             out = []
#             with torch.no_grad():
#                 for idx, param in enumerate(params):
#                     out.append(param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
#             return out
#         Hinv_gy = ts_conjugate_gradient(rhess_prod, rgradfy, params, lam=0.)
#         gradgxy = compute_jvp(loss_lower, hparams, params, Hinv_gy)
#
#         # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
#         gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams,gradgxy)]
#
#         # print(rgradfx)
#         # print(rgradfy)
#         # print(Hinv_gy)
#         # print(gradgxy_proj)
#
#         return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]
#
#     elif option.lower() == 'hinv':
#         assert true_hessinv is not None
#         egradfy = autograd(loss_upper(hparams, params), params)
#         egradfx = autograd(loss_upper(hparams, params), hparams)
#         rgradfy = batch_egrad2rgrad(params, egradfy)
#         rgradfx = batch_egrad2rgrad(hparams, egradfx)
#
#         Hinv_gy = true_hessinv(loss_lower, hparams, params, rgradfy)
#         gradgxy = compute_jvp(loss_lower, hparams, params, Hinv_gy)
#
#         # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
#         gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]
#
#         return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]
#
#     elif option.lower() == 'ns':
#         egradfy = autograd(loss_upper(hparams, params), params)
#         egradfx = autograd(loss_upper(hparams, params), hparams)
#         rgradfy = batch_egrad2rgrad(params, egradfy)
#         rgradfx = batch_egrad2rgrad(hparams, egradfx)
#
#         def reg_rhess_prod(u):
#             egrad = autograd(loss_lower(hparams, params), params, create_graph=True)
#             ehess = autograd(dot(egrad, u), params)
#             out = []
#             with torch.no_grad():
#                 for idx, param in enumerate(params):
#                     out.append(u[idx] - ns_gamma * param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
#             return out
#
#         with torch.no_grad():
#             Hinv_gy_prev = [g.clone().detach() for g in rgradfy]
#             Hinv_gy = [g.clone().detach() for g in rgradfy]
#             for ins in range(ns_iter):
#                 with torch.enable_grad():
#                     Hinv_gy_new = reg_rhess_prod(Hinv_gy_prev)
#                 Hinv_gy = [hg + hg_new for hg, hg_new in zip(Hinv_gy, Hinv_gy_new)]
#                 Hinv_gy_prev = Hinv_gy_new
#             if ns_gamma > 0:
#                 Hinv_gy = [ns_gamma * hg for hg in Hinv_gy]
#
#         gradgxy = compute_jvp(loss_lower, hparams, params, Hinv_gy)
#         gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]
#
#         return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]
#
#     elif option.lower() == 'ad':
#         egradfy = autograd(loss_upper(hparams, params), hparams)
#         return [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, egradfy)]
#
#     else:
#         raise("option not implemented.")




    # test cg
    # import geoopt
    # from manifolds import SymmetricPositiveDefiniteMod
    # import torch
    # from geoopt import linalg

    # torch.manual_seed(42)






################################################################

def compute_jvp2(loss, hparams, params, data, tangents):
    """
    Compute the cross derivative of loss(hparams, params), i.e., G_xy [tangents] where x is hparams, y is params
    :param loss:
    :param inputs: List[Tensors] of size hparams
    :param tangents: List[Tensors] of size params
    :return:
    """
    assert len(params) == len(tangents)
    def function(*params):
        grad = autograd(loss(hparams, list(params), data), hparams, create_graph=True) # list of size hparams
        return tuple([hparam.manifold.egrad2rgrad(hparam, gg) for hparam, gg in zip(hparams, grad)])

    _, gradxy = torch.autograd.functional.jvp(function, tuple(params), tuple(tangents))

    return gradxy


def compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                       data_lower =None, data_upper=None,
                       option='cg',
                      true_hessinv=None,
                       cg_iter = 200, cg_gamma = 0.,
                      ns_iter=200, ns_gamma=0.01):
    """hparams is x, params is y, loss_lower is g, loss_upper is f

    # :param eta_lower: the stepsize for updating inner variables
    # :param S: number of iterations for updating inner variables (set to be large for ad option)
    :param option: the option for estimating hypergradient, ['cg', 'ns', 'hinv', 'ad']
    :param true_hessinv: the function call to compute true hessian inverse (only valid when option = 'hinv'
    :param ns_iter: number of iterations for estimating hypergradient using truncated neumann series
    :param ns_gamma: gamma for truncated neumann series (need to be sufficiently small to ensure gamma*hess has norm bounded by 1)

    """

    if option == 'cg':
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        # hessinv grad
        def rhess_prod(u):
            egrad = autograd(loss_lower(hparams, params, data_lower), params, create_graph=True)
            ehess = autograd(dot(egrad, u), params)
            out = []
            with torch.no_grad():
                for idx, param in enumerate(params):
                    out.append(param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
            return out
        Hinv_gy = ts_conjugate_gradient(rhess_prod, rgradfy, params, lam=cg_gamma, maxiter=cg_iter)

        gradgxy = compute_jvp2(loss_lower, hparams, params, data_lower, Hinv_gy)

        # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams,gradgxy)]

        # print(rgradfx)
        # print(rgradfy)
        # print(Hinv_gy)
        # print(gradgxy_proj)

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

    elif option.lower() == 'hinv':
        assert true_hessinv is not None
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        with torch.no_grad():
            Hinv_gy = true_hessinv(loss_lower, hparams, params, data_lower, rgradfy)

        # hparams = [ManifoldParameter(p.detach().clone(), manifold=p.manifold) for p in hparams]
        # params = [ManifoldParameter(p.detach().clone(), manifold=p.manifold) for p in params]

        gradgxy = compute_jvp2(loss_lower, hparams, params, data_lower, Hinv_gy)

        # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

    elif option.lower() == 'ns':
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        def reg_rhess_prod(u):
            egrad = autograd(loss_lower(hparams, params, data_lower), params, create_graph=True)
            ehess = autograd(dot(egrad, u), params)
            out = []
            with torch.no_grad():
                for idx, param in enumerate(params):
                    out.append(u[idx] - ns_gamma * param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
            return out

        with torch.no_grad():
            Hinv_gy_prev = [g.clone().detach() for g in rgradfy]
            Hinv_gy = [g.clone().detach() for g in rgradfy]
            for ins in range(ns_iter):
                with torch.enable_grad():
                    Hinv_gy_new = reg_rhess_prod(Hinv_gy_prev)
                Hinv_gy = [hg + hg_new for hg, hg_new in zip(Hinv_gy, Hinv_gy_new)]
                Hinv_gy_prev = Hinv_gy_new
            if ns_gamma > 0:
                Hinv_gy = [ns_gamma * hg for hg in Hinv_gy]

        gradgxy = compute_jvp2(loss_lower, hparams, params, data_lower, Hinv_gy)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

    elif option.lower() == 'ad':
        egradfy = autograd(loss_upper(hparams, params, data_upper), hparams)
        return [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, egradfy)]

    else:
        raise("option not implemented.")




def get_subset(data, split_ratio, keepfirst=True):
    n = data[0].shape[0]
    if keepfirst:
        new_data = [dd[:int(n*split_ratio)] for dd in data]
    else:
        new_data = [dd[int(n * split_ratio):] for dd in data]
    return new_data




def compute_hypergrad_stoc(loss_lower, loss_upper, hparams, params,
                       data_lower =None, data_upper=None,
                       option='cg',
                       true_hessinv=None,
                       ns_iter=200, ns_gamma=0.01):
    """hparams is x, params is y, loss_lower is g, loss_upper is f

    # :param eta_lower: the stepsize for updating inner variables
    # :param S: number of iterations for updating inner variables (set to be large for ad option)
    :param option: the option for estimating hypergradient, ['cg', 'ns', 'hinv', 'ad']
    :param true_hessinv: the function call to compute true hessian inverse (only valid when option = 'hinv'
    :param ns_iter: number of iterations for estimating hypergradient using truncated neumann series
    :param ns_gamma: gamma for truncated neumann series (need to be sufficiently small to ensure gamma*hess has norm bounded by 1)

    """

    # n_lower = data_lower.shape[0]

    if option == 'cg':
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        # hessinv grad
        def rhess_prod(u):
            egrad = autograd(loss_lower(hparams, params, get_subset(data_lower, 0.5, True)), params, create_graph=True)
            ehess = autograd(dot(egrad, u), params)
            out = []
            with torch.no_grad():
                for idx, param in enumerate(params):
                    out.append(param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
            return out
        Hinv_gy = ts_conjugate_gradient(rhess_prod, rgradfy, params, lam=0.)
        gradgxy = compute_jvp2(loss_lower, hparams, params, get_subset(data_lower, 0.5, False), Hinv_gy)

        # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams,gradgxy)]

        # print(rgradfx)
        # print(rgradfy)
        # print(Hinv_gy)
        # print(gradgxy_proj)

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

    elif option.lower() == 'hinv':
        assert true_hessinv is not None
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        Hinv_gy = true_hessinv(loss_lower, hparams, params, get_subset(data_lower, 0.5, True), rgradfy)
        gradgxy = compute_jvp2(loss_lower, hparams, params, get_subset(data_lower, 0.5, False), Hinv_gy)

        # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

    elif option.lower() == 'ns':
        egradfy = autograd(loss_upper(hparams, params, data_upper), params)
        egradfx = autograd(loss_upper(hparams, params, data_upper), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)


        def reg_rhess_prod(u):
            egrad = autograd(loss_lower(hparams, params, get_subset(data_lower, 0.5, True)), params, create_graph=True)
            ehess = autograd(dot(egrad, u), params)
            out = []
            with torch.no_grad():
                for idx, param in enumerate(params):
                    out.append(u[idx] - ns_gamma * param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
            return out

        with torch.no_grad():
            Hinv_gy_prev = [g.clone().detach() for g in rgradfy]
            Hinv_gy = [g.clone().detach() for g in rgradfy]
            for ins in range(ns_iter):
                with torch.enable_grad():
                    Hinv_gy_new = reg_rhess_prod(Hinv_gy_prev)
                Hinv_gy = [hg + hg_new for hg, hg_new in zip(Hinv_gy, Hinv_gy_new)]
                Hinv_gy_prev = Hinv_gy_new
            if ns_gamma > 0:
                Hinv_gy = [ns_gamma * hg for hg in Hinv_gy]

        gradgxy = compute_jvp2(loss_lower, hparams, params, get_subset(data_lower, 0.5, False), Hinv_gy)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]

        return [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

    elif option.lower() == 'ad':
        egradfy = autograd(loss_upper(hparams, params, data_upper), hparams)
        return [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, egradfy)]

    else:
        raise("option not implemented.")






