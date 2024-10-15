# modify the manifold class in geoopt by adding more functionality
import torch
from geoopt import linalg
from geoopt import SymmetricPositiveDefinite, Euclidean, Sphere, Lorentz
from geoopt import Manifold
from geoopt.utils import size2shape
from geoopt.tensor import ManifoldTensor
from typing import Union, Tuple, Optional
import warnings

class SymmetricPositiveDefiniteMod(SymmetricPositiveDefinite):
    """SPD manifold with AI metric"""
    def __init__(self):
        super().__init__()

    def ehess2rhess(self, x: torch.Tensor, egrad: torch.Tensor, ehess: torch.Tensor, u: torch.Tensor):
        return x @ self.proju(x, ehess) @ x + linalg.sym(u @ egrad @ x)


class EuclideanMod(Euclidean):
    def __init__(self, ndim=0):
        super().__init__(ndim=ndim)

    def ehess2rhess(self, x: torch.Tensor, egrad: torch.Tensor, ehess: torch.Tensor, u: torch.Tensor):
        return ehess


class SphereMod(Sphere):
    def __init__(self):
        super().__init__()

    def ehess2rhess(self, x: torch.Tensor, egrad: torch.Tensor, ehess: torch.Tensor, u: torch.Tensor):
        return self.proju(x, ehess) - self.inner(x, x, egrad) * u


class LorentzMod(Lorentz):
    def __init__(self):
        super().__init__()

    def ehessrhess(self, x: torch.Tensor, egrad: torch.Tensor, ehess: torch.Tensor, u: torch.Tensor):
        egradmod = egrad.narrow(-1, 0, 1).mul_(-1)
        ehessmod = ehess.narrow(-1, 0, 1).mul_(-1)
        inner = self.inner(x, x, egradmod, keepdim=True)
        return self.proju(x, u * inner + ehessmod)



# modified from mctorch
def SKnopp(A, p, q, maxiters=None, checkperiod=None):

    """

    :param A: size(..., n,m)
    :param p: size(...,n)
    :param q: size(...,m), where ... needs to match among (A, p, q)
    :param maxiters:
    :param checkperiod:
    :return:
    """

    assert A.size()[:-2] == p.size()[:-1] == q.size()[:-1]

    tol = 1e-6
    if maxiters is None:
        maxiters = A.shape[-1]*A.shape[-2]

    if checkperiod is None:
        checkperiod = 10

    p = p.unsqueeze(-2) # (...,1,n)
    q = q.unsqueeze(-2) # (...,1,m)

    # if p.ndim < 2 and q.ndim < 2:
    #     p = p[None, :]
    #     q = q[None, :]

    C = A

    d1 = q / torch.sum(C, dim=-2, keepdim=True) # (..., 1,m)

    d2 = p / (d1 @ C.transpose(-1,-2)) # (...,1,n)

    # if C.ndim < 3:
    #     d2 = p / (d1 @ C.T)
    # else:
    #     d2 = p / torch.sum(C * d1[:, None, :], axis=2)

    gap = float("inf")

    iters = 0
    while iters < maxiters:
        row = d2 @ C # (..., 1,m)
        # if C.ndim < 3:
        #     row = multiprod(d2, C)
        # else:
        #     row = torch.sum(C * d2[:, :, None], axis=1)

        if iters % checkperiod == 0:
            gap = torch.max(torch.absolute(row * d1 - q))
            if torch.any(torch.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        d2 = p / (d1 @ C.transpose(-1,-2))
        # if C.ndim < 3:
        #     d2 = p / multiprod(d1, C.T)
        # else:
        #     d2 = p / torch.sum(C * d1[:, None, :], axis=2)

        if torch.any(torch.isnan(d1)) or torch.any(torch.isinf(d1)) or torch.any(torch.isnan(d2)) or torch.any(torch.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    # return C * (torch.einsum('bn,bm->bnm', d2, d1))
    return C * (d2.transpose(-1,-2) @ d1)



class DoublyStochastic(Manifold):
    name = "DoublyStochastic"
    ndim = 2

    def __init__(self, p: torch.Tensor, q: torch.Tensor):

        # if p is None:
        #     self._p = 1/n * torch.ones(n)
        # else:
        #     assert p.shape[-1] == n
        #     self._p = p
        # if q is None:

        assert p.size()[:-1] == q.size()[:-1], "non-manifold dim should match!"
        assert torch.allclose(p.sum(dim=-1), q.sum(dim=-1)), "Total mass is not the same!"

        self._n = p.shape[-1]
        self._m = q.shape[-1]
        self._size = torch.Size(list(p.size()[:-1]) + [self._n] + [self._m])

        self._p = p
        self._q = q
        super().__init__()

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
                                 )-> Union[Tuple[bool, Optional[str]], bool]:
        row = torch.sum(u, dim=-2)
        col = torch.sum(u, dim=-1)
        okrow = torch.allclose(row, torch.zeros_like(row), atol=atol, rtol=rtol)
        okcol = torch.allclose(col, torch.zeros_like(col), atol=atol, rtol=rtol)
        ok = okrow and okcol
        if not ok:
            return False, "`U^T 1 != 0` or `U 1 != 0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def my_check_point_on_manifold(self, x: torch.Tensor, p:torch.Tensor, q:torch.Tensor, *, atol=1e-5, rtol=1e-5
                                   )-> Union[Tuple[bool, Optional[str]], bool]:
        row = torch.sum(x, dim=-2)
        col = torch.sum(x, dim=-1)
        okrow = torch.allclose(row, q, atol=atol, rtol=rtol)
        okcol = torch.allclose(col, p, atol=atol, rtol=rtol)
        ok = okrow and okcol
        if not ok:
            return False, "`X^T 1 != p` or `X 1 != p` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return self.my_check_point_on_manifold(x, self._p, self._q)

    def random(self, dtype=None, device=None, **kwargs) -> torch.Tensor:
        Z = torch.absolute(torch.randn(*self._size, device=device))
        return SKnopp(Z, self._p, self._q)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size() == self._size
        return SKnopp(torch.absolute(x), self._p, self._q)

    def inner(
            self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        return (u * v / x).sum([-1, -2], keepdim=keepdim)

    def _lsolve(self, x, b):
        # x has size (...,n,m), b has size (...,n+m)
        # alpha = torch.empty(*self._p.size()) # (...,n)
        # beta = torch.empty(*self._q.size()) # (...,m)
        Aup = torch.cat([torch.diag_embed(self._p), x], dim=-1) # (...,n,n+m)
        Alo = torch.cat([x.transpose(-1,-2), torch.diag_embed(self._q)], dim=-1) # (...,m,m+n)
        A = torch.cat([Aup, Alo], dim=-2) # (...,n+m,n+m)
        # zeta = torch.linalg.solve(A, b)
        # use ls solver instead of solve to avoid singularity
        zeta, residual, _, _ = torch.linalg.lstsq(A, b)

        alpha = zeta[...,:self._n]
        beta = zeta[...,self._n:]
        return alpha, beta

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        b = torch.cat([u.sum(dim=-1), u.sum(dim=-2)], dim=-1)
        alpha, beta = self._lsolve(x, b)
        return u - (alpha.unsqueeze(-1).expand_as(x) + beta.unsqueeze(-2).expand_as(x)) * x

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        mu = x * u
        return self.proju(x, mu)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(u / x) * x
        return SKnopp(torch.clamp(temp, min=1e-10, max=1e10), self._p, self._q)

    expmap = retr

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor)-> torch.Tensor:
        return self.proju(y, v)

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp






class Grassmann(Manifold):
    name = "Grassmann"
    ndim = 2

    def __init__(self):
        super().__init__()

    def _check_shape(
            self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] <= shape[-2]
        if not shape_is_ok:
            return (
                False,
                "`{}` should have shape[-1] <= shape[-2], got {} </= {}".format(
                    name, shape[-1], shape[-2]
                ),
            )
        return True, None

    def _check_point_on_manifold(
            self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(
            self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        diff = x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        U, _, V = linalg.svd(x, full_matrices=False)
        return torch.einsum("...ik,...kj->...ij", U, V)

    def random_naive(self, *size, dtype=None, device=None) -> torch.Tensor:
        """
        Naive approach to get random matrix on Grassmann. This function is the
        same as for the geoopt.Stiefel manifold

        A helper function to sample a random point on the Stiefel manifold.
        The measure is non-uniform for this method, but fast to compute.

        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device

        Returns
        -------
        ManifoldTensor
            random point on Stiefel manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        tens = torch.randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(linalg.qr(tens)[0], manifold=self)

    random = random_naive

    def inner(
            self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        return (u * v).sum([-1, -2], keepdim=keepdim)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u - x @ (x.transpose(-1,-2) @ u)

    egrad2rgrad = proju

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.projx(x + u)

    expmap = retr

    def dist(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            keepdim=False,
    ) -> torch.Tensor:
        temp = x.transpose(-1,-2) @ y
        _, S, _ = linalg.svd(temp)
        S[torch.isclose(S, torch.ones_like(S))] = 1
        ret = torch.sqrt(torch.real(torch.acos(S)).pow(2).sum(dim=-1, keepdim=keepdim))
        return ret




if __name__ == '__main__':
    from geoopt import ManifoldParameter
    from geoopt.optim import RiemannianSGD, RiemannianAdam

    #### test on sinkhorn ####
    # n = 7
    # m = 5
    # p = 1/n * torch.ones(3,n)
    # q = 1/m * torch.ones(3,m)
    #
    # mfd = DoublyStochastic(p,q)
    #
    # # test random manifold point (ok)
    # A = mfd.random()
    # assert mfd.check_point_on_manifold(A)
    #
    # # test proju (ok)
    # u = torch.randn_like(A)
    # u_vec = mfd.proju(A, u)
    # assert mfd.check_vector_on_tangent(A, u_vec)
    #
    # # test on linear problem
    # k = 10
    # n = 7
    # m = 5
    # # p = torch.rand((k, n))
    # # q = torch.rand((k, m))
    # # p = p / torch.sum(p, dim=-1, keepdim=True)
    # # q = q / torch.sum(q, dim=-1, keepdim=True)
    # p = 1 / n * torch.ones(k, n)
    # q = 1 / m * torch.ones(k, m)
    # mfd = DoublyStochastic(p,q)
    # A = mfd.random()
    #
    # param = ManifoldParameter(mfd.random(), manifold=mfd)
    #
    # def cost(x):
    #     return 0.5 * (torch.linalg.norm(x - A)**2)
    #
    # optimizer = RiemannianAdam([param], lr=0.1)
    #
    # for epoch in range(20):
    #     optimizer.zero_grad()
    #     loss = cost(param)
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())
    #     assert(mfd.check_point_on_manifold(param))


    #### test on Grassmann ####
    n = 7
    d = 5
    r = 3
    mfd = Grassmann()
    # test random manifold point (ok)
    A = mfd.random(n,d,r)
    assert mfd.check_point_on_manifold(A)

    # test proju (ok)
    u = torch.randn_like(A)
    u_vec = mfd.proju(A, u)
    assert mfd.check_vector_on_tangent(A, u_vec)

    # test on PCA (ok)
    n = 20
    d = 10
    r = 5

    X = torch.randn(n,d)
    C = X.T @ X

    param = ManifoldParameter(mfd.random(d,r), manifold=mfd)

    def cost(x):
        return -0.5 * (torch.trace(x.T @ C @ x))

    optval, optsol = torch.linalg.eigh(C)

    optimizer = RiemannianSGD([param], lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        loss = cost(param)
        loss.backward()
        optimizer.step()
        print(f"optgap: {abs(-loss.item() - optval[-r:].sum()/2):.4f}, disttoopt: {mfd.dist(param, optsol[:,-r:]):.4f}")


