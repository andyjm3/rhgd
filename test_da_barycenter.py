# test on domain adaptation with barycenter learning
import torch
from torchvision import datasets
from torchvision import transforms
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import os

import argparse

import numpy as np

import ot

from geoopt import ManifoldParameter

from manifolds import SymmetricPositiveDefiniteMod, DoublyStochastic
from geoopt.optim import RiemannianSGD, RiemannianAdam
from optimizer import RHGDstep

from geoopt.linalg import sym_inv_sqrtm1, sym_sqrtm, sym_funcm, sym_inv_sqrtm2
from utils import autograd, compute_hypergrad2

from scipy.io import loadmat
from sklearn.decomposition import PCA, KernelPCA

import time
import pickle
# code obtained from https://github.com/s-chh/PyTorch-DANN/blob/main/data_loader.py

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def KNN(x_train, y_train, x_test, k=1):
    """
    Implement KNN classifier
    :param x_train: [N, d]
    :param y_train: [N]
    :param x_test: [M, d]
    :param k: int
    :return:
    """
    assert len(y_train.size()) == 1
    dist = torch.cdist(x_train, x_test)**2 # [N, M]
    indices = torch.topk(dist, k=k,dim=0, largest=False)[1] # (k, M)
    y_pred = torch.mode(y_train[indices], dim=0)[0] # [M]
    return y_pred


def bary_proj(Gamma, mu, t_data):
    """
    :param Gamma: [n,m]
    :param mu: [n]
    :param t_data: [m,d]
    :return:
    """
    return 1/mu.unsqueeze(-1) * (Gamma @ t_data)



def loss_upper(hparams, params, data=None):
    Gamma = hparams[0]
    M = params[0]
    # Minv = torch.matrix_power(M, -1)
    # dist = torch.diag(s_data @ Minv @ s_data.T).unsqueeze(1) + torch.diag(t_data @ Minv @ t_data.T).unsqueeze(
    #     0) - 2 * s_data @ Minv @ t_data.T
    Minvhalf, _ = sym_inv_sqrtm2(M)
    s_data_scale = s_data @ Minvhalf
    t_data_scale = t_data @ Minvhalf
    dist = torch.cdist(s_data_scale, t_data_scale) ** 2
    # with torch.no_grad():
    # dist = dist/dist.max()
    entropy = (Gamma * torch.log(Gamma)).sum()
    return (dist * Gamma).sum() + lam * entropy

def loss_lower(hparams, params, data=None):
    Gamma = hparams[0]
    M = params[0]
    s_data_proj = bary_proj(Gamma, mu, t_data)
    # s_data_proj = Gamma @ t_data
    M1 = s_data.T @ s_data + eps * torch.eye(s_data.shape[1], device=s_data.device)
    M2 = s_data_proj.T @ s_data_proj + eps * torch.eye(t_data.shape[1], device=t_data.device)
    M1 = (M1 + M1.T) * 0.5
    M2 = (M2 + M2.T) * 0.5
    return s_ratio * spd.dist(M, M1)**2 + (1-s_ratio) * spd.dist(M, M2)**2
    # s_data_proj = Gamma @ t_data
    # return s_ratio * spd.dist( s_data.T @ s_data, M)**2 + (1-s_ratio) * spd.dist(s_data_proj.T @ s_data_proj, M)**2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eta_x', type=float, default=0.5)
    parser.add_argument('--eta_y', type=float, default=0.5)
    parser.add_argument('--lower_iter', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--hygrad_opt', type=str, default='cg', choices=['hinv', 'cg', 'ns', 'ad'])
    parser.add_argument('--ns_gamma', type=float, default=0.1)
    parser.add_argument('--ns_iter', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.seed)
    print(device)
    # torch.random.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True

    tr = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize([10, 10]),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5]),
                             ])
    s_train = datasets.MNIST(os.path.join('data/', 'mnist'), train=True, download=True)

    # t_train = datasets.USPS(os.path.join('data/', 'usps'), train=True, download=True)
    t_train = datasets.MNIST(os.path.join('data/', 'mnist'), train=False, download=True)

    path = 'data/mnist_train_test_process.pt'

    if not os.path.exists(path):
        print("Process and save data...")
        s_data = torch.cat([tr(img) for img in s_train.data.unsqueeze(1)], dim=0) # [6w, 10, 10]
        s_label = s_train.targets # [6w]
        t_data = torch.cat([tr(img) for img in t_train.data.unsqueeze(1)], dim=0)
        t_label = t_train.targets

        torch.save({"s_data": s_data, "s_label": s_label, 't_data': t_data, 't_label': t_label}, path)
    else:
        print('Loading from saved data...')
        data_all = torch.load(path)
        s_data = data_all['s_data']
        s_label = data_all['s_label']
        t_data = data_all['t_data']
        t_label = data_all['t_label']



    # extract smaller datasets
    # s_sample_per_class = 10
    # t_sample_per_class = 8
    s_sample = 1000
    t_sample = 1000
    n_class = 10
    s_label_dist = [0.1] * 10
    t_label_dist = [0.3, 0.2, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    s_data_new = []
    s_label_new = []
    t_data_new = []
    t_label_new = []
    for c in range(n_class):
        s_data_sub = s_data[s_label == c][:int(s_sample*s_label_dist[c])]
        s_label_sub = s_label[s_label == c][:int(s_sample*s_label_dist[c])]
        s_data_new.append(s_data_sub)
        s_label_new.append(s_label_sub)

        t_data_sub = t_data[t_label == c][:int(t_sample*t_label_dist[c])]
        t_label_sub = t_label[t_label == c][:int(t_sample*t_label_dist[c])]
        t_data_new.append(t_data_sub)
        t_label_new.append(t_label_sub)

    # data and shuffle
    s_data = torch.cat(s_data_new, dim=0) # [100, 10, 10]
    s_label = torch.cat(s_label_new)
    t_data = torch.cat(t_data_new, dim=0)  # [100, 10, 10]
    t_label = torch.cat(t_label_new)

    s_idx = torch.randperm(s_data.shape[0])
    s_data = s_data[s_idx].view(s_data.shape[0], -1).to(device) # [n, d]
    s_label = s_label[s_idx].to(device)
    t_idx = torch.randperm(t_data.shape[0])
    t_data = t_data[t_idx].view(t_data.shape[0], -1).to(device) # [m, d]
    t_label = t_label[t_idx].to(device)

    s_data = s_data / torch.norm(s_data, dim=1, keepdim=True)
    t_data = t_data / torch.norm(s_data, dim=1, keepdim=True)

    s_data = s_data + torch.randn_like(s_data) * 0.01
    t_data = t_data + torch.randn_like(t_data) * 0.01

    # vanilla KNN (0.71)
    # y_pred = KNN(s_data, s_label, t_data, k=1)
    # print((y_pred == t_label).sum()/y_pred.shape[0])






    #################################

    # n_source_samples = 600
    # n_target_samples = 600
    #
    # s_data, s_label = ot.datasets.make_data_classif('3gauss', n_source_samples, nz=2)
    # t_data, t_label = ot.datasets.make_data_classif('3gauss2', n_target_samples, nz=0.5)
    #
    # s_data = s_data / np.linalg.norm(s_data, axis=1, keepdims=True)
    # t_data = t_data / np.linalg.norm(t_data, axis=1, keepdims=True)
    #
    # # s_data = s_data / np.linalg.norm(s_data)
    # # t_data = t_data / np.linalg.norm(t_data)
    #
    # s_data = torch.from_numpy(s_data).float().to(device)
    # s_label = torch.from_numpy(s_label).to(device)
    # t_data = torch.from_numpy(t_data).float().to(device)
    # t_label = torch.from_numpy(t_label).to(device)

    ## caltech-office dataset
    data = loadmat('data/office_caltech_cleaned.mat')

    i = 3
    j = 2

    s_data = data['feature_cleaned'][0][i]
    s_label = data['label_cleaned'][0][i]
    t_data = data['feature_cleaned'][0][j]
    t_label = data['label_cleaned'][0][j]

    s_data = s_data / np.linalg.norm(s_data, axis=1, keepdims=True)
    t_data = t_data / np.linalg.norm(t_data, axis=1, keepdims=True)

    # s_data = s_data / np.linalg.norm(s_data)
    # t_data = t_data / np.linalg.norm(t_data)

    ns = s_data.shape[0]
    nt = t_data.shape[0]
    # apply pca
    # s_data = PCA(n_components=50).fit_transform(s_data)
    # t_data = PCA(n_components=50).fit_transform(t_data)

    s_data = torch.from_numpy(s_data).float().to(device)
    s_label = torch.from_numpy(s_label).to(device).squeeze()
    t_data = torch.from_numpy(t_data).float().to(device)
    t_label = torch.from_numpy(t_label).to(device).squeeze()

    s_data = s_data.view([ns, 128, 32]).mean(dim=-1)
    t_data = t_data.view([nt, 128, 32]).mean(dim=-1)

    y_pred = KNN(s_data, s_label, t_data, k=1)
    base_acc = (y_pred == t_label).sum() / y_pred.shape[0]
    print(base_acc)


    # OT using manifold optimization
    dist = torch.cdist(s_data, t_data, p=2)**2 # [n,m]
    # dist = dist/dist.max()

    def loss_da(hparams):
        Gamma = hparams[0]
        entropy = (Gamma * torch.log(Gamma)).sum()
        return (dist * Gamma).sum() + lam * entropy

    n = s_data.shape[0]
    m = t_data.shape[0]
    d = s_data.shape[1]
    mu = 1/n * torch.ones(n).to(device)
    nu = 1/m * torch.ones(m).to(device)
    ds_mfd = DoublyStochastic(mu, nu)
    epochs = 300

    # baseline
    # hparams = [ManifoldParameter(ds_mfd.random(device=device), manifold=ds_mfd)]
    # optimizer = RiemannianSGD(hparams, lr=0.5)
    #
    # for ep in range(epochs):
    #     optimizer.zero_grad()
    #     loss = loss_da(hparams)
    #     loss.backward()
    #     optimizer.step()
    #     print(f'Epochs: {ep}: ', loss.item())
    #
    # Gamma_mot = hparams[0]
    # s_data_proj = bary_proj(Gamma_mot, mu, t_data)
    # y_pred = KNN(s_data_proj, s_label, t_data, k=1)
    # print("Baseline accuracy ", (y_pred == t_label).sum()/y_pred.shape[0])

    s_ratio = 0.5
    def my_pow(u):
        return torch.pow(u, 1 - s_ratio)

    # POT baseline
    # (1)
    lam = 0.0001
    Gamma_sot = ot.sinkhorn(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), dist.detach().cpu().numpy(), lam,
                            numItermax=10000, stopThr=1e-06)
    # Gamma_sot = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), dist.detach().cpu().numpy(),
    #                         numItermax=100000)
    s_data_proj = bary_proj(torch.from_numpy(Gamma_sot).to(device), mu, t_data)
    y_pred = KNN(s_data_proj, s_label, t_data, k=1)
    mot_acc = (y_pred == t_label).sum() / y_pred.shape[0]
    print(mot_acc)

    # # (2) OT with metric learning
    # lam = 0.0005
    # # lam = 0.001
    # eps = 0.01
    # Gamma_sot2 = ot.sinkhorn(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), dist.detach().cpu().numpy(), lam,
    #                         numItermax=10000, stopThr=1e-06)
    # M1 = s_data.T @ s_data + eps * torch.eye(d, device=s_data.device)
    # M2 = t_data.T @ t_data + eps * torch.eye(d, device=t_data.device)
    # M1halfinv, M1half = sym_inv_sqrtm2(M1)
    # M_base = M1half @ sym_funcm(M1halfinv @ M2 @ M1halfinv, my_pow) @ M1half
    # s_data_proj = bary_proj(torch.tensor(Gamma_sot2, dtype=t_data.dtype, device=t_data.device), mu, t_data)
    # y_pred = KNN(s_data_proj @ sym_inv_sqrtm1(M_base), s_label, t_data @ sym_inv_sqrtm1(M_base), k=1)
    # print((y_pred == t_label).sum() / y_pred.shape[0])

    # (3) OT_embed
    # Gamma_sot = ot.sinkhorn(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), dist.detach().cpu().numpy(), lam,
    #                         numItermax=10000, stopThr=1e-06)
    Gamma_sot = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), dist.detach().cpu().numpy(),
                            numItermax=100000)
    s_data_proj = bary_proj(torch.from_numpy(Gamma_sot).to(device), mu, t_data)
    y_pred = KNN(s_data_proj, s_label, t_data, k=1)
    emd_acc = (y_pred == t_label).sum() / y_pred.shape[0]
    print(emd_acc)


    # print(torch.norm(torch.from_numpy(Gamma_sot).to(device) - Gamma_mot))

    ######## bilevel optimization #############
    # hyperparameters
    # d = 2
    n = s_data.shape[0]
    m = t_data.shape[0]
    d = s_data.shape[1]
    mu = 1 / n * torch.ones(n).to(device)
    nu = 1 / m * torch.ones(m).to(device)
    # lam = 0.0005
    lam = 0
    eps = 0.01

    spd = SymmetricPositiveDefiniteMod()
    ds = DoublyStochastic(mu, nu)
    params = [ManifoldParameter(torch.eye(d, device=device), manifold=spd)]
    # params = [ManifoldParameter( spd.random(d, d, device=device), manifold=spd)]
    # params = [ManifoldParameter(M_base, manifold=spd)]
    # hparams = [ManifoldParameter(ds.random(device=device), manifold=ds)]
    hparams = [ManifoldParameter(mu.unsqueeze(-1) @ nu.unsqueeze(0), manifold=ds)]
    mfd_params = [param.manifold for param in params]
    mfd_hparams = [hparam.manifold for hparam in hparams]


    loss_u_prev = 1000
    count_early = 0

    loss_u_all= []
    accuracy = []
    runtime = []
    for ep in range(args.epoch):

        step_start_time = time.time()

        # hparams, params, loss_u, hgradnorm, step_time = RHGDstep(loss_lower, loss_upper, hparams, params, args)

        # for ii in range(args.lower_iter):
        #     if args.hygrad_opt == 'ad':
        #         grad = autograd(loss_lower(hparams, params, None), params, create_graph=True)
        #         rgrad = [mfd.egrad2rgrad(param, egrad) for mfd, egrad, param in zip(mfd_params, grad, params)]
        #         params = [mfd.retr(param, - args.eta_y * rg) for mfd, param, rg in zip(mfd_params, params, rgrad)]
        #     else:
        #         grad = autograd(loss_lower(hparams, params, None), params)
        #         with torch.no_grad():
        #             for param, egrad in zip(params, grad):
        #                 rgrad = param.manifold.egrad2rgrad(param, egrad)
        #                 new_param = param.manifold.retr(param, -args.eta_y * rgrad)
        #                 param.copy_(new_param)
        #
        #             # print(loss_lower(hparams, params, None))
        # # compute optimal sol
        # with torch.no_grad():
        #     # s_data_proj = hparams[0] @ t_data
        #     s_data_proj = bary_proj(hparams[0], mu, t_data)
        #     M1 = s_data.T @ s_data + eps * torch.eye(s_data.shape[1], device=s_data.device)
        #     M2 = s_data_proj.T @ s_data_proj + eps * torch.eye(s_data.shape[1], device=s_data.device)
        #     M1halfinv, M1half = sym_inv_sqrtm2(M1)
        #     print(f"Distoopt: {torch.norm(params[0] - M1half @ sym_funcm(M1halfinv @ M2 @ M1halfinv, my_pow) @ M1half).item():.2f}")

        with torch.no_grad():
            s_data_proj = bary_proj(hparams[0], mu, t_data)
            M1 = s_data.T @ s_data + eps * torch.eye(s_data.shape[1], device=s_data.device)
            M2 = s_data_proj.T @ s_data_proj + eps * torch.eye(s_data.shape[1], device=s_data.device)
            M1halfinv, M1half = sym_inv_sqrtm2(M1)
            params[0].copy_(M1half @ sym_funcm(M1halfinv @ M2 @ M1halfinv, my_pow) @ M1half)
            # params[0].copy_(torch.eye(d,device=device))
            # print(torch.norm(params[0] - torch.eye(d,device=device)))
            # with torch.no_grad():
            #     print(loss_lower(hparams, params, None))

        # compute hypergrad estimate
        hypergrad = compute_hypergrad2(loss_lower, loss_upper, hparams, params,
                                       data_lower=None, data_upper=None,
                                       option=args.hygrad_opt, true_hessinv=None,
                                       cg_iter=200, cg_gamma = 0.,
                                       ns_gamma=args.ns_gamma, ns_iter=args.ns_iter)

        with torch.no_grad():
            for hparam, hg in zip(hparams, hypergrad):
                new_hparam = hparam.manifold.retr(hparam, - args.eta_x * hg)
                hparam.copy_(new_hparam)

            loss_u = loss_upper(hparams, params, None).item()

            if loss_u > loss_u_prev:
                s_data_proj = bary_proj(hparams[0], mu, t_data)
                y_pred = KNN(s_data_proj @ sym_inv_sqrtm1(params[0]), s_label, t_data @ sym_inv_sqrtm1(params[0]), k=1)
                print(f"Loss increase! Break at epoch {ep} with loss {loss_u:.4f} and accuracy {(y_pred == t_label).sum()/y_pred.shape[0]:.2f}")
                break
            else:
                loss_u_prev = loss_u

        s_data_proj = bary_proj(hparams[0], mu, t_data)
        y_pred = KNN(s_data_proj @ sym_inv_sqrtm1(params[0]), s_label, t_data @ sym_inv_sqrtm1(params[0]), k=1)

        acc = (y_pred == t_label).sum()/y_pred.shape[0]

        print(f"Epoch {ep}: "
              f"loss upper: {loss_u:.4f}, "
              # f"hypergrad norm: {hgradnorm:.2f}, "
              f"accuracy: {acc:.2f}")

        loss_u_all.append(loss_u)
        accuracy.append(acc.item())
        runtime.append(time.time() - step_start_time)

        # print("Baseline accuracy ", (y_pred == t_label).sum()/y_pred.shape[0])


    stats = {'accuracy': np.array(accuracy), 'loss_upper': np.array(loss_u_all), 'runtime': runtime,
             'base_knn': base_acc.item(), 'base_mot': mot_acc.item(), 'base_emd': emd_acc.item()}

    filename = 'da_' + str(i) + '_' + str(j) + '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)



    # data = scipy.io.loadmat('data/MNIST_data.mat')
    # mnist_x = data['data'][0]
    # mnist_y = data['labels'][0]
    # data = scipy.io.loadmat('data/USPS_data.mat')
    # usps_x = data['data'][0]
    # usps_y = data['labels'][0]
    #
    # mnist_x = np.array(mnist_x)
    # mnist_y = np.array(mnist_y)
    # usps_x = np.array(usps_x)
    # usps_y = np.array(usps_y)



