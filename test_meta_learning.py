"""
Few-shot meta-learning with adaptation over partial parameters
"""
import math
import argparse
import time
import collections
from collections import OrderedDict
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as Tr
from PIL import Image
import higher
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels

from geoopt import ManifoldParameter, Stiefel
from geoopt.optim import RiemannianSGD, RiemannianAdam
from manifolds import EuclideanMod

import random

from torch.optim import SGD

import gc
from utils import autograd, batch_egrad2rgrad, dot, ts_conjugate_gradient

from utils import conjugate_gradient








def compute_hypergrad(task, hparams_ker, hparams, params, option='cg',
                      true_hessinv=None,
                      ns_iter=50, ns_gamma=0.005):

    def compute_jvp(loss, hparams, params, tangents):
        """
        Compute the cross derivative of loss(hparams, params), i.e., G_xy [tangents] where x is hparams, y is params
        :param loss:
        :param inputs: List[Tensors] of size hparams
        :param tangents: List[Tensors] of size params
        :return:
        """
        assert len(params) == len(tangents)

        def function(*params):
            grad = autograd(loss(list(params)), hparams, create_graph=True)  # list of size hparams
            return tuple([hparam.manifold.egrad2rgrad(hparam, gg) for hparam, gg in zip(hparams, grad)])

        _, gradxy = torch.autograd.functional.jvp(function, tuple(params), tuple(tangents))

        return gradxy

    # print("Computing hypergrad using", option)

    if option == 'cg':
        o_loss = task.val_loss_f(params, hparams_ker)
        egradfy = autograd(o_loss, params)
        egradfx = autograd(task.val_loss_f(params, hparams_ker), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        task.compute_feats(hparams_ker) # compute once for all

        def rhess_prod(u):
            egrad = autograd(task.train_loss_f(params), params, create_graph=True)
            ehess = autograd(dot(egrad, u), params)
            out = []
            with torch.no_grad():
                for idx, param in enumerate(params):
                    out.append(param.manifold.ehess2rhess(param, egrad[idx], ehess[idx], u[idx]))
            return out
        # Hinv_gy = ts_conjugate_gradient(rhess_prod, rgradfy, params, lam=1, tol=1e-2, maxiter=50, verbose=0)
        Hinv_gy = conjugate_gradient(rhess_prod, rgradfy, maxiter=50)
        gradgxy = compute_jvp(task.train_loss_f, hparams, params, Hinv_gy)

        # proj to tangent space (it can be a bit off the tangent space due to numerical errors)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams,gradgxy)]

        grads = [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

        update_tensor_grads(hparams, grads)

    elif option == 'ns':
        o_loss = task.val_loss_f(params, hparams_ker)
        egradfy = autograd(o_loss, params)
        egradfx = autograd(task.val_loss_f(params, hparams_ker), hparams)
        rgradfy = batch_egrad2rgrad(params, egradfy)
        rgradfx = batch_egrad2rgrad(hparams, egradfx)

        task.compute_feats(hparams_ker)  # compute once for all

        def reg_rhess_prod(u):
            egrad = autograd(task.train_loss_f(params), params, create_graph=True)
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

        gradgxy = compute_jvp(task.train_loss_f, hparams, params, Hinv_gy)
        gradgxy_proj = [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, gradgxy)]

        grads = [g1 - g2 for g1, g2 in zip(rgradfx, gradgxy_proj)]

        update_tensor_grads(hparams, grads)

    elif option.lower() == 'ad':
        egradfx = autograd(task.val_loss_f(params, hparams_ker), hparams)
        grads = batch_egrad2rgrad(hparams, egradfx)
        update_tensor_grads(hparams, grads)
        # return [hp.manifold.proju(hp, gxy) for hp, gxy in zip(hparams, egradfy)]










# import hypergrad as hg
def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))
def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams

def esj(params_list,
        hparams_ker, hparams,
        us,
        outer_loss,
        mu,
        set_grad=True,
        more=False):

    """HyperGradient Via ES Jacobian

    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
        the list of hypergradients for each element in hparams

    """

    params = params_list[0]
    params_mu = params_list[1]

    params = [w.detach().requires_grad_(True) for w in params]
    if more:
        # o_loss, acc = outer_loss(params, hparams, verbose=True, more=more)
        o_loss, acc = outer_loss(params, hparams_ker, more=more)
    else:
        o_loss = outer_loss(params, hparams_ker)
    grad_outer_ws, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)  # output is (grad wrt last iterate of w, grad wrt hparams)

    deltas = [(param_mu - param) / mu for param_mu, param in zip(params_mu, params)]
    deltas_dot_grad_ws = sum([torch.dot(delta.view(-1), grad_outer_w.view(-1)) for delta, grad_outer_w in zip(deltas, grad_outer_ws)])

    grads = [grad_outer_hparam + deltas_dot_grad_ws * u
             for grad_outer_hparam, u in zip(grad_outer_hparams, us)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    if more:
        return grads, o_loss, acc
    else:
        return grads, o_loss

zoj = esj

class classifier(nn.Module):

    def __init__(self, n_features=84, n_classes=10):
        super(classifier, self).__init__()

        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, feats):
        output = self.fc(feats)

        return output

def MiniimageNetFeats(hidden_size):
    def conv_layer(ic, oc):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(ic, oc, 3, padding=1)),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(2)),
            ("bn", nn.BatchNorm2d(oc, momentum=1., affine=True, track_running_stats=False))
        ]))
        # return nn.Sequential(
        #     nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        #     nn.BatchNorm2d(oc, momentum=1., affine=True,
        #                    track_running_stats=False
        #                    )
        # )

    net = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten())

    #initialize(net)
    return net


def split_into_adapt_eval(batch,
               shots,
               ways,
               device=None):

    # Splits task data into adaptation/evaluation sets

    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adapt_idx = np.zeros(data.size(0), dtype=bool)
    adapt_idx[np.arange(shots * ways) * 2] = True

    eval_idx = torch.from_numpy(~adapt_idx)
    adapt_idx = torch.from_numpy(adapt_idx)
    adapt_data, adapt_labels = data[adapt_idx], labels[adapt_idx]
    eval_data, eval_labels = data[eval_idx], labels[eval_idx]

    return adapt_data, adapt_labels, eval_data, eval_labels


class Task:
    """
    Handles the train and validation loss for a single task
    """
    def __init__(self, reg_param, meta_model, task_model, data, batch_size=None): # here batchsize = number of tasks used at each step. we will do full GD for each task
        device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmeta = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)
        self.ftask = higher.monkeypatch(task_model, device=device, copy_initial_weights=True)

        #self.n_params = len(list(meta_model.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def compute_feats(self, hparams):
        # compute train feats
        self.train_feats = self.fmeta(self.train_input, params= hparams)

    def reg_f(self, params):
        # l2 regularization
        return sum([(p ** 2).sum() for p in params])

    def train_loss_f(self, params):
        # regularized cross-entropy loss
        out = self.ftask(self.train_feats, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.reg_f(params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        feats = self.fmeta(self.test_input, params=hparams)
        out = self.ftask(feats, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss


def inner_solver(task, hparams, params, steps, optim, params0=None, log_interval=None):

    if params0 is not None:
        for param, param0 in zip(params, params0):
            param.data = param0.data

    params_mfd = [param.manifold for param in params]
    task.compute_feats(hparams) # compute feats only once to make inner iterations lighter (only linear transformations!)

    for t in range(steps):
        loss = task.train_loss_f(params)
        optim.zero_grad()
        grads = torch.autograd.grad(loss, params)
        update_tensor_grads(params, grads)
        optim.step()
        # print('Inner step t={}, Loss: {:.6f}'.format(t, loss.item()))
    return [ManifoldParameter(param.detach().clone(), manifold=mfd) for param, mfd in zip(params, params_mfd)]



def ad_inner_solver(task, hparams, params, steps, lr, params0=None, log_interval=None):

    if params0 is not None:
        for param, param0 in zip(params, params0):
            param.data = param0.data

    params_mfd = [param.manifold for param in params]
    task.compute_feats(hparams) # compute feats only once to make inner iterations lighter (only linear transformations!)

    for t in range(steps):
        loss = task.train_loss_f(params)
        grads = torch.autograd.grad(loss, params)
        params = [p - lr * grad for p, grad in zip(params, grads)]
        # print('Inner step t={}, Loss: {:.6f}'.format(t, loss.item()))

    return params


def kernel_to_param_resize(params, orisz_ls, paramsz_ls, ortho_idx):
    new_params = []
    for ii, p in enumerate(params):
        if ii in ortho_idx:
            assert p.size() == orisz_ls[ii]
            p_temp = p.view(orisz_ls[ii][0], -1).transpose(-1,-2)
            assert p_temp.size() == paramsz_ls[ii]
            new_params.append(p_temp)
        else:
            new_params.append(p)
    return new_params


def param_to_kernel_resize(params, orisz_ls, paramsz_ls, ortho_idx):
    new_params = []
    for ii, p in enumerate(params):
        if ii in ortho_idx:
            assert p.size() == paramsz_ls[ii]
            new_params.append(p.transpose(-1, -2).view(orisz_ls[ii]))
        else:
            new_params.append(p)
    return new_params



def main():

    parser = argparse.ArgumentParser(description='MAML with Partial Parameter Adaptation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='N', help='omniglot or miniimagenet or fc100')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume from checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='metalogs', help='path of checkpoint file')
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5, help='num classes in few shot learning')
    parser.add_argument('--shots', type=int, default=5, help='num training shots in few shot learning')
    parser.add_argument('--steps', type=int, default=3000, help='total number of outer steps')
    parser.add_argument('--use_resnet', type=bool, default=False, help='whether to use resnet12 network for minimagenet dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--intrinsic', type=bool, default=True,
                        help='if intrinsic, then update and gradient is on Stiefel manifold, else, on Euclidean and '
                             'project back to Stiefel after each update')

    parser.add_argument('--hess_opt', type=str, default='ad', choices=['ns', 'cg', 'ad'],
                        help='options to compute hessian inverse')
    parser.add_argument('--ns_iter', type=int, default=50,
                        help='N series iterations')
    parser.add_argument('--ns_gamma', type=float, default=0.01,
                        help='N series gamma')

    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    run = 1
    mu = 0.1
    inner_lr = .01
    outer_lr = .01
    inner_mu = 0.9
    K = args.steps
    stop_k = None  # stop iteration for early stopping. leave to None if not using it
    n_tasks_train = 20000
    n_tasks_test = 200  # usually around 1000 tasks are used for testing
    n_tasks_val = 200

    if args.dataset == 'omniglot':
        reg_param = 0.2  # reg_param = 2.
        T = 50  # T = 16
    elif args.dataset == 'miniimagenet':
        reg_param = 0.5  # reg_param = 0.5
        T = 30 # T = 30
    elif args.dataset == 'fc100':
        reg_param = 0.5  # reg_param = 0.5
        T = 30 # T = 30

    else:
        raise NotImplementedError(args.dataset, " not implemented!")

    T_test = T
    log_interval = 25
    eval_interval = 50

    # loc = locals()
    # del loc['parser']
    # del loc['args']
    #
    # args.out_file = open(os.path.join(args.ckpt_dir, 'log_ESJ_' + args.dataset + str(run) + '.txt'), 'w')
    #
    # string = "+++++++++++++++++++ Arguments ++++++++++++++++++++\n"
    # for item, value in args.__dict__.items():
    #     string += "{}:{}\n".format(item, value)
    #
    # args.out_file.write(string + '\n')
    # args.out_file.flush()
    # print(string + '\n')
    #
    # string = ""
    # for item, value in loc.items():
    #     string += "{}:{}\n".format(item, value)
    #
    # args.out_file.write(string + '\n')
    # args.out_file.flush()
    # print(string, '\n')

    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        print('Training on cuda device...')
    else:
        print('Training on cpu...')
    device = torch.device("cuda" if cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)


    ##### data processing #####
    MEAN = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    STD = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    normalize = Tr.Normalize(mean=MEAN, std=STD)

    # use the same data-augmentation as in lee et al.
    transform_train = Tr.Compose([
        # Tr.ToPILImage(),
        # Tr.RandomCrop(84, padding=8),
        # Tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # Tr.RandomHorizontalFlip(),
        # Tr.ToTensor(),
        normalize
    ])

    transform_test = Tr.Compose([
        normalize
    ])

    train_dataset = l2l.vision.datasets.MiniImagenet(
        root='data/MiniImageNet',
        mode='train',
        transform=transform_train,
        download=True)
    # print('got train dataset...')
    val_dataset = l2l.vision.datasets.MiniImagenet(
        root='data/MiniImageNet',
        mode='validation',
        transform=transform_test,
        download=True)
    # print('got val dataset...')
    test_dataset = l2l.vision.datasets.MiniImagenet(
        root='data/MiniImageNet',
        mode='test',
        transform=transform_test,
        download=True)
    # print('got test dataset...')

    hidden_size = 16
    meta_model = MiniimageNetFeats(hidden_size).to(device)
    task_model = classifier(hidden_size * 5 * 5, args.ways).to(device)

    print('meta model is : ', meta_model.__class__.__name__)

    train_dataset = l2l.data.MetaDataset(train_dataset)
    val_dataset = l2l.data.MetaDataset(val_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [FusedNWaysKShots(train_dataset, n=args.ways, k=2 * args.shots),
                        LoadData(train_dataset),
                        RemapLabels(train_dataset),
                        ConsecutiveLabels(train_dataset)]

    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=n_tasks_train)

    val_transforms = [FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
                      LoadData(val_dataset),
                      ConsecutiveLabels(val_dataset),
                      RemapLabels(val_dataset)]

    val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms=val_transforms, num_tasks=n_tasks_val)

    test_transforms = [FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
                       LoadData(test_dataset),
                       RemapLabels(test_dataset),
                       ConsecutiveLabels(test_dataset)]

    test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=n_tasks_test)
    ##############################


    print('got dataset: ', args.dataset)

    print('starting from scratch....')
    start_iter = 0
    total_time = 0

    run_time, accs, vals, evals = [], [], [], []

    w0 = [torch.zeros_like(p).to(device) for p in task_model.parameters()]
    params_mfd = [EuclideanMod(ndim=len(p.size())) for p in task_model.parameters()]

    hparams_ker = list(meta_model.parameters()) # original size kernel parameters

    # create mfd and parameters for hparams
    hparams = [] # ortho param for kernel
    hparams_mfd = []
    orisz_ls = [] # original size (kernel)
    paramsz_ls = [] # param size (ortho)
    # ortho_idx = [0, 4]
    ortho_idx = [4]

    if args.intrinsic:
        for ii, p in enumerate(hparams_ker):
            orisz_ls.append(p.size())
            if ii in ortho_idx:
                mfd = Stiefel(canonical=False)
                hparams.append(ManifoldParameter(mfd.projx(p.detach().clone().view(hidden_size,-1).transpose(-1,-2)), manifold=mfd))
            else:
                mfd = EuclideanMod(ndim=len(p.size()))
                hparams.append(ManifoldParameter(p.detach().clone(), manifold=mfd))
            paramsz_ls.append(hparams[-1].size())
            hparams_mfd.append(mfd)
    else:
        for ii, p in enumerate(hparams_ker):
            orisz_ls.append(p.size())
            if ii in ortho_idx:
                mfd = EuclideanMod(ndim=2)
                hparams.append(ManifoldParameter((p.detach().clone().view(hidden_size, -1).transpose(-1, -2)),
                                                 manifold=mfd))
            else:
                mfd = EuclideanMod(ndim=len(p.size()))
                hparams.append(ManifoldParameter(p.detach().clone(), manifold=mfd))
            paramsz_ls.append(hparams[-1].size())
            hparams_mfd.append(mfd)


    outer_opt = RiemannianAdam(params=hparams, lr=outer_lr)

    inner_log_interval = None
    inner_log_interval_test = None

    meta_bsz = args.batch_size

    # training starts here
    val_losses, val_accs = evaluate(val_tasks, meta_model, task_model,
                                    param_to_kernel_resize(hparams, orisz_ls, paramsz_ls, ortho_idx), params_mfd, w0,
                                    reg_param,
                                    inner_lr, inner_mu, T_test, args.shots, args.ways)
    test_losses, test_accs = evaluate(test_tasks, meta_model, task_model,
                                      param_to_kernel_resize(hparams, orisz_ls, paramsz_ls, ortho_idx), params_mfd, w0,
                                      reg_param,
                                      inner_lr, inner_mu, T_test, args.shots, args.ways)
    epochs_all = [0]
    runtime_all = [0]
    val_acc_all = [val_accs.mean()]
    val_accstd_all = [val_accs.std()]
    val_loss_all = [val_losses.mean()]
    val_lossstd_all = [val_losses.std()]
    test_acc_all = [test_accs.mean()]
    test_accstd_all = [test_accs.std()]
    test_loss_all = [test_losses.mean()]
    test_lossstd_all = [test_losses.std()]
    for k in range(start_iter, K):
        start_time = time.time()

        outer_opt.zero_grad()

        us = [torch.randn(hparam.size()).to(device) for hparam in hparams]
        us = [u / torch.norm(u, 2) for u in us]
        hparams_mu = [mu * u + hparam for u, hparam in zip(us, hparams)]

        val_loss, val_acc = 0, 0
        forward_time, backward_time = 0, 0
        w_accum = [torch.zeros_like(w).to(device) for w in w0]

        th = 0.0

        for t_idx in range(meta_bsz):
            start_time_task = time.time()

            # sample a training task
            task_data = train_tasks.sample()

            task_data = split_into_adapt_eval(task_data,
                                              shots=args.shots,
                                              ways=args.ways,
                                              device=device)
            # single task set up
            task = Task(reg_param, meta_model, task_model, task_data, batch_size=meta_bsz)

            # single task inner loop
            # params = [p.detach().clone().requires_grad_(True) for p in w0]

            params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for p, mfd in zip(w0, params_mfd)]

            hparams_ker = param_to_kernel_resize(hparams, orisz_ls, paramsz_ls, ortho_idx)

            if args.hess_opt == 'ad':
                final_params = ad_inner_solver(task, hparams_ker,
                                                params, T, lr=inner_lr, params0=w0, log_interval=inner_log_interval)
            else:
                # retain_path = True if args.hess_opt == 'ad' else False
                inner_opt = RiemannianSGD(lr=inner_lr, momentum=inner_mu, params=params)
                final_params = inner_solver(task, hparams_ker, ## need to resize hparams to match the kernel shape
                                            params, T, optim=inner_opt, params0=w0, log_interval=inner_log_interval)
                inner_opt.state = collections.defaultdict(dict)  # reset inner optimizer state


            forward_time_task = time.time() - start_time_task

            # final_params_mu = inner_solver(task, param_to_kernel_resize(hparams_mu, orisz_ls, paramsz_ls, ortho_idx), ## need to resize hparams to match the kernel shape
            #                                params, T, optim=inner_opt, params0=w0)

            # single task hypergradient computation
            th0 = time.time()
            # zoj([final_params, final_params_mu], param_to_kernel_resize(hparams, orisz_ls, paramsz_ls, ortho_idx), ## need to resize hparams to match the kernel shape
            #                 hparams, us, task.val_loss_f, mu) # will accumulate single task hypergradient to get overall hypergradient
            compute_hypergrad(task, hparams_ker, hparams, final_params, option=args.hess_opt, ns_iter=args.ns_iter, ns_gamma=args.ns_gamma)
            th += time.time() - th0

            backward_time_task = time.time() - start_time_task - forward_time_task

            val_loss += task.val_loss
            val_acc += task.val_acc/task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task

            w_accum = [p + fp / meta_bsz for p, fp in zip(w_accum, final_params)]

        outer_opt.step()

        if not args.intrinsic:
            # proj back to manifold
            hparams_new = []
            for ii, p in enumerate(hparams):
                if ii in ortho_idx:
                    p_new = ManifoldParameter(Stiefel(canonical=False).projx(p.detach().clone()), manifold=EuclideanMod(ndim=2))
                    hparams_new.append(p_new)
                else:
                    hparams_new.append(p)
            hparams = hparams_new

        w0 = [w.clone() for w in w_accum]  # will be used as initialization for next step

        step_time = time.time() - start_time
        total_time += step_time

        run_time.append(total_time)
        vals.append(val_loss) # this is actually train loss in few-shot learning
        accs.append(val_acc) # this is actually train accuracy in few-shot learning

        if val_loss > 2.0 and k > 20: # quit if loss goes up after some iterations
            print('loss went up! exiting...')
            # exit()

        if k >= 1500: # 2000
            inner_lr = 0.01
            outer_lr = 0.001
            for param_group in outer_opt.param_groups:
                param_group['lr'] = outer_lr

        if k >= 3500: # 5000
            inner_lr = 0.01
            outer_lr = 0.0001 #0.0005
            for param_group in outer_opt.param_groups:
                param_group['lr'] = outer_lr


        if (k+1) % log_interval == 0 or k == 0 or k == K-1:
            string = 'META k={}/{} Lr: {:.5f} mu: {:.3f}  ({:.3f}s F: {:.3f}s, B: {:.3f}s, HG: {:.3f}s) Train Loss: {:.2e}, Train Acc: {:.2f}.'.format(k+1, K, outer_lr, mu, step_time, forward_time, backward_time, th, val_loss, 100. * val_acc)
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)

        if (k+1) % args.save_every == 0:
            state_dict = {'k': k+1,
                          'acc': accs,
                          'val': vals,
                          'eval': evals,
                          'time': run_time,
                          'hp': hparams,
                          'w': w0,
                          'opt': outer_opt.state_dict()
                          }
            filename = 'ESJ_shots5_Resnet_' + args.dataset + '_T' + str(T) + '_run' + str(run) + '.pt'
            save_path = os.path.join(args.ckpt_dir, filename)

            # save_checkpoint(state_dict, save_path)

        if (k+1) == stop_k: # early stopping

            state_dict = {'k': k+1,
                          'acc': accs,
                          'val': vals,
                          'eval': evals,
                          'time': run_time,
                          'hp': hparams,
                          'w': w0,
                          'opt': outer_opt.state_dict()
                          }
            filename = 'ESJ_shots5_Resnet_' + args.dataset + '_T' + str(T) + '_run' + str(run) + '.pt'
            save_path = os.path.join(args.ckpt_dir, filename)

            # save_checkpoint(state_dict, save_path)
            print('exiting...')
            exit()

        if (k+1) % eval_interval == 0:


            val_losses, val_accs = evaluate(val_tasks, meta_model, task_model, param_to_kernel_resize(hparams,orisz_ls,paramsz_ls,ortho_idx), params_mfd, w0, reg_param,
                                              inner_lr, inner_mu, T_test, args.shots, args.ways)

            #evals.append((val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std()))

            string = "Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std(), len(val_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)

            test_losses, test_accs = evaluate(test_tasks, meta_model, task_model, param_to_kernel_resize(hparams,orisz_ls,paramsz_ls,ortho_idx), params_mfd, w0, reg_param,
                                              inner_lr, inner_mu, T_test, args.shots, args.ways)

            evals.append((test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100.*test_accs.std()))

            string = "Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(test_losses.mean(), test_losses.std(), 100. * test_accs.mean(),100.*test_accs.std(), len(test_losses))
            # args.out_file.write(string + '\n')
            # args.out_file.flush()
            print(string)

            # savestats
            epochs_all.append(k)
            runtime_all.append(run_time[-1])
            val_loss_all.append(val_losses.mean())
            val_lossstd_all.append(val_losses.std())
            val_acc_all.append(val_accs.mean())
            val_accstd_all.append(val_accs.std())
            test_loss_all.append(test_losses.mean())
            test_lossstd_all.append(test_losses.std())
            test_acc_all.append(test_accs.mean())
            test_accstd_all.append(test_accs.std())

    test_acc_all = np.array(test_acc_all)
    test_accstd_all = np.array(test_accstd_all)

    stats = {'epochs': np.array(epochs_all), 'runtime': np.array(runtime_all), "val_loss_mean": np.array(val_loss_all),
         "val_loss_std": np.array(val_lossstd_all), "val_acc_mean": np.array(val_acc_all), "val_acc_std": np.array(val_accstd_all),
         'test_loss_mean': np.array(test_loss_all), "test_loss_std": np.array(test_lossstd_all), 'test_acc_mean': np.array(test_acc_all),
         'test_acc_std': np.array(test_accstd_all)}

    filename = 'RHGD_meta_' if args.intrinsic else 'PHGD_meta_'
    filename = filename + args.hess_opt + "_" + str(len(ortho_idx)) + '.pickle'

    with open(filename, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(metadataset, meta_model, task_model, hparams_ker, params_mfd, w0, reg_param, inner_lr, inner_mu, inner_steps, shots, ways):
    #meta_model.train()
    device = next(meta_model.parameters()).device

    iters = metadataset.num_tasks
    eval_losses, eval_accs = [], []

    for k in range(iters):

        data = metadataset.sample()
        data = split_into_adapt_eval(data,
                                     shots=shots,
                                     ways=ways,
                                     device=device)

        task = Task(reg_param, meta_model, task_model, data) # metabatchsize will be 1 here

        # single task inner loop
        # params = [p.detach().clone().requires_grad_(True) for p in w0]
        params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for p, mfd in zip(w0, params_mfd)]
        inner_opt = RiemannianSGD(lr=inner_lr, momentum=inner_mu, params=params)
        final_params = inner_solver(task, hparams_ker, params, inner_steps, optim=inner_opt, params0=w0)
        inner_opt.state = collections.defaultdict(dict)  # reset inner optimizer state
        # params = [ManifoldParameter(p.detach().clone(), manifold=mfd) for p, mfd in zip(w0, params_mfd)]
        # final_params = my_inner_solver(task, hparams_ker, params, inner_steps, lr=inner_lr, params0=w0,
        #                                retain_path=False)

        task.val_loss_f(final_params, hparams_ker)

        eval_losses.append(task.val_loss)
        eval_accs.append(task.val_acc)

        if k >= 999: # use at most 1000 tasks for evaluation
            return np.array(eval_losses), np.array(eval_accs)

    return np.array(eval_losses), np.array(eval_accs)


def update_tensor_grads(params, grads):
    for l, g in zip(params, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


if __name__ == '__main__':

    main()