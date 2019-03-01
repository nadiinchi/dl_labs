import torch
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
plt.switch_backend('agg')

def sd2tensor(sd):
    res = None
    for key in sd:
        if res is None:
            res = sd[key].new_zeros(0)
        res = torch.cat((res, sd[key].reshape(-1).float()))
    return res

def tensor2sd(tensor, sd_example):
    i = 0
    res = OrderedDict()
    for key in sd_example:
        n = sd_example[key].numel()
        res[key] = tensor[i:i+n].reshape(sd_example[key].shape)
        i += n
    assert i == len(tensor)
    return res

"""
def nllloss(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target, reduction='none')
    return loss

def accfun(model, data, target):
    output = model(data)
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred))

standard_metrics = {"per_minibatch":{
                       "loss": ("3.3e", nllloss),
                       "acc": (".4f", accfun)},
                    "per_model":{}}
"""

def compute_along_manifold(manifold, loader, evaluate_manifold_fun, index_grid):
    res = {}
    def update_res(res_index):
        for key in res_index:
            if key not in res:
                res[key] = []
            res[key].append(res_index[key])
    for index in index_grid:
        manifold.index = index
        manifold.freeze_index = True
        loss, acc = evaluate_manifold_fun(manifold, loader)
        update_res({"loss":loss, "acc":acc})
    return res
    
def build_segment_grid(device, num_points=100):
    return torch.linspace(0, 1, num_points).to(device)
    
def plot_along_manifold(manifolds, train_loader, test_loader, evaluate_fun, criterion, device, index_grid, plot=False):
    evaluate_manifold_fun = lambda manifold, loader: evaluate_fun(loader, manifold, criterion, device)
    res = {}
    for con_name, manifold in manifolds.items():
        res[con_name] = {}
        res[con_name]["train"] = compute_along_manifold(manifold, train_loader, \
                                         evaluate_manifold_fun,\
                                         index_grid)
        res[con_name]["test"] = compute_along_manifold(manifold, test_loader, \
                                         evaluate_manifold_fun,\
                                         index_grid)
    if plot:
        num_pics = 4
        plt.figure(figsize=(5, 3*num_pics))
        i = 1
        for regime in ["train", "test"]:
            for metric_name in ["loss", "acc"]:
                plt.subplot(num_pics, 1, i)
                for con_name in res:
                    plt.plot(index_grid.cpu().numpy(), \
                             res[con_name][regime][metric_name], label=con_name)
                plt.legend(loc=(1, 0))
                plt.title(regime+" "+metric_name)
                i += 1
    return res

def plot_2d(components, mean, ws, plane, device, train_loader, test_loader, \
                evaluate_fun, criterion, num_points_hor=11, num_points_ver=11, num_levels=20):
        V = components.T
        coords = V.T.dot(ws.T-mean[:, np.newaxis])
        left = coords[0].min() - 0.2
        right = coords[0].max() + 0.2
        bottom = coords[1].min() - 0.2
        top = coords[1].max() + 0.2
        xv, yv = np.meshgrid(np.linspace(left, right, num_points_hor), \
                             np.linspace(bottom, top, num_points_ver), sparse=False, indexing='ij')
        index_grid = torch.from_numpy(np.vstack((xv.ravel(), yv.ravel())).T).to(device)
        res = plot_along_manifold({"plane":plane}, train_loader, test_loader, evaluate_fun, criterion, device, index_grid, plot=False)
        num_pics = 4
        plt.figure(figsize=(num_points_ver*0.3, num_points_hor*0.3*num_pics))
        i = 1
        for regime in ["train", "test"]:
            for metric_name in ["loss", "acc"]:
                plt.subplot(num_pics, 1, i)
                for con_name in res:
                    plt.contourf(xv, yv, np.array(res[con_name][regime][metric_name]).\
                                 reshape(num_points_hor, num_points_ver), num_levels)
                    plt.colorbar()
                    plt.plot(*V.T.dot(ws.T-mean[:, np.newaxis]), color="white")
                plt.title(regime+" "+metric_name)
                i += 1
                
def npvec_to_tensorlist(vec, params, device):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params
        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net
        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float().to(device))
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def gradtensor_to_npvec(net, include_bn=False):
    """ Extract gradients from net, and return a concatenated numpy vector.
        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.
        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: True #include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])

