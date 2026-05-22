import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from . import networks
from .misc import (
    random_pairs_of_minibatches, split_meta_train_test, ParamDict,
    MovingAverage, l2_between_dicts, proj, Nonparametric
)

from calibration.ece import calc_ece, init_config
from calibration.utils import get_maxprob_and_onehot

from .base_algorithm import Algorithm

from torch.autograd import grad

class ERM_CURV_REG(Algorithm):

    """
    Empirical Risk Minimization with Curvature Regularization (ERM_CURV_REG)
    """

    def __init__(self, model, opt, loss_func, scheduler, ece_config, exp_dict):

        super(ERM_CURV_REG, self).__init__()

        # config
        self.exp_dict = exp_dict
        self.device = exp_dict['device']
        self.ece_config = ece_config

        # training
        self.loss_func = loss_func
        self.opt = opt
        self.scheduler = scheduler
        self.model = model

        # Sould be reset for each epoch
        # metric
        self.correct = 0
        self.total = 0
        self.losses = []

        # ece
        self.labels_list = []
        self.probs_list = []


    def my_calc_hessian_trace(model, data, target, criterion, maxIter=100):
        device = next(model.parameters()).device
        trace_est = torch.tensor(0.0, device=device, requires_grad=True)

        for _ in range(maxIter):
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 勾配計算のための create_graph=True を設定
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            v = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
            for v_i in v:
                v_i[v_i == 0] = -1

            # ここでも create_graph=True を設定
            Hv = grad(grads, model.parameters(), grad_outputs=v, only_inputs=True, create_graph=True)

            # trace_est の更新時にも create_graph=True を維持
            trace_est = trace_est + sum(torch.sum(hv * v_i) for hv, v_i in zip(Hv, v))

        # 平均値を計算し、勾配を持つテンソルとして返却
        return trace_est / maxIter

    def update(self, idx, batch):

        self.model.train()

        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(imgs)

        _, pred_label = torch.max(logits.data, 1)
        probs = torch.nn.functional.softmax(logits, dim=1)

        self.total += labels.size(0)
        self.correct += (pred_label == labels).sum().item()

        train_loss = self.loss_func(logits, labels)
        self.losses.append(train_loss.item())

        self.labels_list.extend(labels.detach().cpu().numpy())
        self.probs_list.extend(probs.detach().cpu().numpy())

        train_loss.backward()

        if ((idx + 1) % self.exp_dict['num_accumulation'] == 0):

            self.opt.step()
            self.opt.zero_grad()

            if self.exp_dict["use_scheduler"]:
               self.scheduler.step()

        return None


    def predict(self, idx, batch):

        self.model.eval()

        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(imgs)

        _, pred_label = torch.max(logits.data, 1)
        probs = torch.nn.functional.softmax(logits, dim=1)

        self.total += labels.size(0)
        self.correct += (pred_label == labels).sum().item()

        val_loss = self.loss_func(logits, labels)
        self.losses.append(val_loss.item())

        self.labels_list.extend(labels.detach().cpu().numpy())
        self.probs_list.extend(probs.detach().cpu().numpy())

        return None
    

    def epoch_end(self):
        all_probs = np.array(self.probs_list)
        all_labels = np.array(self.labels_list)

        maxprob_list, one_hot_labels = get_maxprob_and_onehot(all_probs, all_labels)
        self.ece_config['num_samples'] = int(len(one_hot_labels))
        
        # Epoch-wise Log
        avg_ece = calc_ece(self.ece_config, maxprob_list, one_hot_labels)
        avg_loss = np.average(self.losses)
        avg_acc = 100 * self.correct / self.total

        # Reset
        del all_probs, all_labels, maxprob_list, one_hot_labels

        # metric
        self.correct = 0
        self.total = 0
        self.losses = []

        # ece
        self.labels_list = []
        self.probs_list = []

        return avg_loss, avg_acc, avg_ece


 