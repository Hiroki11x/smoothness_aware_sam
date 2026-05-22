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

class ERM_Vannila(Algorithm):

    """
    Vanilla Empirical Risk Minimization (ERM_Vannila)
    """

    def __init__(self, model, opt, loss_func, scheduler, ece_config, exp_dict):

        super(ERM_Vannila, self).__init__()

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


 