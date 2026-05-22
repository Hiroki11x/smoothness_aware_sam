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
from opt import canonical_optimizer_name, requires_previous_model

from .base_algorithm import Algorithm as Algorithm


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model, prev_model, opt, prev_opt, loss_func, scheduler, ece_config, exp_dict):

        super(ERM, self).__init__()

        # config
        self.exp_dict = exp_dict
        self.device = exp_dict['device']
        self.ece_config = ece_config

        # training
        self.loss_func = loss_func
        self.opt = opt
        self.prev_opt = prev_opt

        self.scheduler = scheduler
        self.use_clip_gradnorm = exp_dict['use_clip_gradnorm']
        self.opt_name = canonical_optimizer_name(exp_dict['opt'])
        self.uses_prev_model = requires_previous_model(exp_dict['opt'])

        self.model = model
        self.prev_model = prev_model

        if self.uses_prev_model and (self.prev_model is None or self.prev_opt is None):
            raise ValueError(f"{self.opt_name} requires prev_model and prev_opt")

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

        # AD-SGD
        if self.uses_prev_model:
            prev_logits = self.prev_model(imgs) 
            
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
        
        # Use CLIP gradnorm for Focal Loss
        if self.use_clip_gradnorm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)

        if ((idx + 1) % self.exp_dict['num_accumulation'] == 0):

            # SAM
            if self.opt_name == 'sam':

                self.opt.first_step(zero_grad=True)
                self.loss_func(self.model(imgs), labels).backward()
                self.opt.second_step(zero_grad=True)

            # AD-SGD
            elif self.opt_name == 'sasgd':

                prev_train_loss = self.loss_func(prev_logits, labels) 
                prev_train_loss.backward() 
                self.opt.compute_dif_norms(self.prev_opt)
                self.prev_model.load_state_dict(self.model.state_dict())
                
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.prev_opt.zero_grad(set_to_none=True)

            # AD-SGD-SAM
            elif self.opt_name == 'sasam':
                prev_train_loss = self.loss_func(prev_logits, labels) # AD-SGD
                prev_train_loss.backward() # AD-SGD
                self.opt.base_optimizer.compute_dif_norms(self.prev_opt)
                self.prev_model.load_state_dict(self.model.state_dict())

                self.opt.first_step(zero_grad=True)
                self.loss_func(self.model(imgs), labels).backward()
                self.opt.second_step(zero_grad=True)
                self.prev_opt.zero_grad(set_to_none=True)

            # Momentum SGD / Adam etc.
            else:
                self.opt.step()
                self.opt.zero_grad()

            if self.exp_dict["use_scheduler"]:
               self.scheduler.step()

        return None


    def predict(self, idx, batch):

        self.model.eval()

        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device).long() 
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

        # reset optim
        self.opt.zero_grad(set_to_none=True)
        if self.uses_prev_model:
            self.prev_opt.zero_grad(set_to_none=True)

        return avg_loss, avg_acc, avg_ece


 
