# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None



def get_algorithm_class(algorithm_name):

    # impot algorithms
    from .algo_erm import ERM
    from .algo_erm_vanilla import ERM_Vannila
    from .algo_erm_curv_reg import ERM_CURV_REG
    from .algo_fish import Fish

    algorithm_map = {
        'ERM': ERM,
        'ERM_Vannila': ERM_Vannila,
        'ERM_CURV_REG': ERM_CURV_REG,
        'Fish': Fish,
    }

    if algorithm_name in algorithm_map:
        return algorithm_map[algorithm_name]
    else:
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self):
        super(Algorithm, self).__init__()

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError