import copy
import numpy as np
import torch
from pyhessian import hessian
from collections import OrderedDict

class HessianCalculator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    # https://github.com/TrustAIoT/CR-SAM/blob/main/metrics/metrics.py
    # https://github.com/TrustAIoT/CR-SAM/blob/3d2d0ee2dd03e51bf02438bc8827a3de00de2ea5/utils/crsam.py#L78
    def grad_norm(self, model, dataloader, lp=2): 
        criterion = torch.nn.CrossEntropyLoss().cuda()
        self.model.eval()
        total_norm = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            labels = labels.long() 
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            total_norm.append(torch.norm(torch.stack([torch.norm(p.grad.detach(), lp) for p in parameters]), lp).item())
        return np.mean(total_norm)


    def calc_hessian_metrics(self, loader, suffix):
        tr_h, eigen_h = self.calc_hessian_from_pyhessian(copy.deepcopy(self.model), loader,
                                                         batch_size=self.config.pyhessian_batch_size_per_gpu,
                                                         sample_size=self.config.pyhessian_sample_size,
                                                         top_n=self.config.pyhessian_top_n)
        wandb_log_dict = {
            f'{suffix}_tr_h': tr_h,
            f'{suffix}_eigen_h': eigen_h[0]
        }

        if self.config.calc_gradnorm:
            gradnorm = self.grad_norm(self.model, loader)
            wandb_log_dict[f'{suffix}_gradnorm'] = gradnorm
        return wandb_log_dict


    def calc_hessian_for_datasets(self, train_loader, val_loader, test_loader, suffix=''):
        wandb_log_dict = {}
        wandb_log_dict.update(self.calc_hessian_metrics(train_loader, f'{suffix}train'))
        wandb_log_dict.update(self.calc_hessian_metrics(val_loader, f'{suffix}val'))
        wandb_log_dict.update(self.calc_hessian_metrics(test_loader, f'{suffix}test'))
        return wandb_log_dict


    def build_hessian_dataloader(self, data_loader, 
                                 hessian_sample_size, hessian_batch_size):
        
        batch_num = hessian_sample_size // hessian_batch_size
        if batch_num == 1:
            for inputs, labels in data_loader:
                hessian_dataloader = (inputs, labels)
                break
        else:
            hessian_dataloader = []
            for i, (inputs, labels) in enumerate(data_loader):
                hessian_dataloader.append((inputs, labels))
                if i == batch_num - 1:
                    break

        return hessian_dataloader


    # Calculation of H from PyHessian
    def calc_hessian_from_pyhessian(self, model, data_loader, batch_size, sample_size, top_n):

        model.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        hessian_dataloader = self.build_hessian_dataloader(data_loader,
                                                    sample_size,
                                                    batch_size)
        
        hessian_comp = hessian(model,
                            loss_fn,
                            dataloader=hessian_dataloader,
                            cuda=True)

        traces = hessian_comp.trace()
        trace_h = np.mean(traces)

        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_n)

        del hessian_comp
        del data_loader
        del hessian_dataloader

        return trace_h, top_eigenvalues