import copy
import numpy as np
import torch
from pyhessian import hessian
from collections import OrderedDict

class HessianCalculator:
    def __init__(self, model, exp_dict):
        self.model = model
        self.exp_dict = exp_dict
        self.calc_gradnorm = exp_dict['calc_gradnorm']
        self.pyhessian_batch_size_per_gpu = exp_dict['pyhessian_batch_size_per_gpu']
        self.pyhessian_sample_size = exp_dict['pyhessian_sample_size']

    # https://github.com/TrustAIoT/CR-SAM/blob/main/metrics/metrics.py
    # https://github.com/TrustAIoT/CR-SAM/blob/3d2d0ee2dd03e51bf02438bc8827a3de00de2ea5/utils/crsam.py#L78
    def grad_norm(self, model, dataloader, lp=2): 
        criterion = torch.nn.CrossEntropyLoss().cuda()
        self.model.eval()
        total_norm = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            labels = labels.long()  # ラベルをLong型に変換
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            # optimizer.zero_grad()
            batch_loss.backward()
            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            total_norm.append(torch.norm(torch.stack([torch.norm(p.grad.detach(), lp) for p in parameters]), lp).item())
        return np.mean(total_norm)


    def calc_hessian_metrics(self, loader, dataset_name):
        # トレースと固有値の計算
        tr_h, eigen_h = self.calc_hessian_from_pyhessian(copy.deepcopy(self.model), loader,
                                                         batch_size=self.exp_dict['pyhessian_batch_size_per_gpu'],
                                                         sample_size=self.exp_dict['pyhessian_sample_size'],
                                                         top_n=self.exp_dict['top_n'])
        # wandbログの準備
        wandb_log_dict = {
            f'{dataset_name}_tr_h': tr_h,
            f'{dataset_name}_eigen_h': eigen_h[0]
        }

        # gradnormの計算
        if self.calc_gradnorm:
            gradnorm = self.grad_norm(self.model, loader)
            wandb_log_dict[f'{dataset_name}_gradnorm'] = gradnorm
        return wandb_log_dict

    def calc_hessian_for_datasets(self, train_loader, val_loader,
                                  ood_dataloaders=None, kinds=None):
        wandb_log_dict = {}
        # Train, Val, Test データセットの計算
        wandb_log_dict.update(self.calc_hessian_metrics(train_loader, 'train'))
        wandb_log_dict.update(self.calc_hessian_metrics(val_loader, 'val'))

        if self.exp_dict['ood_dataset'] == 'None':
            print(f"[calc hessian for test datasets] {self.exp_dict['ood_dataset']} is not supported yet")

        else:
            # OODデータセットの計算（例：imagenet_v2, cifar10_c）
            if ood_dataloaders and kinds:
                test_tr_h, test_eigen_h, test_gradnorm = 0, 0, 0
                for ood_loader, kind in zip(ood_dataloaders, kinds):
                    ood_log = self.calc_hessian_metrics(ood_loader, kind)
                    test_tr_h += ood_log[f'{kind}_tr_h']
                    test_eigen_h += ood_log[f'{kind}_eigen_h']
                    if self.calc_gradnorm:
                        test_gradnorm += ood_log[f'{kind}_gradnorm']
                    wandb_log_dict.update(ood_log)

                num_of_dataloaders = len(ood_dataloaders)
                test_tr_h /= num_of_dataloaders
                test_eigen_h /= num_of_dataloaders
                wandb_log_dict['test_ood_tr_h'] = test_tr_h
                wandb_log_dict['test_ood_eigen_h'] = test_eigen_h
                if self.calc_gradnorm:
                    test_gradnorm /= num_of_dataloaders
                    wandb_log_dict['test_ood_gradnorm'] = test_gradnorm
                    wandb_log_dict['test_ood_tr_h_ce'] = test_tr_h / test_gradnorm
                    wandb_log_dict['test_ood_eigen_h'] = test_eigen_h / test_gradnorm

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