from pathlib import Path
from typing import Optional

import torch
import wandb
from init import get_starting_time, init, optimizer_to
from torch.optim.swa_utils import update_bn
from tqdm import tqdm

from example.config import TrainConfig
from example.dataset import get_train_valid_test_loader
from example.init import (get_criterion, get_device, get_lr_scheduler,
                          get_model, get_optimizer, init_wandb_results)
from example.optimization_step import optimization_step
from swa_utils import MultipleSWAModels

import copy
from example.hessian import HessianCalculator

class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
    ):
        init(config=config) # initialize wandb
        self.steps = 0
        self.config: TrainConfig = config
        self.device = get_device(config=config)
        self.no_tqdm = config.no_tqdm
        self.model = (
            get_model(config=config).to(self.device)
            if model is None
            else model.to(self.device)
        )

        self.optimizer = (
            get_optimizer(
                optimizer_name=self.config.optimizer_name, config=self.config, model=self.model
            )
            if optimizer is None
            else optimizer
        )

        # smoothness aware
        if self.config.optimizer_name == "sasgd" or self.config.optimizer_name == "sasam":
            self.config.is_sa = True
        self.prev_model = None
        self.prev_optimizer = None
        if self.config.is_sa:
            self.prev_model = copy.deepcopy(self.model)
            self.prev_optimizer = get_optimizer(
                optimizer_name=self.config.optimizer_name, config=self.config, model=self.prev_model
            )
            self.config.lr_scheduler="none"

        (
            dataset_size,
            self.train_loader,
            self.valid_loader,
            self.test_loader,
        ) = get_train_valid_test_loader(config=self.config)

        # Caluculate Num of Total Steps for Learning Rate
        self.config.steps_per_epoch = dataset_size // self.config.batch_size 
        
        # if self.config.model == "vit":
        #     # https://arxiv.org/pdf/2202.00661
        #     self.config.total_steps = 12500
        #     self.config.max_epochs = self.config.total_steps // self.config.steps_per_epoch
        # else:
        #     self.config.total_steps = self.config.max_epochs * self.config.steps_per_epoch

        self.config.total_steps = self.config.max_epochs * self.config.steps_per_epoch

        self.scheduler = get_lr_scheduler(config=self.config, optimizer=self.optimizer)
        
        self.criterion = get_criterion(config=self.config)
        self.scheduler = (
            get_lr_scheduler(config=self.config, optimizer=self.optimizer)
            if scheduler is None
            else scheduler
        )
        self.start_time = get_starting_time()

        # For switching optimizer (warmup and Smoothness Aware)
        if config.is_warmup and "sa" in self.config.optimizer_name:  # Modify
            self.config.optimizer_name_after_warmup = self.config.optimizer_name
            self.optimizer_after_warmup = self.optimizer
            
            self.config.optimizer_name = "adamw"
            self.optimizer = get_optimizer(
                optimizer_name="adamw", config=self.config, model=self.model
            )
            
        optimizer_to(optim=self.optimizer, device=self.device)

        if self.config.is_swa: # Modify
            self.swa_models = MultipleSWAModels(
                base_model=self.model,
                device=self.device,
                max_epochs=self.config.max_epochs,
                starts=self.config.swa_starts,
            )
            init_wandb_results(swa_start_times=self.swa_models.swa_start_times)
        else:
            init_wandb_results(swa_start_times=None)
        self.max_val_acc, self.max_val_acc_swa = 0.0, 0.0
        self.epoch = 1
        self.output_path = f"{config.output_path}{config.dataset_name}/{config.model}/{config.optimizer_name}/{self.start_time}/"
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def switch_optimizer(self) -> None:
        # check config.is_warmup == True
        assert self.config.is_warmup
        assert "sa" in self.optimizer_name_after_warmup

        print(f"switch optimizer from {self.config.optimizer_name} to {self.config.optimizer_name_after_warmup} / steps : {self.steps}")
        self.config.optimizer_name = self.config.optimizer_name_after_warmup
        self.optimizer = self.optimizer_after_warmup
        optimizer_to(optim=self.optimizer_after_warmup, device=self.device)

    def train_and_test_model(
        self,
    ) -> None:
        """Training loop with SWA."""
        for epoch in range(1, self.config.max_epochs + 1):
            self.train_for_one_epoch()
            self.eval_models()
            self.epoch += 1

    def log_metrics(self, metrics) -> None:
        wandb.log(metrics)

    def is_calc_hessian(self) -> bool:
        return self.config.calc_hessian and (((self.epoch % self.config.pyhessian_interval) == 0) or (self.config.max_epochs == self.epoch))

    # Add for Hessian Calculation
    def eval_models_hessian(self, model, suffix) -> None:

        hessian_calculator = HessianCalculator(model, self.config)
        hessian_log_dict = hessian_calculator.calc_hessian_for_datasets(
                        train_loader=self.train_loader, 
                        val_loader=self.valid_loader, 
                        test_loader=self.test_loader, 
                        suffix=suffix
        )
        hessian_log_dict['epoch'] = self.epoch
        self.log_metrics(hessian_log_dict)
                

    def eval_models(self) -> None:
        self.eval_model(model=self.model)
        if self.config.is_swa: # Modify
            self.eval_swa_models()

    def eval_swa_models(self) -> None:
        for model_dict in self.swa_models.models:
            swa_model, swa_start = model_dict["model"], model_dict["start"]
            if self.epoch >= swa_start:
                suffix = f"_swa_{swa_start}"
                update_bn(loader=self.train_loader, model=swa_model, device=self.device)
                self.eval_model(model=swa_model, suffix=suffix)

    def eval_model(self, model: torch.nn.Module, suffix: str = "") -> None:
        val_metrics = self.model_evaluation(
            model=model,
            loader=self.valid_loader,
        )
        log_metrics = {}
        if val_metrics["acc"] > wandb.run.summary[f"valid{suffix}_best_acc"]:
            test_metrics = self.model_evaluation(
                model=model,
                loader=self.test_loader,
            )
            if self.config.validate_best_model_saved: # Modify
                self.save_ckpt(
                    model=model,
                    file_name=f"best_valid{suffix}.pth",
                )
            log_metrics.update({f"test{suffix}": test_metrics, "epoch": self.epoch})

            wandb.run.summary[f"valid{suffix}_best_acc"] = val_metrics["acc"]
            wandb.run.summary[f"test{suffix}_best_acc"] = test_metrics["acc"]
        log_metrics.update({f"valid{suffix}": val_metrics, "epoch": self.epoch})
        
        self.log_metrics(log_metrics)

        # Calc Hessian?
        if self.is_calc_hessian():
            self.eval_models_hessian(model=model, suffix=suffix)

        model.train()

    def train_for_one_epoch(
        self,
    ) -> None:
        """Trains for one epoch."""
        self.model.train()
        epoch_loss = epoch_acc = 0.0
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False, disable=self.no_tqdm)
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.steps += 1

            step_dict = optimization_step(
                model=self.model,
                prev_model=self.prev_model,
                optimizer=self.optimizer,
                prev_optimizer=self.prev_optimizer,
                inputs=inputs,
                labels=labels,
                criterion=self.criterion,
                config=self.config,
            )
            
            if self.config.is_warmup and self.steps == self.config.warmup_steps: # Warmup - Switch Optimizer
                self.switch_optimizer()
            elif (self.config.is_warmup and self.steps < self.config.warmup_steps): # Warmup - Even SA
                self.scheduler.step()
            elif not self.config.is_sa: # No SA
                self.scheduler.step()
            
            predictions = step_dict["predictions"]
            loss = step_dict["loss"]
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == labels
            epoch_loss += loss
            epoch_acc += correct.sum()

        epoch_loss /= len(self.train_loader.dataset)
        epoch_acc /= len(self.train_loader.dataset)

        if "sam" in self.config.optimizer_name:
            current_lr = self.optimizer.base_optimizer.param_groups[0]['lr']
            current_rho = self.optimizer.param_groups[0]['rho']
        else:
            current_lr = self.optimizer.param_groups[0]['lr']
            current_rho = None
            
        self.log_metrics(
            {"train": {"acc": epoch_acc, "loss": epoch_loss}, "epoch": self.epoch, "lr": current_lr, "rho": current_rho}
        )
        print(
            f"Epoch {self.epoch}: Train loss: {epoch_loss:.6f}, Train acc: {epoch_acc:.6f}, LR: {current_lr:.6f}, Rho: {current_rho}"
        )
        if self.config.is_swa: # Modify
            self.swa_models.update_parameters(base_model=self.model, epoch=self.epoch)


    from typing import Dict

    def model_evaluation(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Computes loss and acc for given dataloader."""
        model.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(
                tqdm(loader, desc="Validation: ", leave=False, disable=self.no_tqdm)
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                predictions = model.forward(inputs)
                total_loss += self.criterion(input=predictions, target=labels).item()
                total_acc += (torch.argmax(predictions, dim=1) == labels).sum().item()
        total_loss /= len(loader.dataset)
        total_acc /= len(loader.dataset)
        return {"loss": total_loss, "acc": total_acc}


    def save_ckpt(
        self,
        model: torch.nn.Module,
        file_name: str,
    ) -> None:
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        state = {
            "epoch": self.epoch,
            "model_state_dict": model_cpu,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, self.output_path + file_name)
