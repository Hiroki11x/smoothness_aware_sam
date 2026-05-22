import argparse
import wandb
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.autograd import grad  # この行を追加
import numpy as np

from hessian import calculator

# ネットワーク定義
class SimpleNet_MNIST(nn.Module):
    def __init__(self, hidden_size=100):
        super(SimpleNet_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleNet_CIFAR10(nn.Module):
    def __init__(self, hidden_size=100):
        super(SimpleNet_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_calc_hessian_trace(model, data, target, criterion, device, maxIter=100):
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


def calculate_ece(probabilities, labels, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    confidences, predictions = probabilities.max(dim=1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # ビン内のサンプルを選択
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.any():
            # ビン内の正解率
            accuracy_in_bin = labels[in_bin].eq(predictions[in_bin]).float().mean()
            # ビン内の平均確信度
            avg_confidence_in_bin = confidences[in_bin].mean()
            # ECEへの寄与
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * in_bin.float().mean()

    return ece


# 精度とECEの計算
def compute_metrics(model, data_loader, criterion, device):
    model.eval()

    correct = 0
    total = 0
    ece = 0.0
    loss = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss += criterion(outputs, target).item() * data.size(0)

            softmax_outputs = torch.softmax(outputs, dim=1)
            ece += calculate_ece(softmax_outputs, target).item() * data.size(0)

    accuracy = correct / total
    ece /= total
    loss /= total
    return loss, accuracy, ece


def train(exp_dict):

    reg_lambda = exp_dict['reg_lambda']
    hidden_size = exp_dict['hidden_size']
    epochs = exp_dict['epochs']
    batch_size = exp_dict['batch_size']
    lr = exp_dict['lr']

    # ネットワーク、損失関数、オプティマイザの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル・データセットの設定
    if exp_dict['dataset'] == 'mnist':

        model = SimpleNet_MNIST(hidden_size=hidden_size).to(device)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    elif exp_dict['dataset'] == 'cifar10':

        model = SimpleNet_CIFAR10(hidden_size=hidden_size).to(device)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    else:
        raise ValueError('dataset should be mnist or cifar10')

    if torch.cuda.device_count() > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=exp_dict['weight_decay'])
    

    # トレーニングループ
    for epoch in range(epochs):
        hessian_trace = 0.0
        total_batches = 0

        print(f'epoch: {epoch} start')
        wandb_log_dict = {}
        wandb_log_dict['epoch'] = epoch

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # ここでHutchinson法を用いてHessianのトレースを推定 (非常に時間がかかる)
            tr_h = my_calc_hessian_trace(model, data, target, criterion, device)

            # 損失計算: クロスエントロピー損失
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            total_loss = loss + reg_lambda * tr_h

            # 逆伝播とパラメータの更新
            total_loss.backward()
            optimizer.step()

            hessian_trace += tr_h
            total_batches += 1

        if epoch % args.eval_interval == 0:
            
            # Hessianのトレースを計算
            avg_hessian_trace = hessian_trace / total_batches
            loss, accuracy, ece = compute_metrics(model, train_loader, criterion, device)
            
            wandb_log_dict['train_loss'] = loss
            wandb_log_dict['train_acc'] = accuracy
            wandb_log_dict['train_ece'] = ece
            wandb_log_dict['train_tr_h'] = avg_hessian_trace

            # バリデーションデータでの評価
            hessian_trace = 0.0
            total_batches = 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                tr_h = my_calc_hessian_trace(model, data, target, criterion, device)
                hessian_trace += tr_h
                total_batches += 1

            avg_hessian_trace = hessian_trace / total_batches
            loss, accuracy, ece = compute_metrics(model, val_loader, criterion, device)
            wandb_log_dict['val_loss'] = loss
            wandb_log_dict['val_acc'] = accuracy
            wandb_log_dict['val_ece'] = ece
            wandb_log_dict['val_tr_h'] = avg_hessian_trace

            # wandbにログを送信
            wandb.log(wandb_log_dict)


    return accuracy, ece, avg_hessian_trace


def build_wandb_init_kwargs(config, project, name, entity):
    kwargs = {
        "config": config,
        "project": project,
        "name": name,
    }
    if entity:
        kwargs["entity"] = entity
    return kwargs

if __name__ == "__main__":
    print("Start Experiment with MNIST dataset")
    parser = argparse.ArgumentParser(description='Calibration and Out-of-Distribution Project')

    # Environmental Setting
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--reg_lambda', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--eval_interval', type=int, default=1)

    # Wandb Configuration
    parser.add_argument("--wandb_exp_id", type=int, default=99999999)
    parser.add_argument("--wandb_entity", type=str, default='', help="optional wandb team or organization")
    parser.add_argument("--wandb_project_name", type=str, default='default_project', help="should include dataset and model")
    parser.add_argument('--wandb_offline', action = 'store_true')
    args = parser.parse_args()

    # ================ wandb ================ 
    wandb_exp_name = f'exp_id-{args.wandb_exp_id}_reg_lambda-{args.reg_lambda}-hidden_size-{args.hidden_size}-epochs-{args.epochs}-batch_size-{args.batch_size}-lr-{args.lr}'

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(**build_wandb_init_kwargs(
        config=args,
        project=args.wandb_project_name,
        name=wandb_exp_name,
        entity=args.wandb_entity,
    ))

    print('\nWandb Setting:')
    print(f'\twandb_project_name: f{args.wandb_project_name}')
    print(f'\twandb_exp_name: f{wandb_exp_name}')

    # ============= train config =============
    exp_dict = wandb.config

    print('\nExperimental Configuration:')
    for k, v in sorted(exp_dict.items()):
        print('\t{}: {}'.format(k, v))

    # ================ execution ================ 
    train(exp_dict)
