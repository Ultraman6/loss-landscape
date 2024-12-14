"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
from tqdm import tqdm


def eval_loss(net, criterion, loader, device):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        device: torch device (e.g., 'cuda' or 'cpu')
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0  # number of samples

    net.to(device)  # 将模型移动到指定设备
    net.eval()

    # 使用 tqdm 包裹数据加载器
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Evaluating")

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in progress_bar:
                batch_size = inputs.size(0)
                total += batch_size

                # 将数据移动到指定设备
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

                # 更新进度条信息
                progress_bar.set_postfix({
                    "Loss": f"{total_loss / total:.4f}",
                    "Accuracy (%)": f"{100. * correct / total:.2f}"
                })

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in progress_bar:
                batch_size = inputs.size(0)
                total += batch_size

                # 创建 one-hot 的 targets 并移动到设备
                one_hot_targets = torch.zeros(batch_size, 10, device=device).scatter_(1, targets.view(batch_size, 1), 1.0)

                inputs, one_hot_targets = inputs.to(device), one_hot_targets

                outputs = F.softmax(net(inputs), dim=1)  # 添加 dim 参数
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

                # 更新进度条信息
                progress_bar.set_postfix({
                    "Loss": f"{total_loss / total:.4f}",
                    "Accuracy (%)": f"{100. * correct / total:.2f}"
                })
    # 返回的就是模型在数据集上的平均损失和准确率，相当于全梯度下降（GD）直接平均聚合了
    return total_loss / total, 100. * correct / total
