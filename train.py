from __future__ import print_function
import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.parallel
from tqdm import tqdm
import model_loader
from cifar10 import dataloader


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training
def train(trainloader, net, criterion, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # 使用 tqdm 包裹 trainloader
    progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc="Training")

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in progress_bar:
            batch_size = inputs.size(0)
            total += batch_size
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

            # 更新进度条信息
            progress_bar.set_postfix({
                "Loss": f"{train_loss / total:.4f}",
                "Accuracy (%)": f"{100. * correct / total:.2f}"
            })

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in progress_bar:
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            inputs, one_hot_targets = inputs.to(device), one_hot_targets.to(device)

            outputs = F.softmax(net(inputs), dim=1)  # 添加 dim 参数
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

            # 更新进度条信息
            progress_bar.set_postfix({
                "Loss": f"{train_loss / total:.4f}",
                "Accuracy (%)": f"{100. * correct / total:.2f}"
            })

    return train_loss / total, 100. * correct / total


def test(testloader, net, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # 使用 tqdm 包裹 testloader
    progress_bar = tqdm(enumerate(testloader), total=len(testloader), desc="Testing")

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in progress_bar:
            batch_size = inputs.size(0)
            total += batch_size

            # 将数据移动到设备
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * batch_size

            # 计算预测结果
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

            # 更新进度条信息
            progress_bar.set_postfix({
                "Loss": f"{test_loss / total:.4f}",
                "Accuracy (%)": f"{100. * correct / total:.2f}"
            })

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in progress_bar:
            batch_size = inputs.size(0)
            total += batch_size

            # 创建 one-hot 的 targets
            one_hot_targets = torch.zeros(batch_size, 10).scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()

            # 将数据移动到设备
            inputs, one_hot_targets = inputs.to(device), one_hot_targets.to(device)

            # 前向传播
            outputs = F.softmax(net(inputs), dim=1)
            loss = criterion(outputs, one_hot_targets)
            test_loss += loss.item() * batch_size

            # 计算预测结果
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).cpu().sum().item()

            # 更新进度条信息
            progress_bar.set_postfix({
                "Loss": f"{test_loss / total:.4f}",
                "Accuracy (%)": f"{100. * correct / total:.2f}"
            })

    return test_loss / total, 100. * correct / total

def name_save_folder(args):
    save_folder = args.model + '_' + str(args.optimizer) + '_lr=' + str(args.lr)
    if args.lr_decay != 0.1:
        save_folder += '_lr_decay=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mom=' + str(args.momentum)
    save_folder += '_save_epoch=' + str(args.save_epoch)
    if args.loss_name != 'crossentropy':
        save_folder += '_loss=' + str(args.loss_name)
    if args.noaug:
        save_folder += '_noaug'
    if args.raw_data:
        save_folder += '_rawdata'
    if args.label_corrupt_prob > 0:
        save_folder += '_randlabel=' + str(args.label_corrupt_prob)
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)

    return save_folder

if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=10, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')
    parser.add_argument('--proto', type=bool, default=False, help='whether to use protonet')
    # model parameters
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--model', '-m', default='resnet18')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')
    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')

    # protonet parameters
    parser.add_argument('--classes_per_it_tr', default=2, type=int, help='number of classes per iteration for training')
    parser.add_argument('--num_support_tr', default=4, type=int, help='number of support samples per class for training')
    parser.add_argument('--num_query_tr', default=12, type=int, help='number of support samples per class for training')

    parser.add_argument('--classes_per_it_val', default=5, type=int, help='number of classes per iteration for testing')
    parser.add_argument('--num_support_val', default=5, type=int, help='number of support samples per class for testing')
    parser.add_argument('--num_query_val', default=5, type=int, help='number of support samples per class for testing')

    args = parser.parse_args()

    print('\nLearning Rate: %f' % args.lr)
    print('\nDecay Rate: %f' % args.lr_decay)

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    lr = args.lr  # current learning rate
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    save_folder = name_save_folder(args)
    if not os.path.exists('trained_nets/' + save_folder):
        os.makedirs('trained_nets/' + save_folder)

    f = open('trained_nets/' + save_folder + '/log.out', 'a', 1)

    trainloader, testloader = dataloader.load_dataset(args)

    if args.label_corrupt_prob and not args.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.dataset, args.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args.dataset, args.model)
        print(net)
        init_params(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()
    net.to(device)
    criterion.to(device)
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_opt:
        checkpoint_opt = torch.load(args.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])
    # record the performance of initial model
    if not args.resume_model:
        train_loss, train_err = test(trainloader, net, criterion, device)
        test_loss, test_err = test(testloader, net, criterion, device)
        status = 'e: %d loss: %.5f train_acc: %.3f test_top1: %.3f test_loss %.5f \n' % (0, train_loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        state = {
            'acc': 100 - test_err,
            'epoch': 0,
            'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        } # 初始模型
        torch.save(state, 'trained_nets/' + save_folder + '/model_0.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_0.t7')
        net.load_state_dict(state['state_dict'])

    for epoch in range(start_epoch, args.epochs + 1):
        loss, train_acc = train(trainloader, net, criterion, optimizer, device)
        test_loss, test_acc = test(testloader, net, criterion, device)

        status = 'e: %d loss: %.5f train_acc: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_acc, test_acc, test_loss)
        print(status)
        f.write(status)

        # Save checkpoint.
        if epoch == 1 or epoch % args.save_epoch == 0 or epoch == 150:
            state = {
                'acc': test_acc,
                'epoch': epoch,
                'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict(),
            }
            opt_state = {
                'optimizer': optimizer.state_dict()
            }

            print(f"Epoch {epoch}: Saving model with state_dict:")
            for key, value in state['state_dict'].items():
                print(f"{key}: {value.shape}")

            # 训练后模型
            torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
            torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')

        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

    f.close()
