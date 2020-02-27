#! /usr/bin/env python
#! coding:utf-8

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # SGD/ADAM
import torchvision
import torchvision.transforms as transforms

# plain python
import time
import os
import argparse
from tqdm import tqdm
import subprocess  # Popen
from pathlib import Path
import numpy as np
try:
    import visdom
    has_visdom = True
except ImportError:
    has_visdom = False


# user defined
from models.lenet5 import LeNet

# the project root dir path
root_path = Path(__file__)

# argparse
parser = argparse.ArgumentParser(description="a skeleton for training/testing")
parser.add_argument('--lr', '-l', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epoch', '-e', default=201, type=int, help='epoches')
parser.add_argument('--batchsize', '-b', default=64, type=int, help='batchsize')

if(has_visdom):
    # visdom group
    group = parser.add_argument_group("visdom group")
    group.add_argument('--server', '-s', type=str, default='http://localhost',
                       help='visdom server url/ip address')
    group.add_argument('--port', '-p', type=int, default=8097,
                       help='visdom server port')
    group.add_argument('--base_url', '-b', type=str, default='/',
                       help='Visdom Base url')
    group.add_argument('--env_name', '-n', type=str, default='env' + str(int(time.time()//60)),
                       help='Visdom env name,default is env_time_from_epoch')
args = parser.parse_args()

# system monitor
if(has_visdom):
    child_process = subprocess.Popen(
        ['python', root_path.joinpath('utils', 'sysem_visdom_monitor.py'),
         '-s', args.server, '-p', args.port, '-b', args.base_url, '-n', args.env_name])

# fasionMnist
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNISt',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batchsize, shuffle=True, num_workers=4)

test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNISt',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False, num_workers=4)
# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: %s" % (device))
# init
best_acc = 0
start_epoch = 0

# model load
print("==> Model loading")
net = LeNet()
net = net.to(device)

# training
criterion = nn.CrossEntropyLoss()
SGD_flag = True
if SGD_flag:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=00.9, weight_decay=5e-4)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(start_epoch, args.epoch):
    net.train()
    batch_loss_list = []
    correct = 0
    total = 0
    for _, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)  # batch loss
        batch_loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    epoch_mean_loss = np.mean(batch_loss_list)  # 这个epoch里面，每一个batch的loss的平均值
    print("TRAIN: epoch:%d,Loss:%.3f,Acc:%.3f%%,(%d/%d)" %
          (epoch, epoch_mean_loss, 100. * correct/total, correct, total))
    # test
    # global best_acc
    if(epoch != 0 and epoch % 5 == 0):
        batch_loss_list = []
        total = 0
        correct = 0
        net.eval()
        with torch.no_grad():
            for _, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), smoothing=0.9):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)  # batch loss
                batch_loss_list.append(loss.item())

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        epoch_mean_loss = np.mean(batch_loss_list)
        # save checkoints
        acc = 100. * correct/total
        if(acc > best_acc):
            print("==> saving checkpoints")
            best_acc = acc
            state = {'epoch': epoch, 'model_state_dict': net.state_dict(),
                     'acc': acc}
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, 'checkpoints/best.pth')
            best_acc = acc
        print("TEST: epoch:%d,Loss:%.3f,Acc:%.3f%%,(%d/%d)" %
              (epoch, epoch_mean_loss, acc, correct, total))
