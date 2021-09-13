import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import argparse
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision

from utils.options import *
from utils.function import *
from utils.evaluation import *
from dataprocess.data_utils import *
from dataprocess.imbalance_cifar import *


from models.resnet import *
from models.vgg import *
from models.mlp import *
from models.cnn import *

def test_inference(args, net, test_dataset):
    """ Returns the test accuracy and loss.
    """
    net.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    accuracy = correct/total
    return accuracy, loss

"""
for single training in cifar100
"""
def test_inference4cifar100(args, net, test_dataset):
    """ Returns the test accuracy and loss.
    """
    net.eval()
    loss, total, correct_1, correct_5 = 0.0, 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1 
            correct_1 += correct[:, :1].sum()

    accuracy_1 = correct_1 / len(testloader.dataset)
    accuracy_5 = correct_5 / len(testloader.dataset)
    return accuracy_1, accuracy_5, loss



if __name__ == "__main__":
    args = parse_args()
    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    split_type = 'googlesplit' if args.google_split else ''
    TAG = 'local-' + data_name + '-' + args.name + '-' + split_type + '-'
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    logs = []
    if args.mnist:
        train_dataset, test_dataset = get_mnist()
    else:
        if args.cifar100:
            train_dataset, test_dataset  = get_cifar100(False)
        else:
            train_dataset, test_dataset  = get_cifar10(False)
    
    logs_file = TAG
    tf_log = Logs(TAG)
    #net = ResNet18(num_classes)
    net = ResNet34(num_classes)
    #net = get_vgg16(num_classes)
    #net = MLP()
    #net = CNNMnist(num_classes)

    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        net.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    
    for epoch in range(args.epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(args.gpu), labels.to(args.gpu)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            if batch_idx % 1000 == 0:
                print(f"epoch: {epoch}, Loss: {loss}")
            loss.backward()
            optimizer.step()
        #measure_logs.append(get_w_diff(net, pre_model))
        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4cifar100(args, net, test_dataset)
            tf_log.writer.add_scalar('test/Cifar100_Acc/', test_acc1, epoch)
            tf_log.writer.add_scalar('test/Cifar100_Loss/', test_loss, epoch)
            print(f"|----Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
            log_obj = {
                'test_acc1': "{:.2f}%".format(100*test_acc1),
                'test_acc5': "{:.2f}%".format(100*test_acc5),
                'loss': test_loss,
                'epoch': epoch 
                #'num_batch':num_batch
                }
            logs.append(log_obj)
        else:
            test_acc, test_loss = test_inference(args, net, test_dataset)
            tf_log.writer.add_scalar('test/MNIST_Acc/', test_acc, epoch)
            tf_log.writer.add_scalar('test/MNIST_Loss/', test_loss, epoch)
            print(f"|----Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
            log_obj = {
                'test_acc': "{:.2f}%".format(100*test_acc),
                'loss': test_loss,
                'epoch': epoch 
                #'num_batch':num_batch
                }
            logs.append(log_obj)
    if args.cifar100:
        save_cifar100_logs(logs, TAG, args)
    else:
        save_logs(logs, TAG,  args)
    #save_measure(measure_logs, TAG)

    