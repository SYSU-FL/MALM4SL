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
from utils.loss import *
from dataprocess.data_utils import *
from dataprocess.imbalance_cifar import *


from models.resnet import *
from models.vgg import *


class Client(object):
    def __init__(self, model, loader, server, args, epoch):
        self.model = model
        self.loader = loader
        self.server = server
        self.args = args
        self.epoch = epoch
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def train(self, weight=None):
        if weight is not None:
            self.model.load_state_dict(weight)
        self.model.train()
        count = 0
        for batch_idx, (images, labels) in enumerate(self.loader):
            self.model.zero_grad()
            if self.args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = self.model(images)
            flag = False
            loss, grad = self.server.train(images, outputs, labels, flag)
            outputs.backward(grad)
            self.optimizer.step()
            count += 1
            if self.args.verbose and count == (len(self.loader) - 1):
                print(f"| Data Size: {len(self.loader) * args.local_bs}| loss: {loss.item()}, ")

    
    def get_weight(self,factor=None):
        return self.model.state_dict()
    
    

class Server(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.criterion = NoPeekLoss(args, 0.15)
        if self.args.gpu > -1:
            self.criterion.cuda(self.args.gpu)
        self.acc_loss = []
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, images, inputs, labels, flag):
        input_temp = inputs.clone().detach()
        input_temp.requires_grad_()
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(input_temp)
        loss, dist = self.criterion(images, input_temp, outputs, labels)
        if flag:
            dist = dist.clone().detach().cpu().item()
            self.dcor_val = dist 
        loss.backward()
        grad = copy.deepcopy(input_temp.grad.data)
        self.optimizer.step()
        return loss, grad

    def get_weight(self,factors=None):
        return self.model.state_dict()

    def get_dcor(self):
        return self.dcor_val

class Trainer(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.trainloader = []
    def _train_val_test(self, dataset, idxs):
        batch_size = args.local_bs
        #trainloader = DataLoader(DatasetFragment(dataset, 80), batch_size=batch_size, shuffle=True)
        trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        return trainloader
    def reset(self):
        self.trainloader = []
    def assign(self, idxs_list):
        self.reset()
        for idxs in idxs_list:
            train_loader = self._train_val_test(self.dataset, list(idxs))
            self.trainloader.append(train_loader)

    def train(self, extractor, classifer, epoch):
        index = 1
        server = Server(classifer, self.args)
        #TODO delete the deepcopy
        clients = [Client(copy.deepcopy(extractor), self.trainloader[i], server, self.args, epoch) for i in range(len(self.trainloader))]
        client_w = None
        for client in clients:
            print(f"| Global Round: {epoch} | client index: {index} |")
            client.train(client_w)
            client_w = client.get_weight()
            index += 1
       
        return client_w, server.get_weight()




if __name__ == "__main__":
    args = parse_args()
    if args.mnist:
        data_name = 'mnist'
    else:
        data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    split_type = 'googlesplit' if args.google_split else ''
    dp_flag = 'dp' if args.dp else '' 
    TAG = 'nopeeknn-' + dp_flag + '-' + data_name + '-' + split_type + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    
    logs = []
    if args.mnist:
        train_dataset, test_dataset = get_mnist()
    else:
        if args.cifar100:
            train_dataset, test_dataset  = get_cifar100(True)
        else:
            train_dataset, test_dataset  = get_cifar10(True)
    user_groups = random_avg_strategy(train_dataset, args.num_users)
    tf_log = Logs(TAG)
    logs_file = TAG
    client_part, server_part = get_model(args)

    trainer = Trainer(args, train_dataset)
    
    for epoch in range(args.epochs):
        local_weights = []
        idxs_users = random_assign(args)
        idx_list = [user_groups[i] for i in idxs_users]
        trainer.assign(idx_list)
        extractor_w, classifer_w = trainer.train(copy.deepcopy(client_part), copy.deepcopy(server_part), epoch)
        client_part.load_state_dict(extractor_w)
        server_part.load_state_dict(classifer_w)
        if args.cifar100:
            #testdataset = DatasetFragment(test_dataset, 80)
            testdataset = test_dataset
            test_acc1, test_acc5, test_loss = test_inference4split4cifar100(args, client_part, server_part, testdataset)
            tf_log.writer.add_scalar('test/Cifar100_Acc/', test_acc1, epoch)
            tf_log.writer.add_scalar('test/Cifar100_Loss/', test_loss, epoch)
            print("|---- Test Accuracy1: {:.2f}%, Test Accuracy5: {:.2f}%".format(100*test_acc1, 100*test_acc5))
            log_obj = {
                'test_acc1': "{:.2f}%".format(100*test_acc1),
                'test_acc5': "{:.2f}%".format(100*test_acc5),
                'loss': test_loss,
                'epoch': epoch
                }
        else:
            test_acc, test_loss = test_inference4split(args, client_part, server_part, test_dataset)
            tf_log.writer.add_scalar('test/MNIST_Acc/', test_acc, epoch)
            tf_log.writer.add_scalar('test/MNIST_Loss/', test_loss, epoch)

            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            log_obj = {
                    'test_acc': "{:.2f}%".format(100*test_acc),
                    'loss': test_loss,
                    'epoch': epoch
                }   
        logs.append(log_obj)
    if args.cifar100:
        save_cifar100_logs(logs, TAG,  args)
    else:
        save_logs(logs, TAG,  args)


    