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


label_mat = []

class Client(object):
    def __init__(self, model, loader, server, args, num_per_cls=None):
        self.model = model
        self.loader = loader
        self.server = server
        self.args = args
        #self.num_per_cls = num_per_cls
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, weight=None):
        if weight is not None:
            self.model.load_state_dict(weight)
        self.model.train()
        count = 0
        
        for batch_idx, (images, labels) in enumerate(self.loader):
            self.model.zero_grad()
            if self.args.gpu > -1:
                images, labels = images.cuda(self.args.gpu), labels.cuda(self.args.gpu)
            outputs = self.model(images)
            outputs, confused_labels, lams = self._mixup(outputs, labels, self.args.mix_num)
           
            loss, grad = self.server.train(outputs, confused_labels, lams)
            outputs.backward(grad)
            self.optimizer.step()
            count += 1
            if self.args.verbose and count == (len(self.loader) - 1):
                print(f"| Data Size: {len(self.loader) * args.local_bs}| loss: {loss.item()}, ")

    def _mixup(self, x, y, label_num=3):
        x_ = x
        lams = []
        confused_labels = [y]
        batch_size = x.size()[0]
        for i in range(label_num-1):
            lam = np.random.beta(1., 1.)
            index = torch.randperm(batch_size)
            mixed_x = lam * x_ + (1 - lam) * x[index, :]
            lams.append(lam)
            confused_labels.append(y[index])
            x_ = mixed_x
        return mixed_x, confused_labels, lams



    
    def get_weight(self,factor=None):
        return self.model.state_dict()
    
    def get_mean_dcor(self):
        return sum(self.dcor_list)/ len(self.dcor_list)
    
    def get_dcor(self):
        return self.dcor_val

class Server(object):

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.ml = nn.MultiLabelSoftMarginLoss()
        if self.args.gpu > -1:
            self.ce.cuda(self.args.gpu)
            self.kl.cuda(self.args.gpu)
            self.ml.cuda(self.args.gpu)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def _labels2onehot(self, labels):
        class_num = 10 if not self.args.cifar100 else 100
        
        batch_size = labels.size(0)

        bs = batch_size if batch_size < self.args.local_bs else self.args.local_bs

        return torch.zeros(bs, class_num).cuda(self.args.gpu).scatter_(1, labels.view(bs, 1), 1)

    def train(self, inputs, confused_labels, lams, flag=False): ##

        input_temp = inputs.clone().detach()
        input_temp.requires_grad_()
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(input_temp)
        label = confused_labels.pop(0)
        loss = self.ce(outputs, label) 
        for i in range(len(confused_labels)):
            loss = lams[i] * loss + (1 - lams[i]) * self.ce(outputs, confused_labels[i])
        
        if flag:
            label_vector = self._labels2onehot(label)
            for index in range(len(confused_labels)):
                label_vector = lams[index] * label_vector + (1 - lams[index]) * self._labels2onehot(confused_labels[index])
            label_vector = torch.clamp(label_vector, 0., 1.)
            ml_loss = self.ml(outputs, Variable(label_vector))
            #print(f"ml_loss: {ml_loss}")
            loss += 0.2 * ml_loss
            #print(F.log_softmax(Variable(label_vector), dim=1))
            #kl_loss = self.kl(F.log_softmax(outputs, dim=1), Variable(label_vector))
            #loss += 0.2 * kl_loss
            #print(f"kl_loss: {kl_loss}")
            
        loss.backward()
        grad = copy.deepcopy(input_temp.grad.data)
        self.optimizer.step()
        return loss, grad

    def get_weight(self,factors=None):
        return self.model.state_dict()

class Trainer(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.trainloader = []
        self.cls_num_list = []
    def _train_val_test(self, dataset, idxs):
        batch_size = args.local_bs
        trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        #trainloader = DataLoader(DatasetFragment(dataset, 80), batch_size=batch_size, shuffle=True)
        return trainloader
    def reset(self):
        self.trainloader = []
        self.cls_num_list = []
    def assign(self, idxs_list, cls_num_list=None):
        self.reset()
        self.cls_num_list = cls_num_list
        
        for idxs in idxs_list:
            train_loader = self._train_val_test(self.dataset, list(idxs))
            self.trainloader.append(train_loader)

    def train(self, extractor, classifer, epoch):
        index = 1
        server = Server(classifer, self.args)
        #TODO delete the deepcopy
        clients = [Client(copy.deepcopy(extractor), self.trainloader[i], server, self.args) for i in range(len(self.trainloader))]
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
    TAG = 'mixsl-' + data_name + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)

    logs = []
    if args.mnist:
        train_dataset, test_dataset = get_mnist()
    else:
        if args.cifar100:
            train_dataset, test_dataset  = get_cifar100()
        else:
            train_dataset, test_dataset  = get_cifar10()
    user_groups = random_avg_strategy(train_dataset, args.num_users)
    #cls_num_per_clients = count_class_num_per_client(train_dataset, user_groups, 10)
    tf_log = Logs(TAG)
    logs_file = TAG
    
    client_part, server_part = get_model(args)
    trainer = Trainer(args, train_dataset)

    for epoch in range(args.epochs):
        local_weights = []
        idxs_users = random_assign(args)
        idx_list = [user_groups[i] for i in idxs_users]
        #cls_num_list = [list(cls_num_per_clients[i]) for i in idxs_users]
        cls_num_list= None
        trainer.assign(idx_list, cls_num_list)
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
    
    #write_mat('cifar100_'+ str(args.mix_num) +'.txt', label_mat, args.mix_num)