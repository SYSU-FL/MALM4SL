import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from scipy.spatial.distance import pdist, squareform

import pandas as pd
from datetime import datetime
import time
import random
from tensorboardX import SummaryWriter
import os
import sys
from models.cnn import *
from models.vgg import *
from models.mlp import *
from models.resnet import *

class Logs(object):
    def __init__(self, name):
        root = r"../baseline_log"
        if not os.path.exists(root):
            os.mkdir(root)
        self.writer = SummaryWriter(os.path.join(root, name))
    def close(self):
        self.writer.close()


def format_args(args):
    return "frac{}-bs{}-users{}-epochs{}-k(onlyformixup){}-pai(onlyformixup){}-m{}-lr{}".format(args.frac, args.local_bs, args.num_users, args.epochs, args.k, args.pai, args.momentum, args.lr)

def save_logs(logs, tag, args):
    df = pd.DataFrame(logs)
    param_str = format_args(args)
    path = '../logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')
    df['test_acc'] = df['test_acc'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy: {df.loc[:,'test_acc'].max()}")
    print("save logs sucess!")

def save_cifar100_logs(logs, tag, args):
    df = pd.DataFrame(logs)
    param_str = format_args(args)
    path = '../logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')

    df['test_acc1'] = df['test_acc1'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy1: {df.loc[:,'test_acc1'].max()}")
    df['test_acc5'] = df['test_acc5'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy5: {df.loc[:,'test_acc5'].max()}")
    print("save logs sucess!")

def setup_seed(seed, gpu_enabled):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if gpu_enabled:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True





def random_assign(args):
    m = max(int(args.num_users * args.frac), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    return idxs_users



def differential_privacy(data, cuda=0):
    noise = torch.FloatTensor(data.shape).normal_(0, 1.1)
    if cuda > -1:
        noise = noise.to(cuda)
    data.add_(noise)
    return data

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

    
def compute_dcor(x, y, args):

    def _distance_covariance(a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.size(0)

    def _distance_variance(a_matrix):
        return (a_matrix ** 2).sum().sqrt() / a_matrix.size(0)

    def _A_matrix(data):
        distance_matrix = _distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(data):
        n = data.size(0)
        distance_matrix = torch.zeros((n, n)).cuda(args.gpu)

        for i in range(n):
            for j in range(n):
                row_diff = data[i] - data[j]
                distance_matrix[i, j] = (row_diff ** 2).sum()

        return distance_matrix

    input_data = x.clone().detach()
    intermediate_data = y.clone().detach()
    input_data = input_data.view(input_data.size(0), -1)
    intermediate_data = intermediate_data.view(intermediate_data.size(0), -1)

    # Get A matrices of data
    A_input = _A_matrix(input_data)
    A_intermediate = _A_matrix(intermediate_data)

    # Get distance variances
    input_dvar = _distance_variance(A_input)
    intermediate_dvar = _distance_variance(A_intermediate)

    # Get distance covariance
    dcov = _distance_covariance(A_input, A_intermediate)

    # Put it together
    dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

    return dcorr



def compute_similarity(x, y):
    x_ = x.clone().detach().view(x.size(0), -1)
    y_ = y.clone().detach().view(y.size(0), -1)
    sim = torch.cosine_similarity(x_, y_, dim=1)
    mean_sim = sim.mean()
    return mean_sim

def get_model(args):
    client_part, server_part = None, None
    if args.model == 'cnnmnist':
        client_part = MCNN_Extractor()
        server_part = MCNN_Classifier()
    elif args.model == 'mlp':
        client_part = MLP_Extractor()
        server_part = MLP_Classifier()
    elif args.model == 'cnncifar':
        print("init cnncifar........")
        client_part = CCNN_Extractor()
        server_part = CCNN_Classifier(10 if not args.cifar100 else 100)
    elif args.model == 'res18':
        client_part = ResNet18_Extractor()
        server_part = ResNet18_Classifer(10 if not args.cifar100 else 100)
    elif args.model == 'res34':
        client_part = ResNet34_Extractor()
        server_part = ResNet34_Classifer(10 if not args.cifar100 else 100)
    elif args.model == 'vgg16':
        client_part, server_part = get_split_vgg16(10 if not args.cifar100 else 100)
 
    if args.resume:
        print("===> resuming from checkpoint...")
        if not os.path.exists('../checkpoints'):
            os.mkdir('../checkpoints')
        if os.path.exists('../checkpoints/' + args.reload_path + '_ckpt'):
            ckpt = torch.load('../checkpoints/' + args.reload_path + '_ckpt')
            client_part.load_state_dict(ckpt['client'])
            server_part.load_state_dict(ckpt['server'])

    if args.gpu > -1:
        client_part.cuda(args.gpu)
        server_part.cuda(args.gpu)

    return client_part, server_part


def save_checkpoints(client_part, server_part, args):
    if not os.path.exists('../checkpoints'):
        os.mkdir('../checkpoints')

    state = {
        'client': client_part.cpu().state_dict(),
        'server': server_part.cpu().state_dict()
    }
    torch.save(state,  '../checkpoints/' + args.reload_path + '_ckpt')
    print("==> saving checkpoint....")



# def write_mat(name, label_list, num=10):
#     vectors = []
#     for l_l in label_list:
#         l_l = l_l.cpu().numpy()
#         for index in range(l_l.shape[0]):
#             vector = [str(l_l[index][0])]
#             for j_index in range(num):
#                 vector.append(str(l_l[index][j_index]))
#             vectors.append(' '.join(vector))
#     with open(name, 'w+') as f:
#         for v in vectors:
#             f.write(v + '\n')
#     print("save sucess !")

def write_mat(name, label_list, num):
    vectors = []
    #print(len(label_list))
    #print(len(label_list[0]))
    for label_batch_list in label_list:
        size =label_batch_list[0].size(0)
        batch_list = [l.view(size, 1) for l in label_batch_list]
        label_batch = torch.cat(batch_list, dim=1)
        label_batch_mat = label_batch.cpu().numpy()
        #print(label_batch_mat.shape)
        for index in range(label_batch_mat.shape[0]):
            vector = [str(label_batch_mat[index][0])]
            for j_index in range(num):
                vector.append(str(label_batch_mat[index][j_index]))
            vectors.append(' '.join(vector))

    with open(name, 'w+') as f:
        for v in vectors:
            f.write(v + '\n')
    print("save sucess !")




    


    