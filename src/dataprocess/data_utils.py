import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
import numpy as np
from  .imbalance_cifar import IMBALANCECIFAR10, RemainData, IMBALANCECIFAR100

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(int(label))#  根据imbalanced-cifar10修改了这里


class DatasetFragment(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, num_classes=50):
        self.dataset = dataset
        self.idxs = []
        classes = [i for i in range(num_classes)]
        for idx in range(len(self.dataset)):
            image, label = self.dataset[idx]
            if int(label) in classes:
                self.idxs.append(idx)
        print(f"all size: {len(self.dataset)}, dataset fragment size: {len(self.idxs)}")
        
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(int(label))#  根据imbalanced-cifar10修改了这里



class DatasetSlice(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, num_classes=[0,1]):
        self.dataset = dataset
        self.idxs = []
        classes = num_classes
        for idx in range(len(self.dataset)):
            image, label = self.dataset[idx]
            if int(label) in classes:
                self.idxs.append(idx)
        print(f"all size: {len(self.dataset)}, dataset slice size: {len(self.idxs)}")
        
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(int(label))#  根据imbalanced-cifar10修改了这里




def get_cifar10(balanced=False, remain_flag=False):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if balanced:
        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset
    else:
        trainset = IMBALANCECIFAR10(root='../data', train=True, download=True, transform=transform_train)
        #testset = IMBALANCECIFAR10(root='../data', train=False, download=True, transform=transform_test)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        if remain_flag:
            remain_data, remain_labels = trainset.get_remain_data()
            remain_dataset = RemainData(root='../data', remain_data=remain_data, remain_labels=remain_labels, train=True, download=True, transform=transform_train)
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}, remain dat size: {len(remain_dataset)}")
            return trainset, testset, remain_dataset
        else:
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
            return trainset, testset



def get_cifar100(balanced=False, remain_flag=False):
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])

    if balanced:
        trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset
    else:
        trainset = IMBALANCECIFAR100(root='../data', train=True, download=True, transform=transform_train)
        #testset = IMBALANCECIFAR10(root='../data', train=False, download=True, transform=transform_test)
        testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        if remain_flag:
            remain_data, remain_labels = trainset.get_remain_data()
            remain_dataset = RemainData(root='../data', remain_data=remain_data, remain_labels=remain_labels, train=True, download=True, transform=transform_train)
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}, remain dat size: {len(remain_dataset)}")
            return trainset, testset, remain_dataset
        else:
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
            return trainset, testset




def cifar_noniid(dataset, num_users, imbalanced=False):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    """
    if imbalanced is False:
        num_shards, num_imgs = 200, 250
    else:
        num_shards, num_shards = 1000, 50 # 500, 100
    """
    num_shards = 200 if imbalanced is False else 5000
    num_imgs = 250 if imbalanced is False else 10
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    if not hasattr(dataset, 'targets'):
        labels = np.array(dataset.train_labels)
    else:
        labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    if imbalanced is False:
        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    else:
         # Minimum and maximum shards assigned per client:
        min_shard = 13#1
        max_shard = 150#10

        # Divide the shards into random chunks for every client
        # s.t the sum of these chunks = num_shards
        random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
        random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
        random_shard_size = random_shard_size.astype(int)

        # Assign the shards randomly to each client
        
        if sum(random_shard_size) > num_shards:

            for i in range(num_users):
                # First assign each client 1 shard to ensure every client has
                # atleast one shard of data
                rand_set = set(np.random.choice(idx_shard, 1, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            random_shard_size = random_shard_size-1

            # Next, randomly assign the remaining shards
            for i in range(num_users):
                if len(idx_shard) == 0:
                    continue
                shard_size = random_shard_size[i]
                if shard_size > len(idx_shard):
                    shard_size = len(idx_shard)
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)
        else:

            for i in range(num_users):
                shard_size = random_shard_size[i]
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            if len(idx_shard) > 0:
                # Add the leftover shards to the client with minimum images:
                shard_size = len(idx_shard)
                # Add the remaining shard to the client with lowest data
                k = min(dict_users, key=lambda x: len(dict_users.get(x)))
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[k] = np.concatenate(
                        (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)
    
    ps = {k: len(v) for k, v in dict_users.items()}

    return dict_users, ps






def cifar_label_imbalance(dataset, num_users, imbalanced=False):
    
    num_shards = 200 if imbalanced is False else 5000
    num_imgs = 250 if imbalanced is False else 10
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    if not hasattr(dataset, 'targets'):
        labels = np.array(dataset.train_labels)
    else:
        labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    if imbalanced is False:
        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    else:
         # Minimum and maximum shards assigned per client:
        min_shard = 13#1
        max_shard = 150#10

        # Divide the shards into random chunks for every client
        # s.t the sum of these chunks = num_shards
        random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
        random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
        random_shard_size = random_shard_size.astype(int)

        # Assign the shards randomly to each client
        
        if sum(random_shard_size) > num_shards:

            for i in range(num_users):
                # First assign each client 1 shard to ensure every client has
                # atleast one shard of data
                rand_set = set(np.random.choice(idx_shard, 1, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            random_shard_size = random_shard_size-1

            # Next, randomly assign the remaining shards
            for i in range(num_users):
                if len(idx_shard) == 0:
                    continue
                shard_size = random_shard_size[i]
                if shard_size > len(idx_shard):
                    shard_size = len(idx_shard)
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)
        else:

            for i in range(num_users):
                shard_size = random_shard_size[i]
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)

            if len(idx_shard) > 0:
                # Add the leftover shards to the client with minimum images:
                shard_size = len(idx_shard)
                # Add the remaining shard to the client with lowest data
                k = min(dict_users, key=lambda x: len(dict_users.get(x)))
                rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[k] = np.concatenate(
                        (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                        axis=0)
    
    ps = {k: len(v) for k, v in dict_users.items()}

    return dict_users, ps























def random_avg_strategy(trainset, num=100):
    num_items = int(len(trainset)/num)
    dict_users, all_idxs = {}, [i for i in range(len(trainset))]
    for i in range(num):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users




def get_mnist():
    # for cnn
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True,
                                      transform=apply_transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True,
                                      transform=apply_transform)
    return train_dataset, test_dataset

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def fix_count_class_num_per_client(groups):
    for k, v in groups.items():
        groups[k] = v.tolist()
    return groups


def count_class_num_per_client(train_set, groups, num_classes=10):
    g2c = {}
    for i, elems in groups.items():
        count = {i: 0 for i in range(0,num_classes)}
        for e in elems:
            img, label = train_set[e]
            count[label] += 1
        vals = count.values()
        g2c[i] = vals
    return g2c

def count_class_num(train_set, num_classes=10):
    count = {i: 0 for i in range(0, num_classes)}
    size = len(train_set)
    for i in range(size):
        _, label = train_set[i]
        count[label] += 1
    return count


