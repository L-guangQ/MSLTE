import numpy as np
import scipy
import networkx as nx
import torch
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import pandas as pd
import os
import random


def dist_adjacency():
    row_ = np.array(
        [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
         13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 25, 26, 26,
         27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
         41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54,
         54, 55, 55, 56, 57, 58, 59,
         60, 1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12,
         20, 13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26,
         34, 27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47,
         40, 48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58,
         54, 59, 55, 60, 56, 61, 61, 58, 59, 60, 61])

    col_ = np.array(
        [1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20,
         13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26, 34,
         27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47, 40,
         48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58, 54,
         59, 55, 60, 56, 61, 61, 58,
         59, 60, 61, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
         11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25,
         25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38,
         39, 39, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
         53, 53, 54, 54, 55, 55, 56, 57, 58, 59, 60])

    weight_ = np.ones(236).astype('float32')
    A_sparse = scipy.sparse.csr_matrix((weight_, (row_, col_)), shape=(62, 62))
    A_dense = scipy.sparse.coo_matrix(A_sparse).todense()

    return row_, col_, weight_





class SEED_dependent_classify(DGLDataset):
    def __init__(self, data_path, sub: int, session: int, train: bool, loop=False):
        self.data_path = data_path
        self.sub = sub
        self.session = session
        self.train = train
        self.loop = loop
        super(SEED_dependent_classify, self).__init__(name='SEED')

    def process(self):
        src, dst, weight = dist_adjacency()
        label_dict = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

        self.graphs = []
        self.labels = []

        sub_files = os.listdir(self.data_path)
        sessions = os.listdir(os.path.join(self.data_path, sub_files[self.sub]))
        trials = os.listdir(os.path.join(self.data_path, sub_files[self.sub], sessions[self.session]))
        if self.train:
            trials = trials[:9]
            start = 0
        else:
            trials = trials[9:]
            start = 9
        for k, trial in enumerate(trials, start=start):
            data = np.load(os.path.join(self.data_path, sub_files[self.sub], sessions[self.session], trial))
            data = data['de']
            # mean = np.mean(data, axis=-1, keepdims=True)
            data = (data - np.mean(data, axis=0, keepdims=True)) / np.std(data, axis=0, keepdims=True)
            # data = (data - np.mean(data, axis=0, keepdims=True)) / np.max(data, axis=0, keepdims=True)
            # data = (data - np.mean(data, axis=0, keepdims=True)) / (np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
            nums = data.shape[1]
            for i in range(nums):
                g = dgl.graph((src, dst), num_nodes=62)
                g.ndata['x'] = torch.tensor(data[:, i, :])
                g.edata['w'] = torch.tensor(weight)
                if self.loop:
                    self_loop_src = [i for i in range(62)]
                    self_loop_dst = [i for i in range(62)]
                    self_loop_weights = torch.ones(62)
                    g.add_edges(self_loop_src, self_loop_dst, {'w': self_loop_weights})
                label = label_dict[k]

                self.graphs.append(g)
                self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return dgl.to_float(self.graphs[i]), self.labels[i] + 1

    def __len__(self):
        return len(self.graphs)


class SEED_independent_classify(DGLDataset):
    def __init__(self, data_path, sub: int, session: int, train: bool, loop=False):
        self.data_path = data_path
        self.sub = sub
        self.session = session
        self.train = train
        self.loop = loop
        super(SEED_independent_classify, self).__init__(name='SEED')

    def process(self):
        src, dst, weight = dist_adjacency()
        label_dict = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

        self.graphs = []
        self.labels = []

        sub_files = os.listdir(self.data_path)
        if self.train:
            del sub_files[self.sub]
        else:
            sub_files = [sub_files[self.sub]]

        for sub in sub_files:
            sessions = os.listdir(os.path.join(self.data_path, sub))
            trials = os.listdir(os.path.join(self.data_path, sub, sessions[self.session]))
            for k, trial in enumerate(trials):
                data = np.load(os.path.join(self.data_path, sub, sessions[self.session], trial))
                data = data['de']
                # mean = np.mean(data, axis=-1, keepdims=True)
                # data = (data - np.mean(data, axis=0, keepdims=True)) / np.max(data, axis=0, keepdims=True)
                data = (data - np.mean(data, axis=0, keepdims=True)) / np.std(data, axis=0, keepdims=True)
                # data = (data - np.mean(data, axis=0, keepdims=True)) / (np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
                nums = data.shape[1]
                for i in range(nums):
                    g = dgl.graph((src, dst), num_nodes=62)
                    g.ndata['x'] = torch.tensor(data[:, i, :])
                    g.edata['w'] = torch.tensor(weight)
                    if self.loop:
                        self_loop_src = [i for i in range(62)]
                        self_loop_dst = [i for i in range(62)]
                        self_loop_weights = torch.ones(62)
                        g.add_edges(self_loop_src, self_loop_dst, {'w': self_loop_weights})
                    label = label_dict[k]

                    self.graphs.append(g)
                    self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)


    def __getitem__(self, i):
        return dgl.to_float(self.graphs[i]), self.labels[i] + 1

    def __len__(self):
        return len(self.graphs)




def deap_dist_adjacency():
    row_ = np.array(
        [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7,
         8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 14, 14, 14, 14,
         15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20,
         21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25,
         26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31])

    col_ = np.array(
        [1, 16, 0, 2, 3, 18, 1, 3, 4, 5, 18, 1, 2, 4, 2, 3, 5, 6, 7, 2, 4, 6, 18, 22, 23, 4, 5, 7, 8, 9, 23, 4, 6, 8,
         6, 7, 9, 10, 11, 6, 8, 10, 15, 23, 27, 8, 9, 11, 12, 15, 8, 10, 12, 10, 11, 13, 14, 15, 30, 12, 14, 12, 13, 30, 31,
         9, 10, 12, 27, 28, 30, 0, 17, 16, 18, 19, 20, 1, 2, 5, 17, 19, 22, 17, 18, 20, 21, 22, 17, 19, 21,
         19, 20, 22, 24, 25, 5, 18, 19, 21, 23, 24, 5, 6, 9, 22, 24, 27, 21, 24, 23, 25, 26, 27, 21, 24, 26,
         24, 25, 27, 28, 29, 9, 15, 23, 24, 26, 28, 15, 26, 27, 29, 30, 26, 28, 30, 12, 14, 15, 28, 29, 31, 14, 30])

    weight_ = np.ones(144).astype('float32')
    A_sparse = scipy.sparse.csr_matrix((weight_, (row_, col_)), shape=(32, 32))
    A_dense = scipy.sparse.coo_matrix(A_sparse).todense()

    return row_, col_, weight_



class DEAP_classify(DGLDataset):
    def __init__(self, data_path, sub: int, k_fold: int, train: bool, label_flag: str, loop=False):
        self.data_path = data_path
        self.sub = sub
        self.k_fold = k_fold
        self.train = train
        self.label_flag = label_flag
        self.loop = loop
        super(DEAP_classify, self).__init__(name='DEAP')

    def process(self):
        src, dst, weight = deap_dist_adjacency()

        self.graphs = []
        self.labels = []

        sub_files = os.listdir(self.data_path)
        trials = os.path.join(self.data_path, sub_files[self.sub])
        raw_data = np.load(trials)
        if self.label_flag == 'valence':
            labels = raw_data['label'][:, 0]  # 40, 1
        elif self.label_flag == 'arousal':
            labels = raw_data['label'][:, 1]  # 40, 1

        raw_de = raw_data['DE']     # 40, 60, 32, 4
        # raw_de = np.reshape(raw_de, (40*60, 32, 4))
        raw_de = (raw_de - np.mean(raw_de, axis=2, keepdims=True)) / np.max(raw_de, axis=2, keepdims=True)
        # raw_de = (raw_de - np.mean(raw_de, axis=1, keepdims=True)) / np.std(raw_de, axis=1, keepdims=True)
        # raw_de = np.reshape(raw_de, (40, 60, 32, 4))

        index_list = list(range(40))
        if self.train:
            del index_list[4*self.k_fold : 4*self.k_fold+4]
        else:
            index_list = index_list[4*self.k_fold : 4*self.k_fold+4]

        for trial in index_list:
            data = raw_de[trial]    # 60, 32, 4
            # data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            nums = data.shape[0]
            for i in range(nums):
                g = dgl.graph((src, dst), num_nodes=32)
                g.ndata['x'] = torch.tensor(data[i, :, :])
                g.edata['w'] = torch.tensor(weight)
                if self.loop:
                    self_loop_src = [i for i in range(32)]
                    self_loop_dst = [i for i in range(32)]
                    self_loop_weights = torch.ones(32)
                    g.add_edges(self_loop_src, self_loop_dst, {'w': self_loop_weights})
                label = labels[trial]
                if label > 5.0:
                    label = 1
                else:
                    label = 0

                self.graphs.append(g)
                self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return dgl.to_float(self.graphs[i]), self.labels[i]

    def __len__(self):
        return len(self.graphs)


class DEAP_independent(DGLDataset):
    def __init__(self, data_path, sub: int, train: bool, label_flag: str, loop=False):
        self.data_path = data_path
        self.sub = sub
        self.train = train
        self.label_flag = label_flag
        self.loop = loop
        super(DEAP_independent, self).__init__(name='DEAP')

    def process(self):
        src, dst, weight = deap_dist_adjacency()

        self.graphs = []
        self.labels = []

        sub_files = os.listdir(self.data_path)
        if self.train:
            del sub_files[self.sub]
        else:
            sub_files = [sub_files[self.sub]]

        for sub in sub_files:
            trials = os.path.join(self.data_path, sub)
            raw_data = np.load(trials)
            if self.label_flag == 'valence':
                labels = raw_data['label'][:, 0]  # 40, 1
            elif self.label_flag == 'arousal':
                labels = raw_data['label'][:, 1]  # 40, 1


            raw_de = raw_data['DE']     # 40, 60, 32, 4
            raw_de = np.reshape(raw_de, (40*60, 32, 4))
            raw_de = (raw_de - np.mean(raw_de, axis=0, keepdims=True)) / np.max(raw_de, axis=0, keepdims=True)
            raw_de = np.reshape(raw_de, (40, 60, 32, 4))


            index_list = list(range(40))
            for trial in index_list:
                data = raw_de[trial]    # 60, 32, 4
                # data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
                nums = data.shape[0]
                for i in range(nums):
                    g = dgl.graph((src, dst), num_nodes=32)
                    g.ndata['x'] = torch.tensor(data[i, :, :])
                    g.edata['w'] = torch.tensor(weight)
                    if self.loop:
                        self_loop_src = [i for i in range(32)]
                        self_loop_dst = [i for i in range(32)]
                        self_loop_weights = torch.ones(32)
                        g.add_edges(self_loop_src, self_loop_dst, {'w': self_loop_weights})
                    label = labels[trial]
                    if label > 5.0:
                        label = 1
                    else:
                        label = 0

                    self.graphs.append(g)
                    self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return dgl.to_float(self.graphs[i]), self.labels[i]

    def __len__(self):
        return len(self.graphs)


class DEAP_independent2(DGLDataset):
    def __init__(self, data_path, sub: int, train: bool, label_flag: str, loop=False, mean=None, max=None):
        self.data_path = data_path
        self.sub = sub
        self.train = train
        self.label_flag = label_flag
        self.loop = loop
        self.mean = mean
        self.max = max
        super(DEAP_independent2, self).__init__(name='DEAP')

    def process(self):
        src, dst, weight = deap_dist_adjacency()

        self.graphs = []
        self.labels = []

        sub_files = os.listdir(self.data_path)

        if self.train:
            del sub_files[self.sub]
        else:
            sub_files = [sub_files[self.sub]]

        raw_de = []
        label_list = []
        for sub in sub_files:
            trials = os.path.join(self.data_path, sub)
            raw_data = np.load(trials)
            if self.label_flag == 'valence':
                labels = raw_data['label'][:, 0]  # 40, 1
            elif self.label_flag == 'arousal':
                labels = raw_data['label'][:, 1]  # 40, 1

            raw_de.append(raw_data['DE'])
            label_list.append(labels)
            # raw_de = raw_data['DE']     # 40, 60, 32, 4
            # raw_de = np.reshape(raw_de, (40*60, 32, 4))
            # raw_de = (raw_de - np.mean(raw_de, axis=0, keepdims=True)) / np.max(raw_de, axis=0, keepdims=True)
            # raw_de = np.reshape(raw_de, (40, 60, 32, 4))
        raw_de = np.concatenate(raw_de)
        if self.train:
            raw_de = np.reshape(raw_de, (1240 * 60, 32, 4))
            self.mean = np.mean(raw_de, axis=0, keepdims=True)
            self.max = np.max(raw_de, axis=0, keepdims=True)
            raw_de = (raw_de - np.mean(raw_de, axis=0, keepdims=True)) / np.max(raw_de, axis=0, keepdims=True)
            raw_de = np.reshape(raw_de, (1240, 60, 32, 4))
        else:
            raw_de = np.reshape(raw_de, (40 * 60, 32, 4))
            raw_de = (raw_de - self.mean) / self.max
            raw_de = np.reshape(raw_de, (40, 60, 32, 4))
        label_list = np.concatenate(label_list)

        index_list = list(range(40 * len(sub_files)) )
        for trial in index_list:
            data = raw_de[trial]    # 60, 32, 4
            # data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
            nums = data.shape[0]
            for i in range(nums):
                g = dgl.graph((src, dst), num_nodes=32)
                g.ndata['x'] = torch.tensor(data[i, :, :])
                g.edata['w'] = torch.tensor(weight)
                if self.loop:
                    self_loop_src = [i for i in range(32)]
                    self_loop_dst = [i for i in range(32)]
                    self_loop_weights = torch.ones(32)
                    g.add_edges(self_loop_src, self_loop_dst, {'w': self_loop_weights})
                label = label_list[trial]
                if label > 5.0:
                    label = 1
                else:
                    label = 0

                self.graphs.append(g)
                self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return dgl.to_float(self.graphs[i]), self.labels[i]

    def __len__(self):
        return len(self.graphs)


if __name__ == "__main__":
    train = DEAP_independent(data_path=r'D:\EEGdata\deap\DE_PSD', sub=0, train=True, label_flag='valence')
    test = DEAP_independent(data_path=r'D:\EEGdata\deap\DE_PSD', sub=0, train=False, label_flag='valence')

    print()