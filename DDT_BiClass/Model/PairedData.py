import random
import torch.utils.data

class PairedData(object):
    def __init__(self, data_loader_S, data_loader_T, max_dataset_size, flip):
        self.data_loader_S = data_loader_S
        self.data_loader_T = data_loader_T
        self.stop_S = False
        self.stop_T = False
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self):
        self.stop_S = False
        self.stop_T = False
        self.data_loader_S_iter = iter(self.data_loader_S)
        self.data_loader_T_iter = iter(self.data_loader_T)
        self.iter = 0
        return self

    def __next__(self):
        S, S_paths,S_index = None, None, None
        T, T_paths,T_index = None, None, None

        try:
            S, S_paths,S_index = next(self.data_loader_S_iter)
        except StopIteration:
            if S is None or S_paths is None:
                self.stop_S = True
                self.data_loader_S_iter = iter(self.data_loader_S)
                S, S_paths,S_index = next(self.data_loader_S_iter)

        try:
            T, T_paths,T_index = next(self.data_loader_T_iter)
        except StopIteration:
            if T is None or T_paths is None:
                self.stop_T = True
                self.data_loader_T_iter = iter(self.data_loader_T)
                T, T_paths,T_index = next(self.data_loader_T_iter)

        if (self.stop_S and self.stop_T) or self.iter > self.max_dataset_size:
            self.stop_S = False
            self.stop_T = False
            raise StopIteration()
        else:
            self.iter += 1
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(S.size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                S = S.index_select(3, idx)
                T = T.index_select(3, idx)
            return {'S': S, 'S_label': S_paths,'S_index':S_index,
                    'T': T, 'T_label': T_paths,'T_index':T_index}

class CVDataLoader(object):
    def initialize(self, data_loader_S, data_loader_T, dataset_S, dataset_T):

        self.max_dataset_size = float("inf")  # infinite big
        self.dataset_S = dataset_S
        self.dataset_T = dataset_T
        flip = False
        self.paired_data = PairedData(data_loader_S, data_loader_T, self.max_dataset_size, flip)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_S), len(self.dataset_T)), self.max_dataset_size)

if __name__ == '__main__':

    '''Example'''
    # train_loader = CVDataLoader()
    # train_loader.initialize(data_loader_S, data_loader_T, dataset_S, dataset_T)
    # dataset = train_loader.load_data()
    #
    # for batch_idx, data in enumerate(dataset):
    #     data_s = data['S']
    #     label_s = data['S_label']
    #     data_t = data['T']
    #     label_t = data['T_label']
    #
    #     if cuda:
    #         data_s, label_s = data_s.cuda(), label_s.cuda()
    #         data_t, label_t = data_t.cuda(), label_t.cuda()
    #