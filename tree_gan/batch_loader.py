import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable


class Batch:
    def __init__(self, cuda, **kwargs):
        self.cuda = cuda
        self.requires_grad = kwargs.get('requires_grad', True)
        self.batch_size = kwargs.get('batch_size', 1)
        self.seq_len = kwargs.get('seq_len', 1)
        self.rule = kwargs.get('rule', Variable(torch.zeros(self.batch_size, self.seq_len)))
        self.prev = kwargs.get('prev', Variable(torch.zeros(self.batch_size, self.seq_len)))
        self.parent = kwargs.get('parent', Variable(torch.zeros(self.batch_size, self.seq_len)))
        self.target = kwargs.get('target', Variable(torch.zeros(self.batch_size, self.seq_len)))
        if not isinstance(self.rule, Variable):
            self.rule = Variable(torch.FloatTensor([self.rule]))
        if not isinstance(self.prev, Variable):
            self.prev = Variable(torch.FloatTensor([self.prev]))
        if not isinstance(self.parent, Variable):
            self.parent = Variable(torch.FloatTensor([self.parent]))
        if cuda:
            self.__to_cuda__()

    def permute(self):
        self.prev = self.prev.permute(1, 0)
        self.parent = self.parent.permute(1, 0)
        self.rule = self.rule.permute(1, 0)
        self.target = self.target.permute(1, 0)

    def __getitem__(self, item):
        return Batch(self.cuda, batch_size=self.batch_size, seq_len=1, rule=self.rule[item],
                     prev=self.prev[item], parent=self.parent[item])

    def __to_cuda__(self):
        self.prev = self.prev.cuda()
        self.parent = self.parent.cuda()
        self.rule = self.rule.cuda()
        self.target = self.target.cuda()

    def switch_grad(self):
        self.requires_grad = not self.requires_grad
        self.parent.requires_grad = self.requires_grad
        self.prev.requires_grad = self.requires_grad
        self.rule.requires_grad = self.requires_grad
        self.target.requires_grad = self.requires_grad


class BatchLoader:
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df

    def load_action_batch(self, seq_len, batch_size, cuda) -> Batch:
        cols = {col: i for i, col in enumerate(self.data_df.columns)}
        batch_index = 0
        batch = None
        for file_id, group in self.data_df.groupby('file_id'):
            if batch_index == 0:
                batch = Batch(cuda, seq_len=seq_len, batch_size=batch_size, requires_grad=False)
            arr = np.pad(group.values, ((0, max(0, seq_len - group.values.shape[0])), (0, 0)))[:seq_len]
            batch.target[batch_index] = Variable(torch.FloatTensor(arr[:, cols['action_id']]))
            batch.rule[batch_index, 0] = 0
            batch.rule[batch_index, 1:] = Variable(torch.FloatTensor(arr[:-1, cols['rule_id']]))
            for i in range(min(seq_len, group.values.shape[0])):
                batch.prev[batch_index, i] = arr[arr[i, cols['prev_id']], cols['action_id']]
                batch.parent[batch_index, i] = arr[arr[i, cols['parent_id']], cols['rule_id']]
            batch_index += 1
            if batch_index == batch_size:
                batch_index = 0
                batch.switch_grad()
                yield batch
        if batch_index > 0:
            batch.switch_grad()
            yield batch
