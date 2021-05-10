import torch
import pandas as pd


class Batch:
    def __init__(self, cuda, **kwargs):
        self.cuda = cuda
        self.batch_size = kwargs.get('batch_size', 1)
        self.seq_len = kwargs.get('seq_len', 1)
        self.rule = kwargs.get('rule', torch.zeros(self.batch_size, self.seq_len, requires_grad=True))
        self.prev = kwargs.get('prev', torch.zeros(self.batch_size, self.seq_len, requires_grad=True))
        self.parent = kwargs.get('parent', torch.zeros(self.batch_size, self.seq_len, requires_grad=True))
        self.target = kwargs.get('target', torch.zeros(self.batch_size, self.seq_len, requires_grad=True))
        if not isinstance(self.rule, torch.Tensor):
            self.rule = torch.tensor([self.rule], requires_grad=True)
        if not isinstance(self.prev, torch.Tensor):
            self.prev = torch.tensor([self.prev], requires_grad=True)
        if not isinstance(self.parent, torch.Tensor):
            self.parent = torch.tensor([self.parent], requires_grad=True)
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


class BatchLoader:
    def __init__(self, data_df: pd.DataFrame):
        self.data_df = data_df

    def load_action_batch(self, seq_len, batch_size, cuda) -> Batch:
        cols = {col: i for i, col in enumerate(self.data_df.columns)}
        batch_index = 0
        batch = None
        for file_id, group in self.data_df.groupby('file_id'):
            if batch_index == 0:
                batch = Batch(cuda, seq_len=seq_len, batch_size=batch_size)
            arr = group.values
            batch.target[batch_index] = arr[:, cols['action_id']]
            batch.rule[batch_index, 0] = 0
            batch.rule[batch_index, 1:] = batch.target[:-1]
            for i in range(min(seq_len, arr.shape[0])):
                batch.prev[batch_index, i] = arr[arr[i, cols['prev_id']], cols['action_id']]
                batch.parent[batch_index, i] = arr[arr[i, cols['parent_id']], cols['rule_id']]
            batch_index += 1
            if batch_index == batch_size:
                batch_index = 0
                yield batch
        if batch_index > 0:
            yield batch
