import torch
from torch.autograd import Variable


class Batch:
    def __init__(self, seq_len, batch_size):
        self.rule = torch.zeros(batch_size, seq_len)
        self.prev = torch.zeros(batch_size, seq_len)
        self.parent = torch.zeros(batch_size, seq_len)
        self.target = torch.zeros(batch_size, seq_len)

    def transform(self, cuda):
        self.prev = Variable(self.prev).type(torch.LongTensor)
        self.rule = Variable(self.rule).type(torch.LongTensor)
        self.parent = Variable(self.parent).type(torch.LongTensor)
        self.target = Variable(self.target).type(torch.LongTensor)
        if cuda:
            self.prev = self.prev.cuda()
            self.parent = self.parent.cuda()
            self.rule = self.rule.cuda()
            self.target = self.target.cuda()


class BatchLoader:
    def __init__(self, data_df):
        self.data_df = data_df

    def load_action_batch(self, seq_len, batch_size, cuda) -> Batch:
        data_np = self.data_df.values
        n = seq_len * batch_size
        for i in range(0, data_np.shape[0] - n - 1, n):
            new_batch = Batch(seq_len, batch_size)
            for j in range(batch_size):
                for q in range(seq_len):
                    pos = j * batch_size + q
                    new_batch.rule[j, q] = data_np[pos, 2]
                    if data_np[pos, 4] == -1:
                        new_batch.prev[j, q] = data_np[pos, 5]
                    else:
                        new_batch.prev[j, q] = data_np[data_np[pos, 4], 5]
                    if data_np[pos, 3] == -1:
                        new_batch.parent[j, q] = data_np[pos, 2]
                    else:
                        new_batch.parent[j, q] = data_np[data_np[pos, 3], 2]
                    new_batch.target[j, q] = data_np[pos + 1, 5]
            new_batch.transform(cuda)
            yield new_batch
