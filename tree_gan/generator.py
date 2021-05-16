import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from batch_loader import Batch


class Generator(nn.Module):
    def __init__(self, actions_len, rules_len,
                 actions_embedding_dim, rules_embedding_dim, actions, rules,
                 hidden_dim, max_seq_len, matrix, gpu=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.actions_len = actions_len
        self.rules_len = rules_len
        self.actions_embedding_dim = actions_embedding_dim
        self.rules_embedding_dim = rules_embedding_dim
        self.max_seq_len = max_seq_len
        self.actions = actions
        self.rules = rules
        self.gpu = gpu
        self.M = matrix

        self.actions_embeddings = nn.Embedding(actions_len, actions_embedding_dim)
        self.rules_embeddings = nn.Embedding(rules_len, rules_embedding_dim)
        self.rnn = nn.GRU(actions_embedding_dim + rules_embedding_dim, hidden_dim)
        self.rnn2out = nn.Linear(hidden_dim, actions_len + rules_len)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=True)
        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, state: Batch, hidden: Variable):
        prev_emb = self.actions_embeddings(state.prev.to(torch.int64))
        parent_emb = self.rules_embeddings(state.parent.to(torch.int64))
        emb = torch.cat([prev_emb, parent_emb], dim=1)
        emb = emb.view(1, -1, self.actions_embedding_dim + self.rules_embedding_dim)
        out, hidden = self.rnn(emb, hidden)
        out = self.rnn2out(out.view(-1, self.hidden_dim))
        out = F.log_softmax(out, dim=1).detach().numpy()[:, :self.actions_len]
        out = Variable(torch.FloatTensor(out), requires_grad=True)
        return out, hidden

    def sample(self, max_seq_len, num_samples):
        h = self.init_hidden(num_samples)

        samples = np.zeros(num_samples, max_seq_len)
        stacks = [[(0, 1)] for _ in range(num_samples)]

        for t in range(self.max_seq_len):
            for i in range(num_samples):
                if len(stacks[i]) == 0:
                    continue
                parent, rule = stacks[i].pop()
                prev = 0
                if t > 0:
                    prev = samples[i, t - 1]
                batch = Batch(self.gpu, parent=parent, rule=rule, prev=prev)
                batch.convert_to_tensors()
                out, h = self.forward(batch, h)
                out = torch.multinomial(torch.exp(out), 1)
                pred_action = out.view(-1).data
                children = self.actions[int(pred_action[0])][::-1]
                stacks[i].extend(list(zip([rule] * len(children), children)))
                samples[i, t] = int(pred_action[0])
        return samples

    def batchNLLLoss(self, batch: Batch):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = batch.batch_size, batch.seq_len
        h = self.init_hidden(batch_size)
        batch.permute()
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(batch[i], h)
            loss += loss_fn(out, batch.target[i].to(torch.int64))
        return loss

    # def batchPGLoss(self, batch: Batch, reward):
    #     batch_size, seq_len = batch.batch_size, batch.seq_len
    #     batch.permute()
    #     h = self.init_hidden(batch_size)
    #
    #     loss = 0
    #     for i in range(seq_len):
    #         out, h = self.forward(inp[i], h)
    #         for j in range(batch_size):
    #             loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q
    #
    #     return loss / batch_size
