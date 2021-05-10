import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from batch_loader import Batch


class Generator(nn.Module):
    def __init__(self, actions_embedding_dim, rules_embedding_dim, actions, rules,
                 hidden_dim, max_seq_len, matrix, gpu=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.actions_embedding_dim = actions_embedding_dim
        self.rules_embedding_dim = rules_embedding_dim
        self.max_seq_len = max_seq_len
        self.actions = actions
        self.rules = rules
        self.gpu = gpu
        self.M = matrix

        self.actions_embeddings = nn.Embedding(len(actions), actions_embedding_dim)
        self.rules_embeddings = nn.Embedding(len(rules), rules_embedding_dim)
        self.rnn = nn.GRU(actions_embedding_dim + rules_embedding_dim, hidden_dim)
        self.rnn2out = nn.Linear(hidden_dim, len(actions) + len(rules))

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, state: Batch, hidden: Variable):
        prev_emb = self.actions_embeddings(state.prev)
        parent_emb = self.rules_embeddings(state.parent)
        emb = torch.cat([prev_emb, parent_emb], dim=1)
        emb = emb.view(1, -1, self.actions_embedding_dim + self.rules_embedding_dim)
        out, hidden = self.rnn(emb, hidden)
        out = self.rnn2out(out.view(-1, self.hidden_dim))
        out = self.M[state.rule, :] * F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, max_seq_len, num_samples):
        h = self.init_hidden(num_samples)

        samples = torch.zeros(num_samples, max_seq_len)
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
                out, h = self.forward(batch, h)
                out = torch.multinomial(torch.exp(out), 1)
                pred_action = out.view(-1).data
                children = self.actions[pred_action][::-1]
                stacks[i].extend(list(zip([rule] * len(children), children)))
                samples[i, t] = pred_action
        return samples

    def batchNLLLoss(self, batch: Batch):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = batch.batch_size, batch.seq_len
        h = self.init_hidden(batch_size)
        batch.permute()
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(batch[i], h)
            loss += loss_fn(out, batch.target[i])
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
