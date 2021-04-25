import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from batch_loader import Batch


class Generator(nn.Module):
    def __init__(self, actions_embedding_dim, rules_embedding_dim,
                 actions_vocab_size, rules_vocab_size,
                 hidden_dim, max_seq_len, matrix, gpu=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.actions_embedding_dim = actions_embedding_dim
        self.rules_embedding_dim = rules_embedding_dim
        self.max_seq_len = max_seq_len
        self.actions_vocab_size = actions_vocab_size
        self.rules_vocab_size = rules_vocab_size
        self.gpu = gpu
        self.M = matrix

        self.actions_embeddings = nn.Embedding(actions_vocab_size, actions_embedding_dim)
        self.rules_embeddings = nn.Embedding(rules_vocab_size, rules_embedding_dim)
        self.rnn = nn.GRU(actions_embedding_dim + rules_embedding_dim, hidden_dim)
        self.rnn2out = nn.Linear(hidden_dim, actions_vocab_size + rules_vocab_size)

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
        F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_rule_index=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)
        h = self.init_hidden(num_samples)

        x = autograd.Variable(torch.LongTensor([start_rule] * num_samples))
        stacks = [[(-1, 0)] for _ in range(num_samples)]

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples

    def batchNLLLoss(self, prev: Variable, parent: Variable, target: Variable):
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = prev.size()
        prev = prev.permute(1, 0)
        parent = parent.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(prev[i], parent[i], h)
            loss += loss_fn(out, target[i])
        return loss

    # def batchPGLoss(self, inp, target, reward):
    #     """
    #     Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
    #     Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
    #
    #     Inputs: inp, target
    #         - inp: batch_size x seq_len
    #         - target: batch_size x seq_len
    #         - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
    #                   sentence)
    #
    #         inp should be target with <s> (start letter) prepended
    #     """
    #
    #     batch_size, seq_len = inp.size()
    #     inp = inp.permute(1, 0)          # seq_len x batch_size
    #     target = target.permute(1, 0)    # seq_len x batch_size
    #     h = self.init_hidden(batch_size)
    #
    #     loss = 0
    #     for i in range(seq_len):
    #         out, h = self.forward(inp[i], h)
    #         # TODO: should h be detached from graph (.detach())?
    #         for j in range(batch_size):
    #             loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q
    #
    #     return loss/batch_size
