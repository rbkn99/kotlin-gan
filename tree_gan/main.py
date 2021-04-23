import os
import torch
from generator import Generator
import pandas as pd
from torch.autograd import Variable

CUDA = False
ACTIONS_EMBEDDING_DIM = 128
RULES_EMBEDDING_DIM = 64
ACTIONS_VOCAB_SIZE = 6000
RULES_VOCAB_SIZE = 315
GEN_HIDDEN_DIM = 64
PRETRAIN_EPOCHS = 10
BATCH_SIZE = 64
SEQ_LEN = 20


def load_action_batch():
    global data_df
    data_np = data_df.values
    n = SEQ_LEN * BATCH_SIZE
    for i in range(0, data_np.shape[0] - n - 1, n):
        prev_batch = torch.zeros(BATCH_SIZE, SEQ_LEN)
        parent_batch = torch.zeros(BATCH_SIZE, SEQ_LEN)
        target_batch = torch.zeros(BATCH_SIZE, SEQ_LEN)
        for j in range(BATCH_SIZE):
            for q in range(SEQ_LEN):
                pos = j * BATCH_SIZE + q
                if data_np[pos, 4] == -1:
                    prev_batch[j, q] = data_np[pos, 5]
                else:
                    prev_batch[j, q] = data_np[data_np[pos, 4], 5]
                if data_np[pos, 3] == -1:
                    parent_batch[j, q] = data_np[pos, 2]
                else:
                    parent_batch[j, q] = data_np[data_np[pos, 3], 2]
                target_batch[j, q] = data_np[pos + 1, 5]
        prev_batch = Variable(prev_batch).type(torch.LongTensor)
        parent_batch = Variable(prev_batch).type(torch.LongTensor)
        target_batch = Variable(prev_batch).type(torch.LongTensor)
        yield prev_batch, parent_batch, target_batch


def pretrain_generator(gen, gen_opt, epochs):
    for epoch in range(epochs):
        total_loss = 0
        n_iter = 0
        for prev, parent, target in load_action_batch():
            n_iter += 1
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(prev, parent, target)
            loss.backward()
            gen_opt.step()
            total_loss += loss.data.item()
        total_loss /= (data_df.shape[0] / BATCH_SIZE) / SEQ_LEN
        print('iteration = %d, NLL loss = %.4f' % (n_iter, total_loss))


def main():
    gen = Generator(ACTIONS_EMBEDDING_DIM, RULES_EMBEDDING_DIM,
                    ACTIONS_VOCAB_SIZE, RULES_VOCAB_SIZE, GEN_HIDDEN_DIM, SEQ_LEN, [], CUDA)
    if CUDA:
        gen = gen.cuda()
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-2)

    pretrain_generator(gen, gen_optimizer, PRETRAIN_EPOCHS)


if __name__ == '__main__':
    actions_list = pd.read_csv('../data/actions.csv', index_col=0)['action_name'].to_list()
    rules_list = pd.read_csv('../data/rules.csv', index_col=0)['rule_name'].to_list()
    data_df = pd.read_csv('../data/data.csv')
    main()
    # rules_dir_path = os.path.abspath('../data/rules')
    # actions_data = []
    # actions_matrix = []
    #
    # for rule_file_path in os.listdir(rules_dir_path):
    #     cur_ast = ASTInfo(os.path.join(rules_dir_path, rule_file_path))
    #     print(cur_ast.head)
    #     file_actions = open(rule_file_path, 'r').readlines()
    #     rules.update([action.split('_')[0] for action in file_actions])
    #     actions.update(file_actions)
    #     actions_data.append(file_actions)
    # actions_matrix = [[1 if action.startswith(rule) else 0 for action in actions] for rule in rules]

