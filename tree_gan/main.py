import os
import torch
from generator import Generator
import pandas as pd
from batch_loader import BatchLoader
from tqdm import trange, tqdm

CUDA = False
ACTIONS_EMBEDDING_DIM = 128
RULES_EMBEDDING_DIM = 64
ACTIONS_VOCAB_SIZE = 6000
RULES_VOCAB_SIZE = 315
GEN_HIDDEN_DIM = 64
PRETRAIN_EPOCHS = 10
BATCH_SIZE = 64
SEQ_LEN = 20


def pretrain_generator(gen, gen_opt, epochs):
    n_iter = 0
    loader = BatchLoader(data_df)
    for epoch in range(epochs):
        print(f'epoch = {epoch} --------------------------------')
        total_loss = 0
        n_iter += 1
        for prev, parent, target in tqdm(loader.load_action_batch(SEQ_LEN, BATCH_SIZE, CUDA),
                                         total=int(NUM_SAMPLES / BATCH_SIZE / SEQ_LEN)):
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(prev, parent, target)
            loss.backward()
            gen_opt.step()
            total_loss += loss.data.item()
        total_loss /= NUM_SAMPLES / BATCH_SIZE / SEQ_LEN
        print('iteration = %d, NLL loss = %.4f' % (n_iter, total_loss))


def main():
    gen = Generator(ACTIONS_EMBEDDING_DIM, RULES_EMBEDDING_DIM,
                    ACTIONS_VOCAB_SIZE, RULES_VOCAB_SIZE, GEN_HIDDEN_DIM, SEQ_LEN, [], CUDA)
    if CUDA:
        gen = gen.cuda()
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-2)

    pretrain_generator(gen, gen_optimizer, PRETRAIN_EPOCHS)


if __name__ == '__main__':
    base_dir = 'drive/MyDrive/TreeGAN-data/'
    actions_list = pd.read_csv(base_dir + 'actions.csv', index_col=0)['action_name'].to_list()
    rules_list = pd.read_csv(base_dir + 'rules.csv', index_col=0)['rule_name'].to_list()
    data_df = pd.read_csv(base_dir + 'data.csv')
    NUM_SAMPLES = data_df.shape[0]

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
