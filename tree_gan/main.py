import os
import torch
from generator import Generator
import pandas as pd
import numpy as np
from batch_loader import BatchLoader
from tqdm import trange, tqdm

CUDA = False
ACTIONS_EMBEDDING_DIM = 128
RULES_EMBEDDING_DIM = 64
GEN_HIDDEN_DIM = 64
PRETRAIN_EPOCHS = 10
BATCH_SIZE = 64
MAX_SEQ_LEN = 1024


def pretrain_generator(gen, gen_opt, epochs):
    global data_df
    n_iter = 0
    loader = BatchLoader(data_df)
    for epoch in range(epochs):
        print(f'epoch = {epoch} --------------------------------')
        total_loss = 0
        n_iter += 1
        for batch in tqdm(loader.load_action_batch(MAX_SEQ_LEN, BATCH_SIZE, CUDA),
                          total=int(NUM_SAMPLES / BATCH_SIZE / MAX_SEQ_LEN)):
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(batch)
            loss.backward()
            gen_opt.step()
            total_loss += loss.data.item()
        total_loss /= NUM_SAMPLES / BATCH_SIZE / MAX_SEQ_LEN
        print('iteration = %d, NLL loss = %.4f' % (n_iter, total_loss))


def main():
    global actions_list, rules_list
    action_matrix = np.zeros(len(rules_list), len(actions_list))
    rule_to_index = {rule: i for i, rule in rules_list}
    action_to_index = {}
    for i, action in enumerate(actions_list):
        action_rules = action.split('_')
        j = rule_to_index[action_rules[0]]
        action_matrix[j, i] = 1.
        action_to_index[i] = []
        for child in action_rules[1:]:
            action_to_index[i].append(action_to_index[child])

    gen = Generator(ACTIONS_EMBEDDING_DIM, RULES_EMBEDDING_DIM, action_to_index, list(range(len(rules_list))),
                    GEN_HIDDEN_DIM, MAX_SEQ_LEN, action_matrix, CUDA)
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
