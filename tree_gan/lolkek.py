import pandas as pd

base_dir = '../data/'
actions_list = pd.read_csv(base_dir + 'actions.csv', index_col=0)['action_name'].to_list()
rules_list = pd.read_csv(base_dir + 'rules.csv', index_col=0)['rule_name'].to_list()
data_df = pd.read_csv(base_dir + 'data.csv')

print(data_df.groupby('file_id').size().describe())
