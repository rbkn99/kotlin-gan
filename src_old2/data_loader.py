import os
import pickle

import torch
from torch.utils.data import Dataset
import parse_utils



def rule_representation(tree: KotlinParser.KotlinFileContext, rule_names):
    parent = Trees.getNodeText(tree, rule_names)
    production_rule = [parent]
    for child in tree.getChildren():
        if isinstance(child, antlr4.tree.Tree.TerminalNodeImpl):
            symbolicName = KotlinLexer.symbolicNames[child.getSymbol().type]
            production_rule.append(symbolicName)
        else:
            production_rule.append(Trees.getNodeText(child, rule_names))
    production_rule = '_'.join(production_rule)
    return production_rule


class ActionSequenceDataset(Dataset):
    def __init__(self, grammar_path, kotlin_test_box_path, action_getter_path='', action_sequences_dir=''):
        super().__init__()
        self.kotlin_test_box_path = kotlin_test_box_path
        self.action_sequences_dir = action_sequences_dir
        self.kt_files = parse_utils.Enumerator()
        for root, dirs, files in os.walk(kotlin_test_box_path):
            for file in files:
                if file.endswith(".kt"):
                    self.kt_files.append(os.path.join(root, file))
        # Get rule dictionary of the language
        # First check if action getter already exists, if not then parse the language grammar to create it
        if os.path.exists(action_getter_path):
            with open(action_getter_path, 'rb') as f:
                action_getter = pickle.load(f)
        else:
            my_bnf_parser = parse_utils.CustomBNFParser()
            _, rules_dict, symbol_names = my_bnf_parser.parse_file(bnf_path, start=lang_grammar_start)
            action_getter = parse_utils.SimpleTreeActionGetter(rules_dict, symbol_names)
            if action_getter_path:
                with open(action_getter_path, 'wb') as f:
                    pickle.dump(action_getter, f)
        self.action_getter = action_getter

        with open(lark_path) as f:
            self.parser = Lark(f, keep_all_tokens=True, start=lang_grammar_start)

    def index(self, text_filename):
        return self.text_filenames.index(text_filename)

    def __getitem__(self, index):
        # First check if action sequence of parse tree of the text file already exists, if not then calculate it
        text_filename = self.text_filenames[index]
        text_file_path = os.path.join(self.texts_dir, text_filename)
        text_action_sequence_path = os.path.join(self.action_sequences_dir, text_filename + '.pickle')
        if os.path.exists(text_action_sequence_path):
            with open(text_action_sequence_path, 'rb') as f:
                action_sequences = pickle.load(f)
        else:
            with open(text_file_path) as f:
                # Get parse tree of the text file written in the language defined by the given grammar
                text_tree = self.parser.parse(f.read(), start=self.start)
            id_tree = self.action_getter.simple_tree_to_id_tree(parse_utils.SimpleTree.from_lark_tree(text_tree))
            # Get sequence of actions taken by each non-terminal symbol in 'prefix DFS left-to-right' order
            action_sequences = self.action_getter.collect_actions(id_tree)
            if self.action_sequences_dir:
                with open(text_action_sequence_path, 'wb') as f:
                    pickle.dump(action_sequences, f)

        actions, parent_actions = action_sequences
        return torch.tensor(actions), torch.tensor(parent_actions)

    def __len__(self):
        return len(self.text_filenames)
