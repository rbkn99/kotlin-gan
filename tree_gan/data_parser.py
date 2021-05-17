import os
import logging
import antlr4
from antlr4.tree.Trees import Trees
import antlr4.tree.Tree
import pandas as pd
from gen.parser.KotlinLexer import KotlinLexer
from gen.parser.KotlinParser import KotlinParser


def ast_to_actions_seq(node, rule_names, file_id, parent_id: int, action_id):
    global rules_dict, actions_dict, action_counter, rules_in_files, actions_in_files
    rule_name = Trees.getNodeText(node, rule_names)
    if rule_name not in rules_dict:
        rules_dict[rule_name] = (len(rules_dict), 1, 1)
        rules_in_files[rule_name] = {file_id}
    else:
        rules_in_files[rule_name].add(file_id)
        rules_dict[rule_name] = (rules_dict[rule_name][0], len(rules_in_files[rule_name]),
                                 rules_dict[rule_name][2] + 1)
    action_str = rule_name
    rule_seq = [[action_id, file_id, rules_dict[rule_name][0], parent_id, action_counter]]
    action_counter += 1
    if not isinstance(node, antlr4.tree.Tree.TerminalNodeImpl):
        next_action_id = action_id
        for child in node.getChildren():
            child_subtree_seq = ast_to_actions_seq(child, rule_names,
                                                   file_id, action_id, next_action_id + 1)
            rule_seq.extend(child_subtree_seq)
            action_str += '##' + Trees.getNodeText(child, rule_names)
            next_action_id = rule_seq[-1][0]
    if action_str not in actions_dict:
        actions_dict[action_str] = (len(actions_dict), 1, 1)
        actions_in_files[action_str] = {file_id}
    else:
        actions_in_files[action_str].add(file_id)
        actions_dict[action_str] = (actions_dict[action_str][0], len(actions_in_files[action_str]),
                                    actions_dict[action_str][2] + 1)
    rule_seq[0].append(actions_dict[action_str][0])
    return rule_seq


def process_file(file_path, file_id):
    input_stream = antlr4.FileStream(file_path)
    lexer = KotlinLexer(input_stream)
    stream = antlr4.CommonTokenStream(lexer)
    parser = KotlinParser(stream)
    root = parser.kotlinFile()
    rule_names = parser.ruleNames
    global action_counter
    action_counter = 0
    return ast_to_actions_seq(root, rule_names, file_id, 0, 1)


def main():
    global files_dict
    kotlin_test_box_path = os.path.abspath('../../kotlin/compiler/testData/codegen/box')
    actions_list = []
    df_columns = ['id', 'file_id', 'rule_id', 'parent_id', 'prev_id', 'action_id']
    file_id = 0
    # counter = 1
    for root, dirs, files in os.walk(kotlin_test_box_path):
        for file in files:
            if file.endswith(".kt"):
                file_path = os.path.join(root, file)
                logger.debug("Processing \"%s\" id: %d", file_path, file_id)
                files_dict[file_path] = file_id
                actions_seq = process_file(file_path, file_id)
                file_id += 1
                actions_list.extend(actions_seq)
        #         counter -= 1
        #     if counter <= 0:
        #         break
        # if counter <= 0:
        #     break
    actions_df = pd.DataFrame(actions_list, columns=df_columns)
    actions_df.to_csv('../data/data.csv', index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('data_logger')
    files_dict = dict()
    rules_dict = dict()
    actions_dict = dict()
    rules_in_files = dict()
    actions_in_files = dict()
    rules_dict['empty'] = (0, 0, 1)
    actions_dict['empty'] = (0, 0, 1)
    rules_in_files['empty'] = set()
    actions_in_files['empty'] = set()
    action_counter = 0
    main()
    pd.DataFrame({'id': files_dict.values(),
                  'file_path': files_dict.keys()}).to_csv('../data/files.csv', index=False)
    pd.DataFrame({'id': list(map(lambda x: x[0], rules_dict.values())),
                  'rule_name': rules_dict.keys(),
                  'files_counter': list(map(lambda x: x[1], rules_dict.values())),
                  'total_counter': list(map(lambda x: x[2], rules_dict.values())),
                  }).to_csv('../data/rules.csv', index=False)
    pd.DataFrame({'id': list(map(lambda x: x[0], actions_dict.values())),
                  'action_name': actions_dict.keys(),
                  'files_counter': list(map(lambda x: x[1], actions_dict.values())),
                  'total_counter': list(map(lambda x: x[2], actions_dict.values())),
                  }).to_csv('../data/actions.csv', index=False)
