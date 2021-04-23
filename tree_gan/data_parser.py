import os
import logging
import antlr4
from antlr4.tree.Trees import Trees
import antlr4.tree.Tree
import pandas as pd
from gen.parser.KotlinLexer import KotlinLexer
from gen.parser.KotlinParser import KotlinParser


def ast_to_actions_seq(node, rule_names, file_id, parent_id: int, action_id):
    global rules_dict, rules, actions_dict, action_counter
    if isinstance(node, antlr4.tree.Tree.TerminalNodeImpl):
        rule_name = KotlinLexer.symbolicNames[node.getSymbol().type]
    else:
        rule_name = Trees.getNodeText(node, rule_names)
    if rule_name not in rules_dict:
        new_id = max(rules_dict.values(), default=-1) + 1
        rules_dict[rule_name] = new_id
        rules.append(rule_name)
    action_str = rule_name
    rule_seq = [[action_id, file_id, rules_dict[rule_name], parent_id, action_counter]]
    action_counter += 1
    if not isinstance(node, antlr4.tree.Tree.TerminalNodeImpl):
        next_action_id = action_id
        for child in node.getChildren():
            child_subtree_seq = ast_to_actions_seq(child, rule_names,
                                                   file_id, action_id, next_action_id + 1)
            rule_seq.extend(child_subtree_seq)
            action_str += '_' + Trees.getNodeText(child, rule_names)
            next_action_id = rule_seq[-1][0]
    if action_str not in actions_dict:
        actions_dict[action_str] = max(actions_dict.values(), default=-1) + 1
    rule_seq[0].append(actions_dict[action_str])
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
    for root, dirs, files in os.walk(kotlin_test_box_path):
        for file in files:
            if file.endswith(".kt"):
                file_path = os.path.join(root, file)
                logger.debug("Processing \"%s\" id: %d", file_path, file_id)
                files_dict[file_path] = file_id
                actions_seq = process_file(file_path, file_id)
                file_id += 1
                if file_id > 10:
                    break
                actions_list.extend(actions_seq)
        if file_id > 10:
            break
    actions_df = pd.DataFrame(actions_list, columns=df_columns)
    actions_df.to_csv('../data/data.csv', index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('data_logger')
    files_dict = dict()
    rules_dict = dict()
    actions_dict = dict()
    action_counter = 0
    rules = []
    main()
    pd.DataFrame({'id': files_dict.values(),
                  'file_path': files_dict.keys()}).to_csv('../data/files.csv', index=False)
    pd.DataFrame({'id': rules_dict.values(),
                  'rule_name': rules_dict.keys()}).to_csv('../data/rules.csv', index=False)
    pd.DataFrame({'id': actions_dict.values(),
                  'action_name': actions_dict.keys()}).to_csv('../data/actions.csv', index=False)
