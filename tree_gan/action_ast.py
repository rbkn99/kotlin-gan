from typing import Tuple, List


class Action:
    def __init__(self, rule: str, prev: str, parent: str, children: List):
        self.rule = rule
        self.prev = prev
        self.parent = parent
        self.children = children

    def __str__(self):
        s = self.rule + '_' + '_'.join([child.rule for child in self.children]) + '\n'
        for child in self.children:
            s += str(child)
        return s


class ASTInfo:
    def __init__(self, file_path):
        self.str_actions = open(file_path, 'r').readlines()
        self.head = self.traverse(0, None)

    def traverse(self, action_index, parent_index) -> Tuple[Action, int]:
        print(action_index, len(self.str_actions))
        str_action = self.str_actions[action_index].split('_')
        rule = str_action[0]
        if action_index == 0:
            prev = None
            parent = None
        else:
            prev = self.str_actions[action_index - 1]
            parent = self.str_actions[parent_index].split('_')[0]
        children = []
        next_action_index = action_index
        for _ in str_action[1:]:
            child_action, next_action_index = self.traverse(next_action_index + 1, action_index)
            children.append(child_action)
        new_action = Action(rule, prev, parent, children)
        return new_action, action_index
