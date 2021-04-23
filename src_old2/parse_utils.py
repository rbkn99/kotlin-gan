import codecs
import io
import os
import re
import sys
from collections import namedtuple

import antlr4
from antlr4.tree.Trees import Trees
import antlr4.tree.Tree
from gen.parser.KotlinLexer import KotlinLexer
from gen.parser.KotlinParser import KotlinParser

import learning_utils

DerivationSymbol = namedtuple('DerivationSymbol', ['name', 'parent_action'])


class _Symbol:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.name))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(repr(self))


class NonTerminal(_Symbol):
    def __init__(self, name):
        super().__init__(name)


class Terminal(_Symbol):
    def __init__(self, name):
        super().__init__(name)


class SimpleTree:
    def __init__(self, data="", children=None, used_rule=()):
        self.data = data
        self.children = [] if children is None else children
        self.used_rule = used_rule

    @classmethod
    def from_antlr_tree(cls, tree: KotlinParser.KotlinFileContext):
        children = [Terminal(KotlinLexer.symbolicNames[child.getSymbol().type])
                    if isinstance(child, antlr4.tree.Tree.TerminalNodeImpl) else SimpleTree.from_antlr_tree(child)
                    for child in tree.getChildren()]
        used_rule = []
        if children:
            for symbol in children:
                symbol = NonTerminal(symbol.data) if isinstance(symbol, SimpleTree) else symbol
                used_rule.append(symbol)
        else:
            used_rule.append(Terminal(''))
        return cls(Trees.toStringTree(tree, KotlinParser.ruleNames), children, tuple(used_rule))

    def __repr__(self):
        return 'Tree(%s, %s)' % (self.data, self.children)

    def pretty(self, stream=None):
        stream = io.StringIO() if stream is None else stream
        self.pretty_stream(stream, 0)
        return stream.getvalue()

    def pretty_stream(self, stream, depth=0):
        indentation = '| ' * depth
        print(indentation + self.data, file=stream)
        for t in self.children:
            if isinstance(t, SimpleTree):
                t.pretty_stream(stream, depth + 1)
            else:
                print(indentation + '| ' + repr(t), file=stream)


class Enumerator:
    def __init__(self, values=None):
        self._value_to_index = {}
        self._index_to_value = []
        if values:
            for value in values:
                self.append(value)

    def append(self, value):
        if value not in self:
            self._value_to_index[value] = len(self._index_to_value)
            self._index_to_value.append(value)

    def pop(self):
        last_value = self._index_to_value[-1]
        del self._value_to_index[last_value], self._index_to_value[-1]
        return last_value

    def index(self, value):
        return self._value_to_index[value]

    def __contains__(self, value):
        return value in self._value_to_index

    def __getitem__(self, index):
        return self._index_to_value[index]

    def __len__(self):
        return len(self._value_to_index)

    def clear(self):
        self._value_to_index = {}
        self._index_to_value = []

    def __repr__(self):
        return 'Enumerator(%s)' % repr(self._index_to_value)


class CustomBNFParser:
    ESCAPE_SEQUENCE_RE = re.compile(r'''
        ( \\U........      # 8-digit hex escapes
        | \\u....          # 4-digit hex escapes
        | \\x..            # 2-digit hex escapes
        | \\[0-7]{1,3}     # Octal escapes
        | \\N\{[^}]+\}     # Unicode characters by name
        | \\[\\'"abfnrtv]  # Single-character escapes
        )''', re.UNICODE | re.VERBOSE)

    @staticmethod
    def decode_escapes(s):
        def decode_match(match):
            return codecs.decode(match.group(0), 'unicode-escape')

        return CustomBNFParser.ESCAPE_SEQUENCE_RE.sub(decode_match, s)

    def __init__(self, grammar_file_path=None):
        if grammar_file_path is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            grammar_file_path = os.path.join(package_dir, '..', 'data', 'bnf_lang', 'bnf.lark')
        with open(grammar_file_path) as f:
            self.parser = lark.Lark(f)

    def parse_file(self, text_file_path, start=None):
        with open(text_file_path) as f:
            return self.parse(f.read(), start=start)

    def parse(self, text, start=None):
        tree = self.parser.parse(text, start=start)
        rules_dict = {}
        symbol_names = Enumerator()
        syntax = tree.children[0]
        for syntax_child_index in range(1, len(syntax.children), 3):
            syntax_child = syntax.children[syntax_child_index]
            if syntax_child.data == 'comment':
                continue
            assert syntax_child.data == 'rule', 'syntax_child.data: %s' % syntax_child.data
            rule = syntax_child
            non_terminal, expression = rule.children[0], rule.children[4]
            rule_text = non_terminal.children[1]
            rule_name_obj = NonTerminal(''.join([token.value for token in rule_text.children]))
            symbol_names.append(rule_name_obj)
            rule_name_obj = symbol_names.index(rule_name_obj)

            expression_obj = Enumerator()
            for sequence_index in range(0, len(expression.children), 4):
                sequence = expression.children[sequence_index]
                sequence_obj = []
                for term_index in range(0, len(sequence.children), 2):
                    term = sequence.children[term_index]
                    if term.children[0].data == 'terminal':
                        terminal = term.children[0]
                        text_i = terminal.children[1]
                        term_obj = Terminal(
                            CustomBNFParser.decode_escapes(''.join([token.value for token in text_i.children])))
                    else:  # term.children[0].data == 'non_terminal'
                        rhs_non_terminal = term.children[0]
                        rhs_rule_text = rhs_non_terminal.children[1]
                        term_obj = NonTerminal(''.join([token.value for token in rhs_rule_text.children]))
                    symbol_names.append(term_obj)
                    term_obj = symbol_names.index(term_obj)
                    sequence_obj.append(term_obj)
                expression_obj.append(tuple(sequence_obj))
            rules_dict[rule_name_obj] = expression_obj

        return tree, rules_dict, symbol_names

    @staticmethod
    def print_rules_dict(rules_dict, symbol_names, stream=sys.stdout):
        indentation = ' ' * 4
        for non_terminal_id, expression in rules_dict.items():
            print(str(symbol_names[non_terminal_id]) + ':', file=stream)
            for sequence in expression:
                print(indentation + '[%s]' % str([symbol_names[symbol_id] for symbol_id in sequence]), file=stream)


class SimpleTreeActionGetter:
    def __init__(self, rules_dict, symbol_names):
        self.rules_dict = rules_dict
        self.symbol_names = symbol_names
        self.action_offsets = {}
        self.non_terminal_ids = Enumerator(rules_dict.keys())
        cum_sum = 0
        for non_terminal_id, rules in rules_dict.items():
            self.action_offsets[non_terminal_id] = cum_sum
            cum_sum += len(rules)

    def collect_actions(self, id_tree):
        actions, parent_actions = [], []
        self._collect_actions(id_tree, actions, parent_actions)
        return actions, parent_actions

    def _collect_actions(self, id_tree, actions, parent_actions):
        id_tree_stack = [id_tree]
        parent_action_stack = [learning_utils.PADDING_ACTION]
        while id_tree_stack:
            id_tree = id_tree_stack.pop()
            non_terminal_id = id_tree.data
            action = self.action_offsets[non_terminal_id] + self.rules_dict[non_terminal_id].index(id_tree.used_rule)
            actions.append(action)
            parent_actions.append(parent_action_stack.pop())
            for child in reversed(id_tree.children):
                if isinstance(child, SimpleTree):
                    id_tree_stack.append(child)
                    parent_action_stack.append(action)

    def construct_text_partial(self, actions, start='start'):
        actions_iterator = iter(actions)
        stream = io.StringIO()
        try:
            self._construct_text(self.symbol_names.index(NonTerminal(start)), actions_iterator, stream)
        except StopIteration:
            pass
        return stream.getvalue()

    def construct_text(self, actions, start='start'):
        actions_iterator = iter(actions)
        stream = io.StringIO()
        self._construct_text(self.symbol_names.index(NonTerminal(start)), actions_iterator, stream)
        return stream.getvalue()

    def _construct_text(self, non_terminal_id, actions_iterator, stream):
        non_terminal_id_stack = [non_terminal_id]
        used_rule_stack = []
        symbol_count_stack = [0]
        while non_terminal_id_stack:
            non_terminal_id = non_terminal_id_stack[-1]
            if len(used_rule_stack) == len(non_terminal_id_stack) - 1:
                action = next(actions_iterator) - self.action_offsets[non_terminal_id]
                used_rule = self.rules_dict[non_terminal_id][action]
                used_rule_stack.append(used_rule)
            else:
                used_rule = used_rule_stack[-1]
            recurse_next_symbol = False
            for symbol_id in used_rule[symbol_count_stack[-1]:]:
                symbol = self.symbol_names[symbol_id]
                symbol_count_stack[-1] += 1
                if isinstance(symbol, Terminal):
                    stream.write(symbol.name)
                else:  # NonTerminal
                    non_terminal_id_stack.append(symbol_id)
                    symbol_count_stack.append(0)
                    recurse_next_symbol = True
                    break
            if recurse_next_symbol:
                continue
            elif symbol_count_stack[-1] == len(used_rule):
                non_terminal_id_stack.pop()
                used_rule_stack.pop()
                symbol_count_stack.pop()

    def construct_parse_tree_partial(self, actions, start='start'):
        action_iterator = iter(actions)
        tree = SimpleTree()
        try:
            self._construct_parse_tree(self.symbol_names.index(NonTerminal(start)), action_iterator, tree)
        except StopIteration:
            pass
        return tree

    def construct_parse_tree(self, actions, start='start'):
        action_iterator = iter(actions)
        tree = SimpleTree()
        self._construct_parse_tree(self.symbol_names.index(NonTerminal(start)), action_iterator, tree)
        return tree

    def _construct_parse_tree(self, non_terminal_id, actions_iterator, tree):
        non_terminal_id_stack = [non_terminal_id]
        tree_stack = [tree]
        while non_terminal_id_stack:
            non_terminal_id = non_terminal_id_stack.pop()
            tree = tree_stack.pop()
            action = next(actions_iterator)
            tree.data = self.symbol_names[non_terminal_id].name
            tree.children = []
            tree.used_rule = tuple(self.symbol_names[symbol_id] for symbol_id in
                                   self.rules_dict[non_terminal_id][action - self.action_offsets[non_terminal_id]])
            for child in reversed(tree.used_rule):
                if isinstance(child, NonTerminal):
                    child_tree = SimpleTree()
                    tree.children.append(child_tree)
                    non_terminal_id_stack.append(self.symbol_names.index(child))
                    tree_stack.append(child_tree)
                else:
                    tree.children.append(child)
            tree.children = list(reversed(tree.children))

    def simple_tree_to_id_tree(self, tree):
        root_id_tree = SimpleTree()
        non_terminal_id_stack = [self.symbol_names.index(NonTerminal(tree.data))]
        id_tree_stack = [root_id_tree]
        children_stack = [tree.children]
        while non_terminal_id_stack:
            non_terminal_id = non_terminal_id_stack.pop()
            id_tree = id_tree_stack.pop()
            children = children_stack.pop()
            # Start filling the current id tree
            id_tree.data = non_terminal_id
            used_rule = []
            if children:
                for child in reversed(children):
                    if isinstance(child, SimpleTree):
                        symbol_id = self.symbol_names.index(NonTerminal(child.data))
                        child_id_tree = SimpleTree()
                        id_tree.children.append(child_id_tree)
                        # Push next id trees to be filled
                        non_terminal_id_stack.append(symbol_id)
                        id_tree_stack.append(child_id_tree)
                        children_stack.append(child.children)
                    else:
                        symbol_id = self.symbol_names.index(child)
                        id_tree.children.append(symbol_id)
                    used_rule.append(symbol_id)
            else:
                used_rule.append(self.symbol_names.index(Terminal('')))
            id_tree.children = list(reversed(id_tree.children))
            id_tree.used_rule = tuple(reversed(used_rule))
        return root_id_tree
