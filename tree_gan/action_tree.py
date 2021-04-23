import antlr4
from antlr4.tree.Trees import Trees
import antlr4.tree.Tree
from gen.parser.KotlinLexer import KotlinLexer
from gen.parser.KotlinParser import KotlinParser


class Node:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.name))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(repr(self))


class NonTerminal(Node):
    def __init__(self, name):
        super().__init__(name)


class Terminal(Node):
    def __init__(self, name):
        super().__init__(name)


class ActionTree:
    def __init__(self):
        pass

    @classmethod
    def from_antlr_file(cls, file_path):
