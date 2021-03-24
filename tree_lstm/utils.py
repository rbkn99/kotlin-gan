import torch


def message_func(edges):
    return {'h': edges.src['h'], 'c': edges.src['c']}
