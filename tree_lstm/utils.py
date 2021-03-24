import torch


def message_func(edges):
    return {'h': edges.src['h'], 'c': edges.src['c']}


def apply_node_func(self, nodes):
    iou = nodes.data['iou'] + self.b_iou
    i, o, u = torch.chunk(iou, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
    c = i * u + nodes.data['c']
    h = o * torch.tanh(c)
    return {'h': h, 'c': c}
