import torch


class NaryTreeLSTMCell(torch.nn.Module):
    def __init__(self, n_ary, x_size, h_size):
        super(NaryTreeLSTMCell, self).__init__()
        self.n_ary = n_ary
        self.h_size = h_size
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(n_ary * h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(n_ary * h_size, n_ary * h_size)

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        padding_hs = self.n_ary - nodes.mailbox['h'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['h'].size(0), padding_hs * self.h_size))
        h_cat = torch.cat((h_cat, padding), dim=1)
        f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), self.n_ary, self.h_size)
        padding_cs = self.n_ary - nodes.mailbox['c'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['c'].size(0), padding_cs, self.h_size))
        c = torch.cat((nodes.mailbox['c'], padding), dim=1)
        c = torch.sum(f * c, 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_cat), 'c': c}
