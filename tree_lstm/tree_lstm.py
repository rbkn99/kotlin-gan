from copy import deepcopy

import dgl
import torch

from tree_lstm import BatchedTree, ChildSumTreeLSTMCell, NaryTreeLSTMCell
from tree_lstm.utils import message_func, apply_node_func


class TreeLSTM(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 cell_type='n_ary',
                 n_ary=None,
                 num_stacks=2):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        if cell_type == 'n_ary':
            self.cell = NaryTreeLSTMCell(n_ary, x_size, h_size)
        else:
            self.cell = ChildSumTreeLSTMCell(x_size, h_size)
        self.num_stacks = num_stacks

    def forward(self, batch: BatchedTree):
        batches = [deepcopy(batch) for _ in range(self.num_stacks)]
        for stack in range(self.num_stacks):
            cur_batch = batches[stack]
            if stack > 0:
                prev_batch = batches[stack - 1]
                cur_batch.batch_dgl_graph.ndata['x'] = prev_batch.batch_dgl_graph.ndata['h']
            cur_batch.batch_dgl_graph.update_all(message_func, self.cell.reduce_func, apply_node_func)
            cur_batch.batch_dgl_graph.ndata['iou'] = self.cell.W_iou(self.dropout(batch.batch_dgl_graph.ndata['x']))
            dgl.prop_nodes_topo(cur_batch.batch_dgl_graph, message_func, self.cell.reduce_func, False, apply_node_func)
        return batches
