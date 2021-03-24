import torch
import dgl


class BatchedTree:
    def __init__(self, tree_list):
        graph_list = []
        for tree in tree_list:
            graph_list.append(tree.dgl_graph)
        self.batch_dgl_graph = dgl.batch(graph_list)

    def get_hidden_state(self):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        hidden_states = []
        max_nodes_num = max([len(graph.nodes) for graph in graph_list])
        for graph in graph_list:
            hiddens = graph.ndata['h']
            node_num, hidden_num = hiddens.size()
            if len(hiddens) < max_nodes_num:
                padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                hiddens = torch.cat((hiddens, padding), dim=0)
            hidden_states.append(hiddens)
        return torch.stack(hidden_states)
