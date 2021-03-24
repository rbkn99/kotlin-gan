import torch
import dgl


class Tree:
    def __init__(self, h_size):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size

    def add_node(self, parent_id=None, tensor: torch.Tensor = torch.Tensor()):
        added_node_id = self.add_node_dgl(tensor)
        if parent_id:
            self.add_link(added_node_id, parent_id)
        return added_node_id

    def add_node_bottom_up(self, child_ids, tensor: torch.Tensor):
        added_node_id = self.add_node_dgl(tensor)
        for child_id in child_ids:
            self.add_link(child_id, added_node_id)
        return added_node_id

    def add_link(self, child_id, parent_id):
        self.dgl_graph.add_edges([child_id], [parent_id])

    def add_node_dgl(self, tensor: torch.Tensor):
        self.dgl_graph.add_nodes(1, data={'x': tensor.unsqueeze(0),
                                          'h': tensor.new_zeros(size=(1, self.h_size)),
                                          'c': tensor.new_zeros(size=(1, self.h_size))})
        added_node_id = self.dgl_graph.number_of_nodes() - 1
        return added_node_id
