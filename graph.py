import torch
from torch_geometric.datasets import Planetoid, PPI, Reddit


class Graph:

    def __init__(self, dataset):
        super(Graph, self).__init__()
        dataset = self.get_data(data_name=dataset)
        self.nodes_feature = dataset.data.x
        self.nodes_label = dataset.data.y
        self.edge_index = dataset.data.edge_index
        self.node_num = self.nodes_feature.size(0)
        self.edge_num = self.edge_index.size(1)
        self.edges_feature = self.get_edges_feature()
        self.hyperedge_index = self.edge2hyperedge()
        self.label_num = len(torch.unique(self.nodes_label))
        self.train_mask = dataset.data.train_mask
        self.val_mask = dataset.data.val_mask
        self.test_mask = dataset.data.test_mask

    def edge2hyperedge(self):
        edges = self.edge_index.T
        hyperedges = [[] for i in range(self.node_num)]
        for i in range(self.edge_num):
            a, b = edges[i]
            hyperedges[a].append(i)
            hyperedges[b].append(i)
        hyperedge_index = []
        for i in range(self.node_num):
            hyperedge = torch.tensor(hyperedges[i], dtype=torch.long)
            index = torch.empty_like(hyperedge).fill_(i)
            hyperedge_index.append(torch.stack([hyperedge, index]))
        return torch.cat(hyperedge_index, dim=1)

    def get_edges_feature(self):
        edges_feature = []
        for edge in self.edge_index.T:
            a_, b_ = self.nodes_feature[edge[0]], self.nodes_feature[edge[1]]
            edge_feature = (a_ + b_) / 2
            # edge_feature = torch.cat([a_, b_], dim=0)
            edges_feature.append(edge_feature)
        return torch.stack(edges_feature)

    @staticmethod
    def get_data(root_dir='data', data_name='Cora'):
        dataset = Planetoid(root=root_dir, name=data_name)
        return dataset
