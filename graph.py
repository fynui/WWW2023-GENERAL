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


# class Graph:
#
#     def __init__(self, nodes_feature: np.ndarray, edges: np.ndarray,
#                  edges_feature: np.ndarray = None, nodes_label: np.ndarray = None):
#         super(Graph, self).__init__()
#         self.node_num = len(nodes_feature)
#         self.edge_num = len(edges)
#         self.edges = edges
#         self.hyperedges = self.graph2hyper()
#         self.nodes_feature = nodes_feature
#         if edges_feature is not None:
#             self.edges_feature = edges_feature
#         else:
#             self.edges_feature = self.get_edges_feature()
#         self.edge_index = self.get_edge_index()
#         self.hyperedge_index = self.get_hyperedge_index()
#         self.nodes_label = nodes_label
#         self.label_num = len(np.unique(nodes_label))
#
#     def get_edges_feature(self):
#         edges_feature = []
#         for edge in self.edges:
#             a_, b_ = self.nodes_feature[edge[0]], self.nodes_feature[edge[1]]
#             edge_feature = (a_ + b_) / 2
#             edges_feature.append(edge_feature)
#         return np.array(edges_feature)
#
#     def graph2hyper(self):
#         hyperedges = [[] for i in range(self.node_num)]
#         for i in range(self.edge_num):
#             a, b = self.edges[i]
#             hyperedges[a].append(i)
#             hyperedges[b].append(i)
#         return hyperedges
#
#     def get_edge_index(self):
#         return self.edges.T
#
#     def get_hyperedge_index(self):
#         hyperedge_index = []
#         for i in range(self.node_num):
#             hyperedge = np.array(self.hyperedges[i])
#             index = np.empty_like(hyperedge)
#             index.fill(i)
#             hyperedge_index.append(np.stack([hyperedge, index]))
#         return np.concatenate(hyperedge_index, axis=1)
