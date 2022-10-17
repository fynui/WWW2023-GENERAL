import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, HypergraphConv
import logging


class GCNNet(nn.Module):

    def __init__(self, feature_dim, embedding_dim):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(feature_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)

    def forward(self, nodes_feature, edge_index):
        x = self.conv1(nodes_feature, edge_index)
        x = F.leaky_relu_(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu_(x)
        return x
        # return F.log_softmax(x, dim=1)


class HyperGCN(nn.Module):

    def __init__(self, feature_dim, embedding_dim):
        super(HyperGCN, self).__init__()
        self.conv1 = HypergraphConv(feature_dim, embedding_dim)
        self.conv2 = HypergraphConv(embedding_dim, embedding_dim)

    def forward(self, nodes_features, hyperedge_index):
        x = self.conv1(nodes_features, hyperedge_index)
        x = F.leaky_relu_(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, hyperedge_index)
        x = F.leaky_relu_(x)
        return x
        # return F.log_softmax(x, dim=1)


class Contrast(nn.Module):

    def __init__(self, embedding_dim, map_dim, temperature=1.0):
        super(Contrast, self).__init__()
        self.linear_node = Linear(embedding_dim * 2, map_dim)
        self.linear_edge = Linear(embedding_dim, map_dim)
        self.t = temperature

    def forward(self, nodes_embedding, edges_embedding, edge_index):
        nodes_select = nodes_embedding.index_select(dim=0, index=edge_index.reshape(-1))
        nodes_concat = torch.cat(nodes_select.chunk(2, dim=0), dim=-1)
        nodes_map = self.linear_node(nodes_concat)
        edges_map = self.linear_edge(edges_embedding)
        return self.InfoNCELoss(nodes_map, edges_map)

    def InfoNCELoss(self, nodes_map, edges_map):
        # similarity function 1: negative distance
        # map_bias = nodes_map - edges_map
        # map_dif = nodes_map.unsqueeze(dim=1) - edges_map.unsqueeze(dim=0)
        # MI = torch.exp(-map_bias.norm(p=2, dim=-1))
        # MI_total = torch.exp(-map_dif.norm(p=2, dim=-1))

        # similarity function 2: negative inner-product
        nodes_norm = nodes_map.norm(p=2, dim=-1)
        edges_norm = edges_map.norm(p=2, dim=-1)
        inner_product = torch.mul(nodes_map, edges_map).sum(dim=-1) / (nodes_norm * edges_norm)
        mat_product = torch.matmul(nodes_map, edges_map.T) / (nodes_norm.unsqueeze(dim=1) * edges_norm.unsqueeze(dim=0))
        # MI = torch.exp(inner_product / self.t)
        # MI_total = torch.exp(mat_product / self.t)
        MI = torch.exp(-torch.abs(inner_product) / self.t)
        MI_total = torch.exp(-torch.abs(mat_product) / self.t)
        prop = 2 * MI / (MI_total.sum(dim=0) + MI_total.sum(dim=1))
        return -torch.log(prop)


class GCL(nn.Module):

    def __init__(self, feature_dim, embedding_dim, map_dim, temperature=1.0):
        super(GCL, self).__init__()
        self.gcn = GCNNet(feature_dim, embedding_dim)
        self.hypergcn = HyperGCN(feature_dim, embedding_dim)
        self.contrast = Contrast(embedding_dim, map_dim, temperature)

    def forward(self, nodes_feature, edges_feature, edge_index, hyperedge_index):
        nodes_embedding = self.gcn(nodes_feature, edge_index)
        edges_embedding = self.hypergcn(edges_feature, hyperedge_index)
        return self.contrast(nodes_embedding, edges_embedding, edge_index)
