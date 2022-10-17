import torch
from torch import nn
from torch.nn import functional as F


class NodeClassification(nn.Module):

    def __init__(self, embedding_dim, label_num):
        super(NodeClassification, self).__init__()
        self.linear = nn.Linear(embedding_dim, label_num)
        # self.linear1 = nn.Linear(embedding_dim, 16)
        # self.linear2 = nn.Linear(16, label_num)

    def forward(self, nodes_embedding):
        x = self.linear(nodes_embedding)
        # x = self.linear1(nodes_embedding)
        # x = F.leaky_relu_(x)
        # x = F.dropout(x, training=self.training)
        # x = self.linear2(x)
        return torch.log_softmax(x, dim=-1)

    def evaluate(self, nodes_embedding, nodes_label):
        log_prob = self.forward(nodes_embedding)
        pred = log_prob.argmax(dim=-1)
        correct = torch.eq(pred, nodes_label)
        acc = correct.sum() / len(correct)
        return acc
