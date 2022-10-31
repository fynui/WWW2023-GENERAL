import argparse
import logging
import numpy as np
import torch
from torch.nn import functional as F
from graph import Graph
from model import GCL
from task import NodeClassification


def parse_args():
    parser = argparse.ArgumentParser(
        description='Graph Contrast Learning',
        usage='run.py [<args>] [-h | --help]')

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--map_dim', default=64, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
    parser.add_argument('-t', '--temperature', default=1, type=float)
    parser.add_argument('-save', '--save_path', default='output', type=str)

    return parser.parse_args()


def set_logger():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )


def log_metrics(step, loss):
    logging.info('loss %s at step %s' % (loss, step))


def main(args):
    set_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g = Graph(args.dataset)
    logging.info('The program is running on the {}'.format(device))
    logging.info("-----------------------Graph attributes-----------------------")
    logging.info('dataset: {}'.format(args.dataset))
    logging.info('node_num: {}'.format(g.node_num))
    logging.info('edge_num: {}'.format(g.edge_num))
    logging.info('edge_index: {}'.format(g.edge_index))
    logging.info('hyperedge_index: {}'.format(g.hyperedge_index))
    logging.info('nodes_feature shape {}'.format(g.nodes_feature.shape))
    logging.info('edges_feature shape {}:'.format(g.edges_feature.shape))
    logging.info('nodes_label: {}'.format(g.nodes_label))
    logging.info('label_num: {}'.format(g.label_num))

    nodes_feature = g.nodes_feature.to(device)
    edges_feature = g.edges_feature.to(device)
    nodes_label = g.nodes_label.to(device)
    edge_index = g.edge_index.to(device)
    hyperedge_index = g.hyperedge_index.to(device)
    train_mask = g.train_mask.to(device)
    val_mask = g.val_mask.to(device)
    test_mask = g.test_mask.to(device)
    label_num = g.label_num
    feature_dim = nodes_feature.size(-1)
    embedding_dim = args.embedding_dim
    map_dim = args.map_dim

    logging.info('-----------------------pre-train-----------------------')
    net = GCL(feature_dim, embedding_dim, map_dim, temperature=args.temperature).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        optimizer.zero_grad()
        loss = net(nodes_feature, edges_feature, edge_index, hyperedge_index).mean()
        logging.info(loss)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        nodes_embedding = net.gcn(nodes_feature, edge_index)
        edges_embedding = net.hypergcn(edges_feature, hyperedge_index)
        logging.info(nodes_embedding)
        logging.info(edges_embedding)

    logging.info('-----------------------Node Classification-----------------------')
    nc = NodeClassification(embedding_dim, label_num).to(device)
    # nc = NodeClassification(feature_dim, label_num).to(device)
    optimizer_nc = torch.optim.Adam(nc.parameters(), lr=0.001, weight_decay=5e-4)

    for epoch in range(10000):
        optimizer_nc.zero_grad()
        log_prob = nc(nodes_embedding)
        # log_prob = nc(nodes_feature)
        loss_nc = F.nll_loss(log_prob[train_mask], nodes_label[train_mask])
        loss_nc.backward()
        optimizer_nc.step()
        if epoch % 100 == 0:
            nc.eval()
            acc_1 = nc.evaluate(nodes_embedding[val_mask], nodes_label[val_mask])
            acc_2 = nc.evaluate(nodes_embedding[test_mask], nodes_label[test_mask])
            logging.info('Val Accuracy: {}'.format(acc_1.item()))
            logging.info('Test Accuracy: {}'.format(acc_2.item()))
            nc.train()


if __name__ == '__main__':
    main(parse_args())
