import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .GCNConv import GCNConv
import torch_geometric

class Decagon(nn.Module):

    def __init__(self, edge_index, relation_num, dim, init_emb, args):
        super(Decagon, self).__init__()
        self.args = args
        self.edge_index = edge_index
        self.relation_num = relation_num
        self.dim = dim
        self.drop = torch.nn.Dropout(self.args.decagon_drop)

        if args.use_feat:
            self.init_emb = nn.Parameter(init_emb, requires_grad=False)
        else:
            self.init_emb = nn.Parameter(torch.zeros(edge_index.max() + 1, self.dim), requires_grad=False)
            nn.init.xavier_uniform_(self.init_emb)

        self.conv1 = GCNConv(self.init_emb.shape[1], self.dim)
        self.conv2 = GCNConv(self.dim, self.dim)

        self.relation_param = nn.Parameter(torch.zeros(relation_num, self.dim, self.dim))
        self.interaction_model = nn.Parameter(torch.zeros(1, self.dim, self.dim))

        self.batchnorm = nn.BatchNorm1d(self.dim)

        nn.init.xavier_uniform_(self.relation_param)
        nn.init.xavier_uniform_(self.interaction_model)

        self.crossloss = torch.nn.CrossEntropyLoss()
        if self.args.dataset == 'twosides':
            self.bceloss = torch.nn.BCELoss(weight = torch.tensor(args.loss_weight))

    def loss(self, pred, true_label):
        # return self.bceloss(pred, true_label)
        if self.args.dataset == 'drugbank':
            return self.crossloss(pred, true_label)
        elif self.args.dataset == 'twosides':
            return self.bceloss(torch.sigmoid(pred)*true_label[:,:-1], true_label[:,:-1]*true_label[:,-1].unsqueeze(1))

    def forward(self, data):
        head, tail = data[0], data[1]
        x = self.conv1(self.init_emb, self.edge_index)
        x = self.drop(x)
        x = self.conv2(x, self.edge_index)
        x = self.drop(x)
        head_emb = torch.index_select(x, 0, head)
        tail_emb = torch.index_select(x, 0, tail)

        head_trans = torch.matmul(head_emb.unsqueeze(0), self.relation_param)
        tail_trans = torch.matmul(tail_emb.unsqueeze(0), self.relation_param)

        predict = torch.matmul(torch.matmul(head_trans, self.interaction_model).unsqueeze(-2), tail_trans.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        predict = predict.T
        return predict
