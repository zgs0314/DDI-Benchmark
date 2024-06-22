import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from torch.nn.parameter import Parameter
import math

    
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class SkipGNN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, rell, args):
        super(SkipGNN, self).__init__()

        self.args = args
        
        # original graph
        self.o_gc1 = GraphConvolution(nfeat, nhid)
        self.o_gc2 = GraphConvolution(nhid, nhid)
        
        # original graph for skip update
        self.o_gc1_s = GraphConvolution(nhid, nhid)
        
        #skip graph
        self.s_gc1 = GraphConvolution(nfeat, nhid)
        
        #skip graph for original update
        self.s_gc1_o = GraphConvolution(nfeat, nhid)
        self.s_gc2_o = GraphConvolution(nhid, nhid)
       
        self.dropout = dropout
        
        self.decoder1 = nn.Linear(nhid * 2, nhid)
        self.decoder2 = nn.Linear(nhid, rell)

        if self.args.dataset == 'drugbank':
            self.bceloss	= torch.nn.BCELoss()
        elif self.args.dataset == 'twosides':
            # self.bceloss	= torch.nn.BCELoss()
            self.bceloss	= torch.nn.BCELoss(weight = torch.tensor(args.loss_weight))

    def loss(self, pred, true_label):
        # return self.bceloss(pred, true_label)
        if self.args.dataset == 'drugbank':
            return self.bceloss(torch.softmax(pred,1), true_label)
        elif self.args.dataset == 'twosides':
            return self.bceloss(torch.sigmoid(pred)*true_label[:,:-1], true_label[:,:-1]*true_label[:,-1].unsqueeze(1))

    def forward(self, data):
        x, o_adj, s_adj, idx = data
        
        o_x = F.relu(self.o_gc1(x, o_adj) + self.s_gc1_o(x, s_adj))       
        s_x = F.relu(self.s_gc1(x, s_adj) + self.o_gc1_s(o_x, o_adj))
        
        o_x = F.dropout(o_x, self.dropout, training = self.training)
        s_x = F.dropout(s_x, self.dropout, training = self.training)
        
        x = self.o_gc2(o_x, o_adj) + self.s_gc2_o(s_x, s_adj)
        
        feat_p1 = x[idx[0]] # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]] # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim = 1)
        o = self.decoder1(feat)
        o = self.decoder2(o)
        return o