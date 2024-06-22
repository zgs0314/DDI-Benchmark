import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

from sklearn.cross_decomposition import PLSRegression

class CSMDDI(nn.Module):

    def __init__(self, entity_num, relation_num, dim, args, data_record, device):
        super(CSMDDI, self).__init__()

        n = entity_num
        d = dim
        K = relation_num

        self.device = device
        self.d = d

        self.E = nn.Parameter(torch.zeros(n, d)).cpu()
        self.M = nn.Parameter(torch.zeros(K, d, d)).cpu()

        nn.init.xavier_uniform_(self.M)

        self.train_set = torch.tensor(data_record.train_set).cpu()
        self.val_test_set = torch.tensor(data_record.val_test_set).cpu()

        nn.init.xavier_uniform_(self.E)

        # self.A = torch.tensor(data_record.adj_matrix).to(device)

        # self.feat = torch.tensor(data_record.feat).to(device)

        self.A = torch.tensor(data_record.adj_matrix).cpu()

        self.feat = torch.tensor(data_record.feat).cpu()

        # self.E_record = self.E.clone() ### consider the grad???
        self.E_record = nn.Parameter(torch.zeros(n, d), requires_grad=False).cpu()

    def forward_to(self):
        E = F.normalize(self.E).cpu()
        A_predict = E.matmul(self.M).matmul(E.transpose(0, 1)).cpu()

        # loss = torch.norm(self.A - A_predict) ** 2
        loss = torch.norm((self.A - A_predict)[:,self.train_set,:][:,:,self.train_set]) ** 2
        return loss
    
    def forward(self, data):
        # (self.E_record[data[0]].matmul(self.M)*(self.E_record[data[1]])).sum(-1)[0]
        pred = (self.E_record[data[0].cpu()].matmul(self.M)*(self.E_record[data[1].cpu()])).sum(-1).transpose(0,1)
        return pred

    def pre_process(self):
        self.predictor = PLSRegression(n_components=self.d)
        self.predictor.fit(self.feat[self.train_set].cpu().detach().numpy(), self.E[self.train_set].cpu().detach().numpy())
        pred_emb = torch.tensor(self.predictor.predict(self.feat[self.val_test_set].cpu().detach().numpy())).float()
        self.E_record[self.val_test_set] = pred_emb
        self.E_record[self.train_set] = self.E[self.train_set].clone().detach()