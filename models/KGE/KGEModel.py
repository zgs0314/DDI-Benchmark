import time
import os
import random
import itertools
import logging
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import DataLoader
from   collections import defaultdict
from   tqdm import tqdm
from   .dataloader import TestDataset
from   utils import *

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args):
        super(KGEModel, self).__init__()
        '''
            KGEModel class
            components:
                - definition of KGE models 
                - train and test functions
        '''
        # checking parameters

        # self.device = torch.device('cuda')
        self.args = args

        if self.args.dataset == 'twosides':
            self.loss_weight = torch.tensor(self.args.loss_weight,requires_grad=False).to(args.device)

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'MSTE']:
            raise ValueError('model %s not supported' % model_name)
        
        if model_name == 'MSTE':
            self.softplus = torch.nn.Softplus()

        # build model
        self.model_name           = model_name
        self.nentity              = nentity
        # self.nrelation            = nrelation if config.training_strategy.shareInverseRelation else 2*nrelation
        if model_name == 'ComplEx':
            self.nrelation            = 2 * nrelation 
        else:
            self.nrelation            = nrelation 
        self.hidden_dim           = args.kge_dim
        self.gamma                = nn.Parameter(torch.Tensor([args.kge_gamma]), requires_grad=False)
        # self.embedding_range      = nn.Parameter(torch.Tensor([params_dict['embedding_range']]), requires_grad=False)
        self.embedding_range      = nn.Parameter(torch.Tensor([0.01]), requires_grad=False)

        # set relation dimension according to specific model
        self.relation_dim = self.hidden_dim

        self.entity_embedding     = nn.Parameter(torch.zeros(self.nentity, self.hidden_dim))
        if model_name == 'MSTE' and self.args.dataset == 'twosides':
            self.relation_embedding   = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim, 2))
        else:
            self.relation_embedding   = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
        
        # read essential training config (from global_config)
        self.dropoutRate          = args.kge_dropout
        self.dropout              = nn.Dropout(p=self.dropoutRate)
        self.training_mode        = 'negativeSampling'
        self.label_smooth         = 0
        self.loss_name            = args.kge_loss
        self.uni_weight           = 1
        self.filter_falseNegative = False

        # setup candidate loss functions
        self.KLLoss               = nn.KLDivLoss(size_average=False)
        self.MRLoss               = nn.MarginRankingLoss(margin=float(self.gamma), reduction='none')
        self.CELoss               = nn.CrossEntropyLoss(reduction='none')
        self.BCELoss              = nn.BCEWithLogitsLoss(reduction='none')
        self.weightedBCELoss      = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([self.nentity]))

        # initialize embedding
        self.init_embedding('uniform') ### choice: ['uniform','xavier_norm', 'xavier_uniform', 'normal']
        self.model_func = {
            'TransE':   self.TransE,
            'DistMult': self.DistMult,
            'ComplEx':  self.ComplEx,
            'MSTE':  self.MSTE
        }
        
    def init_embedding(self, init_method):
        if init_method == 'uniform':
            # Fills the input Tensor with values drawn from the uniform distribution
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item() )
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item() )
        
        elif init_method == 'xavier_norm':
            nn.init.xavier_normal_(tensor=self.entity_embedding)
            nn.init.xavier_normal_(tensor=self.relation_embedding)

        elif init_method == 'normal':
            # Fills the input Tensor with values drawn from the normal distribution
            nn.init.normal_(tensor=self.entity_embedding, mean=0.0, std=self.embedding_range.item())
            nn.init.normal_(tensor=self.relation_embedding, mean=0.0, std=self.embedding_range.item())

        elif init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor=self.entity_embedding)
            nn.init.xavier_uniform_(tensor=self.relation_embedding)

        return

    # def forward(self, sample, mode='single'):
    def forward(self, sample):
        '''
            3 available modes: 
                - single     : for calculating positive scores
                - neg_sample : for negative sampling
                - all        : for 1(k) vs All training 
        '''
        pos_data, neg_data, mode = sample
        inv_relation_mask =  None
        # relation_index    = relation_index
        head = self.dropout(self.entity_embedding[pos_data[:,0]])
        tail = self.dropout(self.entity_embedding[pos_data[:,1]])
        if 'valid' in mode or 'test' in mode:
            if self.model_name == 'ComplEx':
                pos_relation = self.dropout(self.relation_embedding[torch.tensor([i for i in range(int(self.nrelation/2))]).repeat(pos_data.shape[0],1).to(self.args.device)])
            else:
                pos_relation = self.dropout(self.relation_embedding[torch.tensor([i for i in range(int(self.nrelation))]).repeat(pos_data.shape[0],1).to(self.args.device)])
        else:
            pos_relation = self.dropout(self.relation_embedding[pos_data[:,2]])

        pos_score = self.model_func[self.model_name](head, pos_relation, tail, inv_relation_mask=inv_relation_mask, mode=mode)
        if 'valid' in mode or 'test' in mode:
            pass
        else:
            if self.args.dataset == 'twosides':
                neg_head = self.entity_embedding[neg_data[:,0]]
                neg_tail = self.entity_embedding[neg_data[:,1]]
                neg_score = self.model_func[self.model_name](neg_head, pos_relation, neg_tail, inv_relation_mask=inv_relation_mask, mode=mode)
                self.relation_cun = pos_data[:,2]
            else:
                neg_relation   =   self.dropout(self.relation_embedding[neg_data])
                neg_score = self.model_func[self.model_name](head, neg_relation, tail, inv_relation_mask=inv_relation_mask, mode=mode)

        if 'valid' in mode or 'test' in mode:
            if self.model_name == 'ComplEx':
                return pos_score.transpose(0,1)
            else:
                    return pos_score
        else:
            return [pos_score, neg_score]


    def TransE(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
            (h,r,t):     h + r = t
            (t,INV_r,h): t + (-r) = h, INV_r = -r
            ori: score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        '''

        if mode == 'all':
            score = (head + relation).unsqueeze(1) - tail.unsqueeze(0)
        else:
            score = (head + relation).unsqueeze(1) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score

    def DistMult(self, head, relation, tail, inv_relation_mask, mode='single'):
        if mode == 'all':
            score = torch.mm(head * relation, tail.transpose(0,1))
        else:
            score = torch.sum((head * relation).unsqueeze(1) * tail, dim=-1)

        return score
    def ComplEx(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
        INV_r = Conj(r)
        '''
        re_head, im_head = torch.chunk(head, 2, dim=-1)

        # if mode == 'neg_sample':
        if len(relation.shape) == 3:
            relation = relation.transpose(1,0)

        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)

        re_hrvec = re_head * re_relation - im_head * im_relation
        im_hrvec = re_head * im_relation + im_head * re_relation
        hr_vec   = torch.cat([re_hrvec, im_hrvec], dim=-1)

        # regularization on positive samples
        if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], hr_vec)

        # <φ(h,r), t> -> score
        if mode == 'all':
            score = torch.mm(hr_vec, tail.transpose(0, 1))
        else:
            if len(relation.shape) == 3:
                score = torch.sum(hr_vec * tail.unsqueeze(0), dim=-1)
            else:
                score = torch.sum(hr_vec.unsqueeze(1) * tail, dim=-1)

        return score

    def MSTE(self, head, relation, tail, inv_relation_mask, mode='single'):
        if self.args.dataset == 'twosides':
            score = 0
            head = head.unsqueeze(2)
            tail = tail.unsqueeze(2)
            if len(relation.shape) == 4:
                if relation.shape[1] == 1:
                    relation = relation.squeeze(1)
                else:
                    head = head.unsqueeze(1)
                    tail = tail.unsqueeze(1)
            head_proj = torch.sin(tail * relation) * head
            trlation_proj = torch.sin(tail * head) * relation
            tail_proj = torch.sin(head * relation) * tail
            score = torch.norm(head_proj + trlation_proj - tail_proj, dim=-2)
            if len(score.shape) == 2:
                score = score[:,1] - score[:,0]
            else:
                score = score[:,:,1] - score[:,:,0]
        else:
            score = 0
            if len(relation.shape) == 3:
                if relation.shape[1] == 1:
                    relation = relation.squeeze(1)
                else:
                    head = head.unsqueeze(1)
                    tail = tail.unsqueeze(1)
            head_proj = torch.sin(tail * relation) * head
            trlation_proj = torch.sin(tail * head) * relation
            tail_proj = torch.sin(head * relation) * tail
            score = torch.norm(head_proj + trlation_proj - tail_proj, dim=-1)
        return score

    def loss(self, pred, true_label):
        # return self.bceloss(pred, true_label)
        positive_score, negative_score = pred
        subsampling_weight = 0 ### do not use
        filter_mask = 0 ### do not use
        all_score = 0 ### do not use

        if self.model_name == 'MSTE':
            if len(negative_score.shape) == 2:
                negative_score = negative_score.mean(1)
            if self.args.dataset == 'twosides':
                loss = self.softplus((self.gamma + negative_score - positive_score)*(self.loss_weight[self.relation_cun.int().tolist()])).mean()
            else:
                loss = self.softplus((self.gamma + negative_score - positive_score)).mean()
            return loss

        if self.loss_name == 'MR':
            # only supporting training mode of negativeSampling
            target = torch.ones(positive_score.size()).to(self.args.device)
            loss   = self.MRLoss(positive_score, negative_score, target)
            loss   = (loss * filter_mask).mean(-1) if self.filter_falseNegative else loss.mean(-1)                                  # [B]
            loss   = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())             # [1]

        # Binary Cross Entropy Loss (BCE) 
        elif self.loss_name == 'BCE_mean':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).to(self.args.device)
                neg_label = torch.zeros(negative_score.size()).to(self.args.device)
                
                # label smoothing
                pos_label = (1.0 - self.label_smooth)*pos_label + (1.0/self.nentity) if self.label_smooth > 0 else pos_label
                neg_label = (1.0 - self.label_smooth)*neg_label + (1.0/self.nentity) if self.label_smooth > 0 else neg_label
                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).mean(-1) if self.filter_falseNegative else neg_loss.mean(-1)                   # [B]
                loss      = pos_loss.mean() + neg_loss.mean()

            else:
                # label smoothing
                labels   = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss     = self.weightedBCELoss(all_score, labels).mean(dim=1)                                                     # [B]
                loss     = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())

        elif self.loss_name == 'BCE_sum':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).to(self.args.device)
                neg_label = torch.zeros(negative_score.size()).to(self.args.device)
                
                # label smoothing
                pos_label = (1.0 - self.label_smooth)*pos_label + (1.0/self.nentity) if self.label_smooth > 0 else pos_label
                neg_label = (1.0 - self.label_smooth)*neg_label + (1.0/self.nentity) if self.label_smooth > 0 else neg_label
                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).sum(-1) if self.filter_falseNegative else neg_loss.sum(-1)                     # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())   
            else: # 1vsAll or kvsAll
                # label smoothing
                labels = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss   = self.BCELoss(all_score, labels).sum(dim=1)                                                              # [B]
                loss   = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())
        # Cross Entropy (CE)
        elif self.loss_name == 'CE':
            if self.training_mode == 'negativeSampling':
                # note that filter false negative samples is not supported here
                cat_score = torch.cat([positive_score, negative_score], dim=1)
                labels    = torch.zeros((positive_score.size(0))).long().to(self.args.device)
                loss      = self.CELoss(cat_score, labels)
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())
 
        return loss
 

    def saveEmbeddingToFile(self, savePath):
        saveData = {}
        saveData['entity_embedding']   = self.entity_embedding.cpu()
        saveData['relation_embedding'] = self.relation_embedding.cpu()
        logging.info(f'save embedding tensor to: {savePath}')
        pkl.dump(saveData, open(savePath, "wb" ))
        return

    def loadEmbeddingFromFile(self, savePath):
        if not os.path.exists(savePath):
            logging.info(f'[Error] embedding file does not exist: {savePath}')
            return
        data = savePickleReader(savePath)
        self.entity_embedding   = nn.Parameter(data['entity_embedding'])
        self.relation_embedding = nn.Parameter(data['relation_embedding'])
        logging.info(f'successfully loaded pretrained embedding from: {savePath}')
        return

