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
from   utils import *

class KGDDI_pretrain(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args):
        super(KGDDI_pretrain, self).__init__()
        '''
            KGEModel class
            components:
                - definition of KGE models 
                - train and test functions
        '''

        self.args = args

        self.device = args.device

        # build model
        self.model_name           = model_name
        self.nentity              = nentity

        if model_name == 'ComplEx':
            self.nrelation            = 2 * nrelation 
        else:
            self.nrelation            = nrelation 
        
        self.hidden_dim           = 200
        self.gamma                = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        self.embedding_range      = nn.Parameter(torch.Tensor([0.01]), requires_grad=False)

        self.entity_embedding     = nn.Parameter(torch.zeros(self.nentity, self.hidden_dim))
        self.relation_embedding   = nn.Parameter(torch.zeros(self.nrelation, self.hidden_dim))
        
        self.dropoutRate          = 0
        self.dropout              = nn.Dropout(p=self.dropoutRate)
        self.training_mode        = 'negativeSampling'
        self.label_smooth         = 0
        self.loss_name            = 'BCE_mean'
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
            'ComplEx':  self.ComplEx
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
        if mode == 'all':
            score = torch.mm(hr_vec, tail.transpose(0, 1))
        else:
            if len(relation.shape) == 3:
                score = torch.sum(hr_vec * tail.unsqueeze(0), dim=-1)
            else:
                score = torch.sum(hr_vec.unsqueeze(1) * tail, dim=-1)

        return score


    def forward(self, sample, mode='single'):
        '''
            3 available modes: 
                - single     : for calculating positive scores
                - neg_sample : for negative sampling
                - all        : for 1(k) vs All training 
        '''
        head_index, relation_index, tail_index = sample
        inv_relation_mask =  None
        relation_index    =  relation_index
        head              = self.dropout(self.entity_embedding[head_index])
        relation          = self.dropout(self.relation_embedding[relation_index])
        tail              = self.dropout(self.entity_embedding if mode == 'all' else self.entity_embedding[tail_index])

        score = self.model_func[self.model_name](head, relation, tail, inv_relation_mask=inv_relation_mask, mode=mode)
        
        return score

    def train_step(self, model, optimizer, data):
        # prepare
        optimizer.zero_grad()

        positive_sample, negative_sample, mode0 = data
        labels = - torch.ones((1, positive_sample.size(0))).to(self.args.device)

        labels             = labels.to(self.args.device)
        filter_mask        = filter_mask.to(self.args.device) if self.filter_falseNegative else None
        # forward
        positive_score = model((positive_sample[:,0], positive_sample[:,1], positive_sample[:,2].unsqueeze(1)), mode='single')     # [B, 1]
        if self.training_mode == 'negativeSampling':    
            negative_score = model((positive_sample[:,0], positive_sample[:,1], negative_sample), mode='neg_sample')               # [B, N_neg]
        else:
            all_score = model((positive_sample[:,0], positive_sample[:,1], negative_sample), mode='all')     # [B, N_neg]

        # Margin Ranking Loss (MR)
        if self.loss_name == 'MR':
            # only supporting training mode of negativeSampling
            target = torch.ones(positive_score.size()).to(self.args.device)
            loss   = self.MRLoss(positive_score, negative_score, target)
            loss   = (loss * filter_mask).mean(-1) if self.filter_falseNegative else loss.mean(-1)                                  # [B]
            loss   = loss.mean()          # [1]

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
                neg_loss  = neg_loss.mean(-1)                   # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean() 

            else:
                # label smoothing
                labels   = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss     = self.weightedBCELoss(all_score, labels).mean(dim=1)                                                     # [B]
                loss     = loss.mean() 

        elif self.loss_name == 'BCE_sum':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).to(self.args.device)
                neg_label = torch.zeros(negative_score.size()).to(self.args.device)

                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).sum(-1) if self.filter_falseNegative else neg_loss.sum(-1)                     # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean()    
            else: # 1vsAll or kvsAll
                # label smoothing
                labels = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss   = self.BCELoss(all_score, labels).sum(dim=1)                                                              # [B]
                loss   = loss.mean() 

        elif self.loss_name == 'BCE_adv':
            pos_loss = self.BCELoss(positive_score, torch.ones(positive_score.size()).to(self.args.device)).squeeze(-1)                          # [B]
            neg_loss = self.BCELoss(negative_score, torch.zeros(negative_score.size()).to(self.args.device))                                     # [B, N_neg]
            neg_loss = ( F.softmax(negative_score * self.adv_temperature, dim=1).detach() * neg_loss )

            if self.training_mode == 'negativeSampling' and self.filter_falseNegative:
                neg_loss  = (neg_loss * filter_mask).sum(-1) 
            else:
                neg_loss  =  neg_loss.sum(-1)                                                                                      # [B]

            loss     = pos_loss + neg_loss 
            loss     = loss.mean()                                  
        
        # Cross Entropy (CE)
        elif self.loss_name == 'CE':
            if self.training_mode == 'negativeSampling':
                # note that filter false negative samples is not supported here
                cat_score = torch.cat([positive_score, negative_score], dim=1)
                labels    = torch.zeros((positive_score.size(0))).long().to(self.args.device)
                loss      = self.CELoss(cat_score, labels)
                loss      = loss.mean() 

        loss.backward()
        # positive_score.mean().backward()
        optimizer.step()

        return loss

    def test_step(self, model, data, label):
        ### remember to add test_dataset
        ### consider the label?
        '''
        Evaluate the model on test or valid datasets
        '''

        all_ranking = []
        positive_sample, negative_sample, mode0 = data

        # forward
        score = model((positive_sample[:,0], positive_sample[:,1], None), 'all')
        score = score - torch.min(score, dim=1)[0].unsqueeze(1)

        mask = 1 - label
        mask[torch.arange(positive_sample[:,2].shape[0]).to(self.device), positive_sample[:,2]] = 1

        score = score * mask
        
        # explicitly sort all the entities to ensure that there is no test exposure bias
        argsort = torch.argsort(score, dim=1, descending=True)
        positive_arg = positive_sample[:, 2] # indexes of target entities
        
        # obtain rankings for the batch
        tmp_ranking = torch.nonzero(argsort == positive_arg.unsqueeze(1))[:, 1].cpu().numpy() + 1
        all_ranking += list(tmp_ranking)

        # calculate metrics
        all_ranking        = np.array(all_ranking)
        mrr = 1 / all_ranking

        return mrr
    
    def update_emb(self):
        self.best_emb = self.entity_embedding.clone().detach()
    
    def return_emb(self):
        return self.best_emb


