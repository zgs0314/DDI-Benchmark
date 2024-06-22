import os
import numpy as np
from models.CompGCN import *
from models.SkipGNN import *
from models.KGE import *
from models.MLP import *
from models.KGDDI import *
from models.CSMDDI import *
from models.HINDDI import *
from models.Decagon import *
import json
import sys
import torch
import time

import random

import fcntl
import pickle as pkl

from collections import defaultdict as ddict

from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
import torch.optim as optim

def load_data(args):
    if 'drugbank' in args.dataset:
        if args.DDIsetting == 'S0':
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[]}
            sets = ['train', 'valid_S0', 'test_S0']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
        elif args.DDIsetting == 'S1':
            triple_dict = {'train':[], 'valid_S1':[], 'test_S1':[]}
            sets = ['train', 'valid_S1', 'test_S1']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
        elif args.DDIsetting == 'S2':
            triple_dict = {'train':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S2', 'test_S2']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
        else:
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[], 'valid_S1':[], 'test_S1':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    str_lin = j.strip().split(' ')
                    triple_dict[i].append([int(j) for j in str_lin])
                file.close()
    elif 'twosides' in args.dataset:
        if args.DDIsetting == 'S0':
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[]}
            sets = ['train', 'valid_S0', 'test_S0']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
        elif args.DDIsetting == 'S1':
            triple_dict = {'train':[], 'valid_S1':[], 'test_S1':[]}
            sets = ['train', 'valid_S1', 'test_S1']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
        elif args.DDIsetting == 'S2':
            triple_dict = {'train':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S2', 'test_S2']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
        else:
            triple_dict = {'train':[], 'valid_S0':[], 'test_S0':[], 'valid_S1':[], 'test_S1':[], 'valid_S2':[], 'test_S2':[]}
            sets = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']
            for i in sets:
                file = open('./data/{}/{}.txt'.format(args.dataset, i))
                for j in file:
                    h,t,r,p = j[:-1].split(' ')
                    r_0 = r.split(',')
                    list_cun = [int(j) for j in r_0]
                    list_cun.append(int(p))
                    tuple_cun = tuple(list_cun)
                    triple_dict[i].append([int(h), int(t), tuple_cun])
                file.close()
    return triple_dict

def load_feature(args):
    feat = 0
    if 'drugbank' in args.dataset:
        with open('data/drugbank/DB_molecular_feats.pkl', 'rb') as f:
            x = pkl.load(f, encoding='utf-8')
        feat = []
        for y in x['Morgan_Features']:
            feat.append(y)
    if 'twosides' in args.dataset:
        with open('data/twosides/DB_molecular_feats.pkl', 'rb') as f:
            feat = pkl.load(f, encoding='utf-8')
    return feat

def add_model(args, data_record, device):
    model = 0
    if args.model == 'CompGCN':
        if args.Comp_sfunc == 'TransE':
            model = CompGCN_TransE_DDI(data_record.edge_index, data_record.edge_type, args, data_record.feat)
        model.to(device)
    elif args.model == 'SkipGNN':
        model = SkipGNN(data_record.feat.shape[1], args.skip_hidden, args.skip_dropout, data_record.num_rel, args).to(device)
    elif args.model in ['ComplEx', 'MSTE']:
        if args.model == 'ComplEx':
            model = KGEModel('ComplEx', data_record.num_ent, data_record.num_rel, args).to(device)
        elif args.model == 'MSTE':
            model = KGEModel('MSTE', data_record.num_ent, data_record.num_rel, args).to(device)
    elif args.model == 'MLP':
        model = MLP(data_record.num_ent, data_record.num_rel, args.mlp_dim, args, data_record.feat).to(device)
    elif args.model == 'KGDDI':
        model = KGDDI_MLP(data_record.num_ent, data_record.num_rel, args.kgddi_dim, args, data_record.feat).to(device)
    elif args.model == 'CSMDDI':
        model = CSMDDI(data_record.num_ent, data_record.num_rel, args.csm_dim, args, data_record, device)
    elif args.model == 'HINDDI':
        model = HIN_MLP(data_record.num_ent, data_record.num_rel, args.hin_dim, args, data_record.meta_feature).to(device)
    elif args.model == 'Decagon':
        model = Decagon(data_record.edge_index, data_record.num_rel, args.decagon_dim, data_record.feat, args).to(device)
    return model

def read_batch(batch, split, device, args, data_record = None):
    if args.model in ['CompGCN', 'MLP', 'CSMDDI', 'HINDDI', 'Decagon']:
        if split == 'train':
            triple, label = [ _.to(device) for _ in batch]
            return [triple[:, 0], triple[:, 1], triple[:, 2]], label
        else:
            triple, label = [ _.to(device) for _ in batch]
            return [triple[:, 0], triple[:, 1], triple[:, 2]], label
    elif args.model == 'SkipGNN':
        triple, label = [ _.to(device) for _ in batch]
        return [data_record.feat.to(device), data_record.adj.to(device), data_record.adj2.to(device), [triple[:, 0], triple[:, 1]]], label
    elif args.model in ['ComplEx', 'MSTE']:
        triple, label = [ _.to(device) for _ in batch]
        if args.dataset == 'drugbank':
            num_rel = data_record.num_rel
            neg_data = []
            samp_set_0 = [i for i in range(num_rel)]
            for j in triple:
                samp_set = list(set(samp_set_0) - set([j[2].item()]))
                n_neg = 1 if args.model == 'MSTE' else 16
                neg_data.append(random.sample(samp_set, n_neg))
            neg_data = torch.LongTensor(neg_data).to(device)
            return [triple, neg_data, split], label
        elif args.dataset == 'twosides':
            if split == 'train':
                return [triple[:,:3], triple[:,3:], split], label
            else:
                return [triple, 0, split], label
    elif args.model in ['KGDDI']:
        if args.KGDDI_pre == 1:
            triple, label = [ _.to(device) for _ in batch]
            num_ent = args.num_bnent
            neg_data = []
            samp_set_0 = [i for i in range(num_ent)]
            for j in triple:
                samp_set = list(set(samp_set_0) - set([j[2].item()]))
                n_neg = 16
                neg_data.append(random.sample(samp_set, n_neg))
            neg_data = torch.LongTensor(neg_data).to(device)
            return [triple, neg_data, split], label
        else:
            if split == 'train':
                triple, label = [ _.to(device) for _ in batch]
                return [triple[:, 0], triple[:, 1], triple[:, 2]], label
            else:
                triple, label = [ _.to(device) for _ in batch]
                return [triple[:, 0], triple[:, 1], triple[:, 2]], label

### for SkipGNN
import scipy.sparse as sp

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def savePickleReader(file):
    if os.path.exists(file):
        while True:
            try:
                with open(file, "rb") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    unpickler = pkl.Unpickler(f)
                    data = unpickler.load()
                    f.close()
                    break
            except:
                continue
        return data
    else:
        return None

### for KG-DDI
def pretrain_bn(args, device, file_name):
    ### load and set data
    if args.bionet == 'HetioNet':
        BN_path = 'relations_2hop'
    elif args.bionet == 'PrimeKG':
        BN_path = 'BN_Primekg'
    triple_bn = []
    file = open('./data/{}/{}.txt'.format(args.dataset, BN_path)) ###relations_2hop/relations_2hop_small
    for j in file:
        str_lin = j.strip().split(' ')
        triple_bn.append([int(j) for j in str_lin])
    triple_bn_np = np.array(triple_bn)[:,[0,2,1]] ### swap the second and third column
    num_bn = len(triple_bn)
    list_all = [j for j in range(num_bn)]
    train_num = int(num_bn / 10 * 9.9)
    train_set = random.sample(list_all, int(train_num))
    list_all = list(set(list_all) - set(train_set))
    valid_set = list_all

    num_bnent = triple_bn_np.max() + 1
    num_bnrel = triple_bn_np[:, 1].max() + 1

    train_triplets = triple_bn_np[train_set]
    valid_triplets = triple_bn_np[valid_set]

    data = ddict(list)
    sr2o = ddict(set)

    for j in train_triplets:
        sub, rel, obj = j[0], j[1], j[2]
        data['train'].append((sub, rel, obj))
        sr2o[(sub, rel)].add(obj)
    
    sr2o_train = {k: list(v) for k, v in sr2o.items()}

    for j in valid_triplets:
        sub, rel, obj = j[0], j[1], j[2]
        data['valid'].append((sub, rel, obj))
        sr2o[(sub, rel)].add(obj)

    sr2o_all = {k: list(v) for k, v in sr2o.items()}

    triples  = ddict(list)

    for (sub, rel), obj in sr2o_train.items():
        triples['train'].append({'triple':(sub, rel, -1), 'label': sr2o_train[(sub, rel)], 'sub_samp': 1})

    for sub, obj, rel in data['valid']:
        triples['valid'].append({'triple': (sub, obj, rel), 	   'label': sr2o_all[(sub, obj)]})
    
    triples = dict(triples)

    data_iter_train = get_data_loader(triples, args, KGDDI_TrainDataset, 'train', 512)
    data_iter_valid = get_data_loader(triples, args, KGDDI_TestDataset, 'valid', 512)

    ### begin the training step
    ### model name: 'TransE', 'ComplEx'
    pre_model = KGDDI_pretrain('ComplEx', num_bnent, num_bnrel, args).to(device)
    pre_optimizer = optim.AdamW(pre_model.parameters(), lr=0.0001, weight_decay=args.weight_decay)

    best_val_mrr = 0
    mrr_change = 0

    args.num_bnent = num_bnent
    args.KGDDI_pre = 1

    for epoch in range(20):
        train_iter = iter(data_iter_train)
        pre_model.train()
        loss_list = []
        for step, batch in enumerate(train_iter):
            data, label = read_batch(batch, 'train', device, args)
            loss = pre_model.train_step(pre_model, pre_optimizer, data)
            loss_list.append(loss.item())
        print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, Loss: {}'.format(epoch, np.mean(np.array(loss_list))))
        with open(os.path.join('record', file_name), 'a+') as f:
            f.write(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, Loss: {}'.format(epoch, np.mean(np.array(loss_list))))
        pre_model.eval()
        valid_iter = iter(data_iter_valid)
        mrr_list = []
        for step, batch in enumerate(valid_iter):
            data, label = read_batch(batch, 'valid', device, args)
            mrr = pre_model.test_step(pre_model, data, label)
            mrr_list.append(mrr)
        mrr_final = np.concatenate(mrr_list).mean()
        print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, MRR: {}'.format(epoch, mrr_final))
        with open(os.path.join('record', file_name), 'a+') as f:
            f.write(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + '| Pretraining Epoch: {}, MRR: {}'.format(epoch, mrr_final))
        if mrr_final > best_val_mrr:
            best_val_mrr = mrr_final
            pre_model.update_emb()
    emb_final = pre_model.return_emb()
    return emb_final

def get_data_loader(triples, args, dataset_class, split, batch_size, shuffle=True):
    return  DataLoader(
        dataset_class(triples[split], args),
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = 10, ### set the default numworkers to 10
        collate_fn      = dataset_class.collate_fn
    )

class KGDDI_TrainDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		label, sub_samp	= np.array(ele['label']).astype(int), np.float32(ele['sub_samp'])
		trp_label		= self.get_label_ddi(label)
		triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][0]])

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_bnent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

class KGDDI_TestDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int)
		label		= self.get_label_ddi(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label

	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_bnent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

### for CSMDDI
def train_wo_batch(model, optimizer, device, args, data_record):
    loss = model.forward_to()
    loss.backward()
    optimizer.step()
    return loss