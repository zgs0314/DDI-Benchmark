import os
from torch.utils.data import Dataset
from utils import *
from ordered_set import OrderedSet
from collections import defaultdict as ddict
import torch
from torch.utils.data import DataLoader 
import scipy.sparse as sp
from itertools import combinations

num_ent = {'drugbank': 1710, 'twosides': 645, 'HetioNet': 34124}
num_rel = {'drugbank': 86, 'twosides': 209} # 209, 309

class Data_record():
    def __init__(self, args, emb_return):
        self.args = args

        if args.bionet == 'HetioNet':
            self.BN_path = 'relations_2hop'
        elif args.bionet == 'PrimeKG':
            self.BN_path = 'BN_Primekg'

        self.link_aug_num = 0

        self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
        self.triplets = load_data(args)

        self.data = ddict(list)
        sr2o = ddict(set)

        self.link_aug_num = 0
        if args.data_aug:
            self.selected_link = [1,5,6,7,10,15,18,21]
            self.link_aug_num = len(self.selected_link)
            trip_aug = []
            file = open('./data/{}/{}.txt'.format(args.dataset, self.BN_path))
            for j in file:
                str_lin = j.strip().split(' ')
                trip = [int(j) for j in str_lin]
                if trip[2] in self.selected_link:
                    trip_aug.append([trip[0], trip[1], trip[2]])
            aug_num_begin = num_ent[args.dataset]
            aug_ent_in = np.unique(np.array(trip_aug).flatten())
            ind_dict = {}
            rel_dict = {self.selected_link[j]:num_rel[args.dataset] + j for j in range(len(self.selected_link))}
            for j in aug_ent_in:
                if j >= num_ent[args.dataset]:
                    ind_dict[j] = aug_num_begin
                    aug_num_begin += 1
                else:
                    ind_dict[j] = j
            for j in trip_aug:
                j = [ind_dict[j[0]], ind_dict[j[1]], rel_dict[j[2]]]
                self.triplets['train'].append(j)

        if args.use_reverse_edge:
            self.num_rel, self.args.num_rel = (num_rel[args.dataset] + self.link_aug_num) * 2, (num_rel[args.dataset] + self.link_aug_num) * 2
        else:
            self.num_rel, self.args.num_rel = num_rel[args.dataset] + self.link_aug_num, num_rel[args.dataset] + self.link_aug_num
        
        if args.data_aug:
            self.num_ent, self.args.num_ent = aug_num_begin, aug_num_begin ### 31130
        else:
            self.num_ent, self.args.num_ent = num_ent[args.dataset], num_ent[args.dataset]

        self.include_splits = list(self.triplets.keys())
        self.split_not_train = [j for j in self.include_splits if j != 'train']
        
        if args.use_reverse_edge:
            index_plus = int(self.args.num_rel/2)
        
        for split in self.include_splits:
            if split == 'train' and self.args.model in ['ComplEx', 'MSTE'] and args.dataset == 'twosides': 
                for j in range(int(len(self.triplets[split])/2)):
                    sub, obj, rel, neg_add = self.triplets[split][j*2][0], self.triplets[split][j*2][1], np.where(np.array(self.triplets[split][j*2][2])[:-1]==1)[0], [self.triplets[split][j*2+1][0], self.triplets[split][j*2+1][1]]
                    for k in rel:
                        self.data[split].append((sub, obj, [neg_add[0], neg_add[1], k]))
                        sr2o[(sub, obj)].add((neg_add[0], neg_add[1], k))
            else:
                for j in self.triplets[split]:
                    sub, obj, rel = j[0], j[1], j[2]
                    self.data[split].append((sub, obj, rel))

                    if split == 'train': 
                        if self.args.model in ['CompGCN', 'SkipGNN', 'Decagon'] and args.dataset == 'twosides':
                            self.true_data = self.data[split]
                        sr2o[(sub, obj)].add(rel)
                        if args.use_reverse_edge:
                            # sr2o[(obj, sub)].add(rel+index_plus)
                            self.data[split].append((obj,sub , rel+index_plus))
        
        if args.use_feat or args.model == 'CSMDDI':
            self.feat = torch.FloatTensor(np.array(load_feature(args))).to(self.device)
            self.feat_dim = self.feat.shape[1]
        else:
            if args.model == 'SkipGNN':
                features = np.eye(self.num_ent)
                self.feat = normalize(features)
                self.feat = torch.FloatTensor(features)
            elif args.model == 'KGDDI':
                self.feat = emb_return[:num_ent[args.dataset]]
            else:
                self.feat = 0
        
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.data = dict(self.data)

        for split in self.split_not_train:
            for sub, obj, rel in self.data[split]:
                sr2o[(sub, obj)].add(rel)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        ### train triples
        if self.args.dataset == 'twosides' and self.args.model in ['ComplEx', 'MSTE']:
            for sub, obj, rel in self.data['train']:
                self.triples['train'].append({'triple':(sub, obj, -1), 'label': rel, 'sub_samp': 1})
        else:
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        ### valid & test triplets
        for split in self.split_not_train:
            for sub, obj, rel  in self.data[split]:
                self.triples[split].append({'triple': (sub, obj, rel), 	   'label': self.sr2o_all[(sub, obj)]})

        self.triples = dict(self.triples)

        if args.model == 'CompGCN':
            self.p = args
            self.p.embed_dim	= self.p.comp_dim

            ### remark: see whether we need the reverse links
			
            self.edge_index, self.edge_type = [], []
            for sub, obj, rel in self.data['train']:
                if args.dataset == 'twosides' and rel[-1] == 0:
                    continue
                self.edge_index.append((sub, obj))
                self.edge_type.append(rel)

            self.edge_index	= torch.LongTensor(self.edge_index).to(self.device).t()
            if args.dataset == 'drugbank':
                self.edge_type	= torch.LongTensor(self.edge_type).to(self.device)
            elif args.dataset == 'twosides':
                self.edge_type	= torch.LongTensor(self.edge_type)[:,:-1].to(self.device)

            if args.data_aug and args.use_feat:
                feat = torch.zeros((self.num_ent, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat

        elif args.model == 'SkipGNN':
            if args.dataset == 'drugbank':
                self.num_skipnode = np.array(self.triplets['train']).max() + 1
                self.link = np.array(self.triplets['train'])[:,:2]
            elif args.dataset == 'twosides':
                self.num_skipnode = self.num_ent
                self.link = np.array([[j[0], j[1]] for j in self.triplets['train'] if j[2][-1] == 1])
            adj = sp.coo_matrix((np.ones(self.link.shape[0]), (self.link[:, 0], self.link[:, 1])),
                    shape=(self.num_skipnode, self.num_skipnode),
                    dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            #create skip graph
            adj2 = adj.dot(adj)
            adj2 = adj2.sign()
            adj2 = normalize_adj(adj2)
            self.adj2 = sparse_mx_to_torch_sparse_tensor(adj2)
            adj = adj + sp.eye(adj.shape[0])
            #normalize original graph
            adj = normalize_adj(adj)
            self.adj = torch.FloatTensor(np.array(adj.todense()))

            if args.data_aug and args.use_feat:
                feat = torch.zeros((self.num_ent, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat

        elif args.model == 'CSMDDI':
            self.adj_matrix = np.zeros((num_rel[args.dataset], num_ent[args.dataset], num_ent[args.dataset]))
            np_tri = np.array(self.triplets['train'])
            if args.dataset == 'drugbank':
                self.adj_matrix[np_tri[:,2], np_tri[:,0], np_tri[:,1]] += 1
                self.adj_matrix[np_tri[:,2], np_tri[:,1], np_tri[:,0]] += 1
            elif args.dataset == 'twosides':
                for j in np_tri:
                    self.adj_matrix[:, j[0], j[1]] = np.array(np_tri[0][2]).astype('float64')[:-1]
                    self.adj_matrix[:, j[1], j[0]] = np.array(np_tri[0][2]).astype('float64')[:-1]
            self.train_set = []
            file = open('./data/{}/train_set.txt'.format(args.dataset))
            for j in file:
                self.train_set.append(int(j.strip()))
            self.val_test_set = list(set(range(num_ent[args.dataset])) - set(self.train_set))
            x = 0
        elif args.model == 'HINDDI':
            flag = 0
            if args.with_1hop:
                pathh = './data/{}/metapath/meta_with.npy'.format(args.dataset)
                if not os.path.exists('./data/{}/metapath/meta_with.npy'.format(args.dataset)):
                    flag = 1
            else:
                pathh = './data/{}/metapath/meta_without.npy'.format(args.dataset)
                if not os.path.exists('./data/{}/metapath/meta_without.npy'.format(args.dataset)):
                    flag = 1
            if flag:
                hop_triple = []
                selected_link = [0,1,2,5,6,7,8,9,10,13,14,15,16,17,18,19,21]
                file = open('./data/{}/{}.txt'.format(args.dataset, self.BN_path))
                for j in file:
                    str_lin = j.strip().split(' ')
                    trip = [int(j) for j in str_lin]
                    if trip[2] in selected_link:
                        hop_triple.append(trip)
                if args.with_1hop:
                    for j in self.triplets['train']:
                        hop_triple.append(j)
                if args.dataset == 'twosides':
                    entity_type = [0 for j in range(35964)]
                elif args.dataset == 'drugbank':
                    entity_type = [0 for j in range(num_ent[args.bionet])]
                for i in range(num_ent[args.dataset]):
                    entity_type[i] = 'drug'
                if args.dataset == 'twosides':
                    with open('./data/{}/allname2id.json'.format(args.dataset), 'r') as file:
                        data = json.load(file)
                elif args.dataset == 'drugbank':
                    with open('./data/{}/entity_drug.json'.format(args.dataset), 'r') as file:
                        data = json.load(file)
                for j in data:
                    if 'Gene' in j:
                        entity_type[data[j]] = 'gene'
                    elif 'Anatomy' in j:
                        entity_type[data[j]] = 'anatomy'
                    elif 'Disease' in j:
                        entity_type[data[j]] = 'disease'
                    elif 'Symptom' in j:
                        entity_type[data[j]] = 'symptom'
                    elif 'Compound' in j or 'CID' in j:
                        entity_type[data[j]] = 'drug'
                    elif 'Pathway' in j:
                        entity_type[data[j]] = 'pathway'
                    elif 'Molecular Function' in j:
                        entity_type[data[j]] = 'molecular function'
                    elif 'Cellular Component' in j:
                        entity_type[data[j]] = 'cellular component'
                    elif 'Side Effect' in j:
                        entity_type[data[j]] = 'side effect'
                    elif 'Pharmacologic Class' in j:
                        entity_type[data[j]] = 'pharmacologic class'
                entity_class_no = ['pathway', 'symptom', 'anatomy', 'cellular component', 'molecular function']
                link_record = {}
                for j in hop_triple:
                    if entity_type[j[0]] in entity_class_no or entity_type[j[1]] in entity_class_no:
                        continue
                    if j[0] in link_record:
                        link_record[j[0]].append(j[1])
                    else:
                        link_record[j[0]] = [j[1]]
                    if j[1] in link_record:
                        link_record[j[1]].append(j[0])
                    else:
                        link_record[j[1]] = [j[0]]
                meta_matrix = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 14))
                meta_path_dict = {'drug':1, 'disease': 2, 'gene': 3, 'pharmacologic class': 4, 'side effect': 5,
                                'drug_drug': 6, 'drug_disease': 7, 'drug_gene': 8, 'drug_pharmacologic class': 9, 'drug_side effect': 10, 'gene_gene': 11, 'gene_disease': 12, 'disease_disease': 13, 
                                'disease_drug': 7, 'gene_drug': 8, 'pharmacologic class_drug': 9, 'side effect_drug': 10, 'disease_gene': 12}
                for i in range(num_ent[args.dataset]):
                    if i in link_record: ### begin
                        for j in link_record[i]:
                            if j < num_ent[args.dataset]:
                                meta_matrix[i,j,0] += 1
                            if j in link_record:
                                for k in link_record[j]:
                                    if k < num_ent[args.dataset]:
                                        meta_matrix[i,k,meta_path_dict[entity_type[j]]] += 1
                                    if k in link_record:
                                        for l in link_record[k]:
                                            if l < num_ent[args.dataset]:
                                                meta_matrix[i,l,meta_path_dict[entity_type[j] + '_' + entity_type[k]]] += 1
                    print(i)
                meta_feature = np.zeros((num_ent[args.dataset], num_ent[args.dataset], 14 * 4))
                for j in range(14):
                    meta_feature[:,:,j*4] = meta_matrix[:,:,j]
                    meta_feature[:,:,j*4+1] = 2 * meta_matrix[:,:,j]/np.maximum((meta_matrix[:,:,j].sum(0) + (meta_matrix[:,:,j].sum(1))[None,:].T), np.ones((num_ent[args.dataset], num_ent[args.dataset])))
                    meta_feature[:,:,j*4+2] = meta_matrix[:,:,j]/np.maximum(meta_matrix[:,:,j].sum(0), np.ones(num_ent[args.dataset]))
                    meta_feature[:,:,j*4+3] = meta_matrix[:,:,j]/np.maximum(meta_matrix[:,:,j].sum(0), np.ones(num_ent[args.dataset])) + meta_matrix[:,:,j]/np.maximum(meta_matrix[:,:,j].sum(1), np.ones((num_ent[args.dataset], 1)))
                self.meta_feature = meta_feature
                np.save(pathh, meta_feature)
            else:
                self.meta_feature = np.load(pathh)
            self.meta_feature = torch.FloatTensor(self.meta_feature)
        elif args.model == 'Decagon':
            self.edge_index = []
            if args.dataset == 'twosides':
                for sub, obj, rel in self.data['train']:
                    if rel[-1] == 1:
                        self.edge_index.append([sub, obj])
            else:
                for sub, obj, rel in self.data['train']:
                    self.edge_index.append([sub, obj])
            trip_de = []
            file = open('./data/{}/{}.txt'.format(args.dataset, self.BN_path))
            if args.bionet == 'HetioNet':
                for j in file:
                    str_lin = j.strip().split(' ')
                    trip = [int(j) for j in str_lin]
                    # if trip[2] in [1,21]:
                    if trip[2] in [1, 5, 6, 7, 10, 18]:
                        trip_de.append([trip[0], trip[1]])
            elif args.bionet == 'PrimeKG':
                for j in file:
                    str_lin = j.strip().split(' ')
                    trip = [int(j) for j in str_lin]
                    # if trip[2] in [1,21]:
                    if trip[2] in [1,2,3,4,5,6,7,8]:
                        trip_de.append([trip[0], trip[1]])
            num_begin = num_ent[args.dataset]
            ent_in = np.unique(np.array(trip_de).flatten())
            ind_dict = {}
            for j in ent_in:
                if j >= num_ent[args.dataset]:
                    ind_dict[j] = num_begin
                    num_begin += 1
                else:
                    ind_dict[j] = j
            for j in trip_de:
                j = [ind_dict[j[0]], ind_dict[j[1]]]
                self.edge_index.append(j)

            self.edge_index	= torch.LongTensor(self.edge_index).to(self.device).t()

            if args.use_feat:
                feat = torch.zeros((num_begin, self.feat_dim))
                torch.nn.init.xavier_uniform_(feat)
                feat[:num_ent[args.dataset]] = self.feat
                self.feat = feat

        ### the main part

        self.data_iter = {}
        self.data_iter['train'] = self.get_data_loader(TrainDataset, 'train', args.batch_size)
        for j in self.split_not_train:
            self.data_iter[j] = self.get_data_loader(TestDataset, j, args.batch_size, shuffle = False)

    def get_data_loader(self, dataset_class, split, batch_size, shuffle=True):
        return  DataLoader(
            dataset_class(self.triples[split], self.args),
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = 10, ### set the default numworkers to 10
            collate_fn      = dataset_class.collate_fn
        )

class TrainDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int), np.float32(ele['sub_samp'])
		if self.p.dataset == 'drugbank': trp_label = self.get_label_ddi(label) 
		elif self.p.dataset == 'twosides': 
			label = label[0]
			trp_label = torch.FloatTensor(label)
        
		if self.p.model in ['ComplEx', 'MSTE']:
			if self.p.dataset == 'drugbank':
				triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][0]])
			elif self.p.dataset == 'twosides':
				triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][2], ele['label'][0] , ele['label'][1]])
				trp_label = torch.LongTensor([ele['label'][2]])

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label


	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)


class TestDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		if self.p.dataset == 'drugbank': triple, label	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int)
		elif self.p.dataset == 'twosides': triple, label	= torch.LongTensor([ele['triple'][0], ele['triple'][1], -1]), np.array(ele['label'])[0]
		if self.p.dataset == 'drugbank': label		= self.get_label_ddi(label)
		elif self.p.dataset == 'twosides': label = torch.FloatTensor(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label

	def get_label_ddi(self, label):
		if self.p.use_reverse_edge:
			y = np.zeros([int(self.p.num_rel/2)], dtype=np.float32)
		else:
			y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

