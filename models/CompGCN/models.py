from .helper import *
from .compgcn_conv import CompGCNConv
import torch.nn as nn

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		if self.p.dataset == 'drugbank':
			self.bceloss	= torch.nn.BCELoss()
		elif self.p.dataset == 'twosides':
			self.bceloss	= torch.nn.BCELoss(weight = torch.tensor(self.p.loss_weight))

	def loss(self, pred, true_label):
		if self.p.dataset == 'drugbank':
			return self.bceloss(pred, true_label)
		else:
			return self.bceloss(torch.sigmoid(pred)*true_label[:,:-1], true_label[:,:-1]*true_label[:,-1].unsqueeze(1))

class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None, feat = None):
		super(CompGCNBase, self).__init__(params)
		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.device		= self.edge_index.device
		self.p.gcn_dim		= self.p.comp_dim if self.p.gcn_layer == 1 else self.p.comp_dim
		if params.use_feat:
			self.init_embed	= nn.Parameter(feat, requires_grad=False) ### 
			self.init_rel = get_param((num_rel,   feat.shape[1]))
			self.conv1 = CompGCNConv(feat.shape[1], self.p.comp_dim,      num_rel, act=self.act, params=self.p)
		else:
			self.init_embed	= get_param((self.p.num_ent,   self.p.comp_dim))
			self.init_rel = get_param((num_rel,   self.p.comp_dim))
			self.conv1 = CompGCNConv(self.p.comp_dim, self.p.comp_dim,      num_rel, act=self.act, params=self.p)
  
		self.conv2 = CompGCNConv(self.p.comp_dim,    self.p.comp_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		# self.register_parameter('bias', Parameter(torch.zeros(self.p.num_rel * 2)))
		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_rel)))

	def forward_base(self, sub, rel, drop1, drop2):

		r	= self.init_rel if self.p.Comp_sfunc != 'TransE' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub) ### select the corresponding embedding from x
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x

	def forward_base_ddi(self, sub, obj, drop1, drop2):
		r	= self.init_rel 
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub) ### select the corresponding embedding from x
		obj_emb	= torch.index_select(x, 0, obj) 

		return sub_emb, obj_emb, r
	
class CompGCN_TransE_DDI(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None, feat = None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, feat)
		self.drop = torch.nn.Dropout(self.p.comp_drop)

	def forward(self, data): ### specially for obj

		sub_emb, obj_emb, all_ent	= self.forward_base_ddi(data[0], data[1], self.drop, self.drop)
		rel_emb	= obj_emb - sub_emb 

		x	= self.p.gamma - torch.norm(rel_emb.unsqueeze(1) - all_ent, p=1, dim=2)	/ 10	
		if self.p.dataset == 'drugbank':
			score	= torch.sigmoid(x)
		else:
			score = x

		return score
