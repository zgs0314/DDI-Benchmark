from .helper import *
from .message_passing import MessagePassing

class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels))

		self.drop		= torch.nn.Dropout(self.p.comp_drop)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, rel_embed): 
		# x: embedding for all nodes, edge index: saves all (h,t)s, edge type: save corresponding (r), 
		# rel_embedding: relation embeddings, contain relation in both direction
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0) # here add one relation as the self-loop
		num_edges = edge_index.size(1) // 2 # number of edges
		num_ent   = x.size(0) 
		
		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:] # cut triplets forward and backward
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:] 

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device) # repeat node index two times
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device) # repeat the number of relations (14541)

		self.in_norm     = self.compute_norm(self.in_index,  num_ent) # give each edge a weight based on appearance times
		self.out_norm    = self.compute_norm(self.out_index, num_ent)

		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop') # seems like self loop
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3) ### each for 1/3 ? maybe we can adapt that

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed): # the composition step
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode)) # obtain weights according to node
		if len(edge_type.shape) == 1: # edge_type.float() @ rel_embed
			rel_emb = torch.index_select(rel_embed, 0, edge_type) # form the matrix of relation embedding
		else:
			rel_emb = edge_type.float() @ rel_embed[:-1]
		# rel_emb = torch.index_select(rel_embed, 0, edge_type) 
		xj_rel  = self.rel_transform(x_j, rel_emb) # the composition step
		out	= torch.mm(xj_rel, weight) # compute weight with compositioned embedding

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index # h and t
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges, same function as np.bincount(), count times of appearance of each nodes
		deg_inv		= deg.pow(-0.5)							# D^{-0.5} 1/sqrt(D)
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5} # I cannot 

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
