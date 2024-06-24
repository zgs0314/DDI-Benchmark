import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning

from .subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from .utils.initialization_utils import initialize_experiment, initialize_model
from .utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_batch_to_device_dgl_ddi2

from .model.dgl.graph_classifier import GraphClassifier as dgl_model

from .managers.evaluator import Evaluator, Evaluator_ddi2
from .managers.trainer import Trainer
import numpy as np
from warnings import simplefilter
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(params):
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)
    else:
        print('yes, we have')

    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model)
    #assert 0
    valid = SubgraphDataset(params.db_path, f'valid_{params.setting}_pos', f'valid_{params.setting}_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model,
                            ssp_graph = train.ssp_graph, 
                            id2entity= train.id2entity, id2relation= train.id2relation, rel= train.num_rels,  graph = train.graph)
    test = SubgraphDataset(params.db_path, f'test_{params.setting}_pos', f'test_{params.setting}_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model,
                            ssp_graph = train.ssp_graph,  
                            id2entity= train.id2entity, id2relation= train.id2relation, rel= train.num_rels,  graph = train.graph)
    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim
    params.train_rels = 209 if params.dataset == 'twosides' else params.num_rels
    params.num_nodes = 91000

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label
    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    graph_classifier = initialize_model(params, dgl_model)
    if params.dataset == 'twosides':
        mfeat = []
        rfeat = []
        import pickle 
        with open('data/{}/DB_molecular_feats.pkl'.format(params.dataset), 'rb') as f:
            mfeat = pickle.load(f, encoding='utf-8')
            params.feat_dim = len(mfeat[0])
    else:
        if params.feat == 'morgan':
            import pickle 
            with open('data/{}/DB_molecular_feats.pkl'.format(params.dataset), 'rb') as f:
                x = pickle.load(f, encoding='utf-8')
            mfeat =  []
            for y in x['Morgan_Features']:
                mfeat.append(y)
            params.feat_dim = 1024
        elif  params.feat == 'pca':
            mfeat = np.loadtxt('data/{}/PCA.txt'.format(params.dataset))
            params.feat_dim = 200
        elif  params.feat == 'pretrained':
            mfeat = np.loadtxt('data/{}/pretrained.txt'.format(params.dataset))
            params.feat_dim = 200

    graph_classifier.drug_feat(torch.FloatTensor(np.array(mfeat)).to(params.device))
    
    valid_evaluator = Evaluator_ddi2(params, graph_classifier, valid) if params.dataset == 'twosides' else Evaluator(params, graph_classifier, valid)
    test_evaluator = Evaluator_ddi2(params, graph_classifier, test) if params.dataset == 'twosides' else Evaluator(params, graph_classifier, test)
    train_evaluator = Evaluator_ddi2(params, graph_classifier, train) if params.dataset == 'twosides' else Evaluator(params, graph_classifier, train)
    
    trainer = Trainer(params, graph_classifier, train, train_evaluator, valid_evaluator,test_evaluator)

    logging.info('Starting training with full batch...')
    trainer.train()
    with open(params.save_result, "a") as f:
        f.write('------------------------------------End-----------------------------------------\n')


# if __name__ == '__main__':
def run_sumgnn(args, file_name, save_path):

    logging.basicConfig(level=logging.INFO)

    # params = parser.parse_args()
    params = args

    ### params from the main function
    params.model_type = 'dgl'
    params.save_pathh = save_path
    params.setting = args.setting_SumGNN
    params.train_file = "train"
    params.valid_S0_file = "valid_S0"
    params.test_S0_file = "test_S0"
    params.valid_S1_file = "valid_S1"
    params.test_S1_file = "test_S1"
    params.valid_S2_file = "valid_S2"
    params.test_S2_file = "test_S2"
    # Training regime params
    params.num_epochs = 40
    params.eval_every = 3
    params.eval_every_iter = 526
    params.save_every = 10
    params.early_stop = 10
    params.optimizer = "Adam"
    params.clip = 1000
    params.l2 = 1e-5
    # Data processing pipeline params
    params.max_links = 250000
    params.hop = 2
    params.max_nodes_per_hop = 200
    params.use_kge_embeddings = False
    params.kge_model = "TransE"
    params.constrained_neg_prob = 0.0
    params.num_neg_samples_per_link = 0
    params.num_workers = 10
    params.add_traspose_rels = False
    params.enclosing_sub_graph = True
    # Model params
    params.rel_emb_dim = 32
    params.attn_rel_emb_dim = 32
    params.emb_dim = 32
    params.num_gcn_layers = 2
    params.num_bases = 8
    params.dropout = 0.3
    params.edge_dropout = 0.4
    params.gnn_agg_type = 'sum'
    params.add_ht_emb = True
    params.add_sb_emb = True
    params.has_attn = True
    params.has_kg = True
    params.feat = 'morgan'
    params.add_feat_emb = True
    params.add_transe_emb = True
    params.gamma = 0.2

    if params.dataset == 'drugbank':
        params.feat_dim = 1024
    elif params.dataset == 'twosides':
        params.feat_dim = 2048
    
    initialize_experiment(params, __file__, file_name)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid_S0': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_S0_file)),
        'test_S0': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_S0_file)),
        'valid_S1': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_S1_file)),
        'test_S1': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_S1_file)),
        'valid_S2': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_S2_file)),
        'test_S2': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_S2_file))
    }
    # params.save_result = os.path.join(params.exp_dir, f'{params.dataset}_{params.setting}.txt')
    params.save_result = os.path.join(params.exp_dir, 'record' ,file_name + '.txt')


    if params.dataset != 'twosides':
        params.relation_class = {'Pharmacokinetic interactions - Absorption interactions': [2, 12, 17, 61, 66],
                    'Pharmacokinetic interactions - Distribution interacitons': [42, 44, 72, 74],
                    'Pharmacokinetic interactions - Metabolic interactions': [3, 10, 46],
                    'Pharmacokinetic interactions - Excretion interactions': [64, 71],
                    'Pharmacodynamic interactions - Additive or synergistic effects': [0, 1, 5, 6, 7, 8, 9, 14, 15, 18, 19, 20, 21, 22, 23, 
                    24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 45, 51, 52, 53, 54, 55, 56, 58, 59, 62, 63, 67, 68, 70,
                    73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
                    'Pharmacodynamic interacitons - Antagonistic effects': [4, 11, 13, 16, 25, 28, 36, 47, 48, 49, 50, 57, 60, 65, 69, 75]}

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl_ddi2 if params.dataset == 'twosides' else move_batch_to_device_dgl

    main(params)
