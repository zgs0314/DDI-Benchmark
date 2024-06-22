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

    graph_classifier = initialize_model(params, dgl_model, params.load_model)
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

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default1",
                        help="A folder with this name would be created to dump saved models and log files")
    # parser.add_argument("--dataset", "-d", type=str, default="drugbank", ### drugbank, twosides
    #                     help="Dataset string")
    # parser.add_argument("--gpu", type=int, default=2,
    #                     help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', type=bool, default=0,
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_S0_file", "-vf0", type=str, default="valid_S0",
                        help="Name of S0 file containing validation triplets")
    parser.add_argument("--test_S0_file", "-ttf0", type=str, default="test_S0",
                        help="Name of S0 file containing validation triplets")
    parser.add_argument("--valid_S1_file", "-vf1", type=str, default="valid_S1",
                        help="Name of S1 file containing validation triplets")
    parser.add_argument("--test_S1_file", "-ttf1", type=str, default="test_S1",
                        help="Name of S1 file containing validation triplets")
    parser.add_argument("--valid_S2_file", "-vf2", type=str, default="valid_S2",
                        help="Name of S2 file containing validation triplets")
    parser.add_argument("--test_S2_file", "-ttf2", type=str, default="test_S2",
                        help="Name of S2 file containing validation triplets")
    # parser.add_argument("--setting", "-s", type=str, default="S0",
    #                     help="DDI problem setting")
    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=40,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=526,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    # parser.add_argument("--lr", type=float, default=5e-3,
    #                     help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=1e-5,
                        help="Regularization constant for GNN weights")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=250000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=200,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=0,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=2,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=8,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.4,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--add_sb_emb', '-sb', type=bool, default=True,
                        help='whether to concatenate subgraph embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument('--has_kg', '-kg', type=bool, default=True,
                        help='whether to have kg in model or not')
    parser.add_argument('--feat', '-f', type=str, default='morgan',
                        help='the type of the feature we use in molecule modeling')
    # parser.add_argument('--feat_dim', type=int, default=1024, ### 1024, 2048
    #                     help='the dimension of the feature')
    parser.add_argument('--add_feat_emb', '-feat', type=bool, default=True,
                        help='whether to morgan feature embedding in model or not')
    parser.add_argument('--add_transe_emb', type=bool, default=True,
                        help='whether to have knowledge graph embedding in model or not')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='The threshold for attention')
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")

    params = parser.parse_args()
    initialize_experiment(params, __file__, file_name)

    ### params from the main function
    params.save_pathh = save_path
    params.dataset = args.dataset
    params.gpu = args.gpu
    params.lr = args.lr
    params.setting = args.setting_SumGNN
    if params.dataset == 'drugbank':
        params.feat_dim = 1024
    elif params.dataset == 'twosides':
        params.feat_dim = 2048

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
