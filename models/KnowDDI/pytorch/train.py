import os
import argparse
import random
import torch
import numpy as np
import logging
from warnings import simplefilter
from scipy.sparse import SparseEfficiencyWarning
from .manager.trainer import Trainer
from .manager.evaluator import Evaluator_multiclass, Evaluator_multilabel
from .model.Classifier_model import Classifier_model
from .utils.initialization_utils import initialize_experiment, initialize_model
from .data_processor.datasets import SubgraphDataset
from .data_processor.subgraph_extraction import generate_subgraph_datasets
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def process_dataset(params):
    params.db_path = os.path.join(params.main_dir, f'../../../data/{params.dataset}/digraph_hop_{params.hop}_{params.BKG_file_name}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)
 
    train_data = SubgraphDataset(db_path=params.db_path,
                                db_name='train_subgraph',
                                raw_data_paths=params.file_paths,
                                add_traspose_rels=params.add_traspose_rels,
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)

    
    test_S0_data = SubgraphDataset(db_path=params.db_path,
                                db_name='test_S0_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)
    valid_S0_data = SubgraphDataset(db_path=params.db_path,
                                db_name='valid_S0_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)
    test_S1_data = SubgraphDataset(db_path=params.db_path,
                                db_name='test_S1_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)
    valid_S1_data = SubgraphDataset(db_path=params.db_path,
                                db_name='valid_S1_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)
    test_S2_data = SubgraphDataset(db_path=params.db_path,
                                db_name='test_S2_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)
    valid_S2_data = SubgraphDataset(db_path=params.db_path,
                                db_name='valid_S2_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)


    params.num_rels = train_data.num_rels  # only relations in dataset
    params.global_graph = train_data.global_graph.to(params.device)
    params.aug_num_rels = train_data.aug_num_rels  # including relations in BKG and self loop
    if params.BKG_file_name == 'BN_Primekg':
        params.num_nodes = 91000
    else:
        params.num_nodes = 40000 # 35000
    logging.info(f"Device: {params.device}")
    logging.info(f" # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    return train_data, valid_S0_data, test_S0_data, valid_S1_data, test_S1_data, valid_S2_data, test_S2_data

def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'train')),
        'valid_S0': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'valid_S0')),
        'test_S0': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'test_S0')),
        'valid_S1': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'valid_S1')),
        'test_S1': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'test_S1')),
        'valid_S2': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'valid_S2')),
        'test_S2': os.path.join(params.main_dir, '../../../data/{}/{}.txt'.format(params.dataset, 'test_S2')),
    }
    train_data, valid_S0_data, test_S0_data, valid_S1_data, test_S1_data, valid_S2_data, test_S2_data = process_dataset(params)
    classifier = initialize_model(params, Classifier_model)
    if params.dataset == 'drugbank':
        valid_evaluator = [Evaluator_multiclass(params, classifier, valid_S0_data), Evaluator_multiclass(params, classifier, valid_S1_data), Evaluator_multiclass(params, classifier, valid_S2_data)]
        test_evaluator = [Evaluator_multiclass(params, classifier, test_S0_data,is_test=True), Evaluator_multiclass(params, classifier, test_S1_data,is_test=True), Evaluator_multiclass(params, classifier, test_S2_data,is_test=True)]
    elif params.dataset == 'twosides':
        valid_evaluator = [Evaluator_multilabel(params, classifier, valid_S0_data), Evaluator_multilabel(params, classifier, valid_S1_data), Evaluator_multilabel(params, classifier, valid_S2_data)]
        test_evaluator = [Evaluator_multilabel(params, classifier, test_S0_data), Evaluator_multilabel(params, classifier, test_S1_data), Evaluator_multilabel(params, classifier, test_S2_data)]
    
    print(classifier)

    trainer = Trainer(params, classifier, train_data, valid_evaluator, test_evaluator)
    logging.info('start training...')
    trainer.train()

def run_knowddi(args, file_name, save_path):
    
    logging.basicConfig(level=logging.INFO)
    # parser = argparse.ArgumentParser(description="model params")

    # params = parser.parse_args()
    
    params = args

    params.file_namee = file_name
    params.save_pathh = save_path
    params.load_model = False
    # dataset
    params.train_file = 'train'
    params.valid_file = 'valid'
    params.test_file = 'test'
    params.kge_model = 'TransE'
    params.use_pre_embeddings = False
    params.BKG_file_name = 'relations_2hop'
    # extract subgraphs 
    params.max_links = 250000
    params.hop = 2
    params.max_nodes_per_hop = 200
    params.enclosing_subgraph = True
    params.add_traspose_rels = False
    # trainer
    params.eval_every_iter = 526
    params.save_every_epoch = 10
    params.early_stop_epoch = 10
    params.optimizer = "Adam"
    params.lr_decay_rate = 0.93
    params.weight_decay_rate = 1e-5
    params.batch_size = 256
    params.num_epochs = 100
    params.num_workers = 32
    # GraphSAGE params
    params.emb_dim = 32
    params.num_gcn_layers = 2
    params.gcn_aggregator_type = 'mean'
    params.gcn_dropout = 0.2
    # gsl_Model params
    params.num_infer_layers = 3
    params.num_dig_layers = 3
    params.MLP_hidden_dim = 16
    params.MLP_num_layers = 2
    params.MLP_dropout = 0.2
    params.func_num = 1
    params.sparsify = 1
    params.threshold = 0.05
    params.edge_softmax = 1
    params.gsl_rel_emb_dim = 32
    params.lamda = 0.7
    params.gsl_has_edge_emb = 1

    if params.dataset == 'twosides':
        params.eval_every_iter = 452
        params.weight_decay_rate = 0.00001
        params.threshold = 0.1
        params.lamda = 0.5
        params.num_infer_layers = 1
        params.num_dig_layers = 3
        params.gsl_rel_emb_dim = 24
        params.MLP_hidden_dim = 24
        params.MLP_num_layers = 3
        params.MLP_dropout = 0.2

    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    set_seed(params.seed)
    initialize_experiment(params, __file__, file_name)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    main(params)


