import os
import argparse
from trainer import Trainer
from utils import *
import torch
import numpy as np

print('pid:', os.getpid())

def main():
    ### set process name

    ### set hyperparameters
    parser = argparse.ArgumentParser(description='DDI Benchmark')
    # general hyperparameters
    parser.add_argument('--model', type=str, default='EmerGNN', choices=['CompGCN', 'SkipGNN', 'ComplEx', 'MSTE', 'MLP', 'KGDDI', 'CSMDDI', 'HINDDI', 'Decagon', 'SumGNN', 'KnowDDI', 'EmerGNN'])
    parser.add_argument('--problem', type=str, default='DDI', choices=['DDI'])
    parser.add_argument('--DDIsetting', type=str, default='all', choices=['S0', 'S1', 'S2', 'all'])
    parser.add_argument('--dataset', type=str, default='twosides', choices=['drugbank', 'twosides'])
    parser.add_argument('--bionet', type=str, default='PrimeKG', choices=['HetioNet', 'PrimeKG'])
    parser.add_argument('--name', default='testrun', help='Set run name for saving/restoring models')

    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay")
    parser.add_argument('--lbl_smooth',	type=float,     default=0.0,	help='Label Smoothing') ### usually 0-1
    parser.add_argument("--epoch", type=int, default=100, help="training epoch")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--use_feat', default=1, type=bool, help='Whether to use drug feature')
    parser.add_argument('--use_reverse_edge', default=0, type=bool, help='Whether to add reverse edges in the training step')
    parser.add_argument('--data_aug', default=0, type=bool, help='Whether to add data as augmentation')

    parser.add_argument('--seed', default=123, type=int, help='Seed for randomization')
    parser.add_argument('--eval_skip', default=1, type=int, help='Evaluate every x epochs')

    # CompGCN hyper_parameters
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')
    parser.add_argument("--Comp_sfunc", type=str, default='TransE', help="the score function for CompGCN")
    parser.add_argument('--gamma',		type=float,             default=32.0,			help='Margin') 
    parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('--comp_dim',  type=int, default=200, help='dimension setting for CompGCN model')
    parser.add_argument('--comp_drop', type=float,	default=0, help='Dropout to use in CompGCN model')

    # SkipGNN hyper_parameters
    parser.add_argument('--skip_hidden', type=int, default=200, help='Number of hidden units for decoding layer 1.')
    parser.add_argument('--skip_dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    
    # KGE models
    parser.add_argument('--kge_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--kge_gamma', type=int, default=1, help='gamma parameter.')
    parser.add_argument('--kge_dropout', type=float, default=0, help='dropout rate.') ### DDI best 0
    parser.add_argument('--kge_loss', type=str, default='BCE_mean',  help='loss function')
    
    # MLP model
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--mlp_dim', type=int, default=200, help='hidden dimension.')

    ### KG-DDI model
    parser.add_argument('--kgddi_dim', type=int, default=200, help='hidden dimension.')

    ### CSMDDI model
    parser.add_argument('--csm_dim', type=int, default=200, help='hidden dimension.')

    ### HINDDI model
    parser.add_argument('--hin_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--hin_featset', type=int, default=3, help='feature that use.')
    parser.add_argument('--with_1hop', type=int, default=1, help='whether to add 1-hop neighbor to data.')

    ### Decagon model decagon_drop
    parser.add_argument('--decagon_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--decagon_drop', type=float,	default=0.1, help='Dropout to use in Decagon model')

    ### SumGNN model
    parser.add_argument('--setting_SumGNN', type=str, default='S0', help='hidden dimension.')

    ### EmerGNN model
    parser.add_argument('--setting_EmerGNN', type=str, default='S0', help='hidden dimension.')

    ### set basic configurations
    args = parser.parse_args()

    ### set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model in ['ComplEx', 'MSTE']:
        args.use_feat = 0

    if args.dataset in ['twosides']:
        # args.use_feat = 0
        if args.model in ['ComplEx', 'MSTE']:
            args.batch_size = 512 # 2048
        if args.model == 'CompGCN':
            args.use_feat = 0

    if args.model == 'CSMDDI':
        args.epoch = 40
        args.eval_skip = 1
    
    if args.data_aug:
        if args.model == 'CompGCN':
            args.batch_size = 128
        elif args.model == 'SkipGNN':
            args.batch_size = 512
        else:
            args.batch_size = 512
    
    args.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"

    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
