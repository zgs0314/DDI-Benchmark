import os
import argparse
import torch
import random
from .load_data import DataLoader

from .base_model import BaseModel
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
import time

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"


# if __name__ == '__main__':
def run_emergnn_drugbank(argss, file_name, save_path):

    parser = argparse.ArgumentParser(description="Parser for EmerGNN")
    parser.add_argument('--task_dir', type=str, default='./', help='the directory to dataset')
    parser.add_argument('--dataset', type=str, default='S1', help='the directory to dataset')
    parser.add_argument('--lamb', type=float, default=7e-4, help='set weight decay value')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id to load.')
    parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
    parser.add_argument('--lr', type=float, default=0.03, help='set learning rate')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--n_batch', type=int, default=512, help='batch size')
    parser.add_argument('--epoch_per_test', type=int, default=5, help='frequency of testing')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--out_file_info', type=str, default='', help='extra string for the output file name')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    args.relation_class = {'Pharmacokinetic interactions - Absorption interactions': [2, 12, 17, 61, 66],
                'Pharmacokinetic interactions - Distribution interacitons': [42, 44, 72, 74],
                'Pharmacokinetic interactions - Metabolic interactions': [3, 10, 46],
                'Pharmacokinetic interactions - Excretion interactions': [64, 71],
                'Pharmacodynamic interactions - Additive or synergistic effects': [0, 1, 5, 6, 7, 8, 9, 14, 15, 18, 19, 20, 21, 22, 23, 
                24, 26, 27, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 43, 45, 51, 52, 53, 54, 55, 56, 58, 59, 62, 63, 67, 68, 70,
                73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85],
                'Pharmacodynamic interacitons - Antagonistic effects': [4, 11, 13, 16, 25, 28, 36, 47, 48, 49, 50, 57, 60, 65, 69, 75]}
    args.sett = 'drugbank'
    torch.cuda.set_device(args.gpu)
    dataloader = DataLoader(args)
    eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
    args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, dataloader.eval_rel
    KG = dataloader.KG
    vKG = dataloader.vKG
    tKG = dataloader.tKG
    triplets = dataloader.triplets
    train_pos, train_neg = torch.LongTensor(triplets['train']).cuda(), None
    valid_pos, valid_neg = torch.LongTensor(triplets['valid']).cuda(), None
    test_pos,  test_neg  = torch.LongTensor(triplets['test']).cuda(), None

    if not os.path.exists('results'):
        os.makedirs('results')

    def run_model(seed, file_name, save_path):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.dataset.startswith('S1'):
            args.lr = 0.0003
            args.lamb = 0.00000001
            args.weight = 0.
            args.length = 3
            args.n_batch = 64
            args.n_dim = 64
            args.feat = 'M'
        elif args.dataset.startswith('S2'):
            args.lr = 0.003
            args.lamb = 0.0001
            args.weight = 0.
            args.length = 3
            args.n_batch = 32
            args.n_dim = 32
            args.feat = 'M'
        elif args.dataset.startswith('S0'):
            args.lr = 0.003
            args.lamb = 0.0001
            args.weight = 0
            args.length = 3
            args.n_batch = 32
            args.n_dim = 64
            args.feat = 'E'

        model = BaseModel(eval_ent, eval_rel, args)
        best_acc = -1
        for e in range(args.n_epoch):
            dataloader.shuffle_train()
            KG = dataloader.KG
            train_pos = torch.LongTensor(dataloader.train_data).cuda()
            model.train(train_pos, None, KG)
            if (e+1) % args.epoch_per_test == 0:
                v_f1, v_acc, v_kap, _, _ = model.evaluate(valid_pos, valid_neg, vKG)
                t_f1, t_acc, t_kap, t_for_six_class, t_per_class = model.evaluate(test_pos,  test_neg,  tKG)
                model.scheduler.step(v_f1)
                time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                out_str = time_now + ' :epoch:%d\tfeat:%s lr:%.6f lamb:%.8f n_batch:%d n_dim:%d layer:%d\t[Valid] f1:%.4f acc:%.4f kap:%.4f\t[Test] f1:%.4f acc:%.4f kap:%.4f' % (e+1, args.feat, args.lr, args.lamb, args.n_batch, args.n_dim, args.length, v_f1, v_acc, v_kap, t_f1, t_acc, t_kap)
                out_str_class = f'[Test per six class]: {t_for_six_class} \n [Test per class]: {t_per_class} \n'
                if v_f1 > best_acc:
                    best_acc = v_f1
                    best_str = out_str
                    best_str_class = out_str_class
                    if args.save_model:
                        model.save_model(best_str, save_path)
                print(out_str)
                with open(os.path.join('record' ,file_name + '.txt'), 'a+') as f:
                    f.write(out_str + '\n')
                    f.write(out_str_class + '\n')
        print('Best results:\t' + best_str)
        with open(os.path.join('record' ,file_name + '.txt'), 'a+') as f:
            f.write('Best results:\t' + best_str + '\n\n')
            f.write(best_str_class + '\n')
        return -best_acc

    run_model(args.seed, file_name, save_path)
    

