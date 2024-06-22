import torch
import os
import numpy as np
import time
import logging
import random
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import json
from sklearn import metrics
from ..utils.graph_utils import collate_dgl, move_batch_to_device_dgl

GLOBAL_SEED=1
GLOBAL_WORKER_ID=None

class Trainer(object):
    def __init__(self, params, model, train_data,valid_evaluator,test_evaluator):

        self.params = params

        self.pathh = params.save_pathh
        self.patht = './record/' + params.file_namee

        self.graph_classifier = model

        with open(self.patht , 'a+') as f:
            f.write(str(vars(self.params)) + '\n')

        self.train_data = train_data

        self.valid_evaluator_S0, self.valid_evaluator_S1, self.valid_evaluator_S2 = valid_evaluator
        self.test_evaluator_S0, self.test_evaluator_S1, self.test_evaluator_S2 = test_evaluator

        self.batch_size = params.batch_size
        self.collate_fn = collate_dgl
        self.num_workers = params.num_workers
        self.params = params
        self.updates_counter = 0
        self.early_stop = 0
        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        self.optimizer = Adam(self.graph_classifier.parameters(), lr=params.lr, weight_decay=params.weight_decay_rate)
        self.scheduler = ExponentialLR(self.optimizer, params.lr_decay_rate)

        if params.dataset in ['drugbank']:
            self.criterion = nn.CrossEntropyLoss()
        elif params.dataset in ['twosides']:
            self.criterion = nn.BCELoss(reduce=False) 
        self.move_batch_to_device = move_batch_to_device_dgl
        self.reset_training_state()
        

    def train_batch(self):
        total_loss = 0
        all_labels = []
        all_scores = []
        def init_fn(worker_id): 
            global GLOBAL_WORKER_ID
            GLOBAL_WORKER_ID = worker_id
            seed = GLOBAL_SEED + worker_id
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False

        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn, worker_init_fn=init_fn)
        self.graph_classifier.train()
        bar = tqdm(enumerate(train_dataloader))

        for b_idx, batch in bar:
            #polarity are whether these data are pos or neg 
            if self.params.dataset in ['drugbank']:
                subgraph_data, relation_labels, polarity = self.move_batch_to_device(batch, self.params.device,multi_type=1)
            elif self.params.dataset in ['twosides']:
                subgraph_data, relation_labels, polarity = self.move_batch_to_device(batch, self.params.device,multi_type=2)
            self.optimizer.zero_grad()
            scores = self.graph_classifier(subgraph_data)

            if self.params.dataset in ['drugbank']:
                loss = self.criterion(scores, relation_labels)
            elif self.params.dataset in ['twosides']:
                m = nn.Sigmoid()
                scores = m(scores)
                polarity = polarity.unsqueeze(1)
                loss_train = self.criterion(scores, relation_labels * polarity)
                loss = torch.sum(loss_train * relation_labels) 
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            bar.set_description('batch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()))
            with torch.no_grad():
                total_loss += loss.item()
                if self.params.dataset != 'twosides':
                    label_ids = relation_labels.to('cpu').numpy()
                    all_labels += label_ids.flatten().tolist()
                    all_scores += torch.argmax(scores, dim=1).cpu().flatten().tolist() 

            # valid and test
            if self.updates_counter % self.params.eval_every_iter == 0:
                improved = False
                result_S0 = self.valid_evaluator_S0.eval_0()
                result_S1 = self.valid_evaluator_S1.eval_0()
                result_S2 = self.valid_evaluator_S2.eval_0()
                logging.info(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
                logging.info('Eval Performance S0:' + str(result_S0))
                logging.info('Eval Performance S1:' + str(result_S1))
                logging.info('Eval Performance S2:' + str(result_S2))
                with open(self.patht , 'a+') as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n')
                    f.write('Eval Performance S0:' + str(result_S0) + '\n')
                    f.write('Eval Performance S1:' + str(result_S1) + '\n')
                    f.write('Eval Performance S2:' + str(result_S2) + '\n\n')
                # logging.info('Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'s')
                if result_S0['acc'] > self.best_S0:
                    self.best_S0 = result_S0['acc']
                    improved = True
                    self.save_model(self.pathh + '_S0')
                if result_S1['acc'] > self.best_S1:
                    self.best_S1 = result_S1['acc']
                    improved = True
                    self.save_model(self.pathh + '_S1')
                if result_S2['acc'] > self.best_S2:
                    self.best_S2 = result_S2['acc']
                    improved = True
                    self.save_model(self.pathh + '_S2')
                if not improved:
                    self.not_improve += 1
                self.scheduler.step()

        
        return total_loss/b_idx

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))
        logging.info('Better models found w.r.t accuracy. Saved it!')

    def save_model(self, save_path):
        state = {
			'state_dict'	: self.graph_classifier.state_dict(),
			# 'optimizer'	: self.optimizer.state_dict(),
		}
        torch.save(state, save_path)

    def load_model(self, load_path):
        # print(torch.cuda.device_count())
        state			= torch.load(load_path, map_location = self.params.device)
        self.graph_classifier.load_state_dict(state['state_dict'])
        # self.optimizer.load_state_dict(state['optimizer'])

    def reset_training_state(self):
        self.best_metric = 0
        self.test_best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 1

    def train(self):
        ### use accuracy
        self.best_S0, self.best_S1, self.best_S2, self.not_improve = 0, 0, 0, 0
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss = self.train_batch()
            logging.info(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + f' Epoch {epoch} with loss: {loss}')
            with open(self.patht , 'a+') as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + f' Epoch {epoch} with loss: {loss}')
            if self.not_improve >= 20:
                break

        ### for test performance evaluation
        self.load_model(self.pathh + '_S0')
        if self.params.dataset == 'drugbank':
            result_S0, class_S0, str_S0 = self.test_evaluator_S0.eval_0()
        elif self.params.dataset == 'twosides':
            result_S0 = self.test_evaluator_S0.eval_0()
        self.load_model(self.pathh + '_S1')
        if self.params.dataset == 'drugbank':
            result_S1, class_S1, str_S1 = self.test_evaluator_S1.eval_0()
        elif self.params.dataset == 'twosides':
            result_S1 = self.test_evaluator_S1.eval_0()
        self.load_model(self.pathh + '_S2')
        if self.params.dataset == 'drugbank':
            result_S2, class_S2, str_S2 = self.test_evaluator_S2.eval_0()
        elif self.params.dataset == 'twosides':
            result_S2 = self.test_evaluator_S2.eval_0()
        logging.info(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        logging.info('Test Performance S0:' + str(result_S0))
        logging.info('Test Performance S1:' + str(result_S1))
        logging.info('Test Performance S2:' + str(result_S2))
        
        if self.params.dataset == 'drugbank':
            with open(self.patht , 'a+') as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n')
                f.write('Test Performance S0:' + str(result_S0) + '\n')
                f.write(str_S0)
                f.write('Test Performance S1:' + str(result_S1) + '\n')
                f.write(str_S1)
                f.write('Test Performance S2:' + str(result_S2) + '\n')
                f.write(str_S2)
        elif self.params.dataset == 'twosides':
            with open(self.patht , 'a+') as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + '\n')
                f.write('Test Performance S0:' + str(result_S0) + '\n')
                f.write('Test Performance S1:' + str(result_S1) + '\n')
                f.write('Test Performance S2:' + str(result_S2) + '\n')

