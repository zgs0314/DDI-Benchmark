import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import *

from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, average_precision_score,accuracy_score

from pprint import pprint

from data_process import *

### ddd
from models.KnowDDI import *
###
from models.SumGNN import *
###
from models.EmerGNN import *
### 

num_ent = {'drugbank': 1710, 'twosides': 645, 'HetioNet': 34124}
num_rel = {'drugbank': 86, 'twosides': 209} # 209, 309, 188

# import warnings
# warnings.filterwarnings('always')

class Trainer():
    def __init__(self, args):
        super(Trainer, self).__init__()

        self.args = args

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        if not os.path.exists('./record'):
            os.makedirs('./record')

        self.file_name = self.args.dataset + '_' + self.args.model + '_' + self.args.DDIsetting + '_' + str(self.args.gpu) + '_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) 
        self.save_path = os.path.join('./checkpoints', self.args.dataset + '_' + self.args.model + '_' + self.args.DDIsetting + '_' + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        ### things need to be recorded in the record name: dataset, model, setting, time
        if self.args.model in ['KnowDDI', 'EmerGNN', 'SumGNN']:
            pass
        else:
            pprint(vars(self.args))

            with open(os.path.join('record', self.file_name), 'w') as f:
                f.write(str(vars(self.args)) + '\n')
                # f.close()
            
            self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
            args.device = self.device
            
            if self.args.model == 'KGDDI':
                emb_return = pretrain_bn(args, self.device, self.file_name)
                args.KGDDI_pre = 0
            else:
                emb_return = None

            self.data_record = Data_record(args, emb_return)

            if self.args.dataset == 'twosides':
                occur = (np.array([j[2] for j in self.data_record.triplets['train']]).sum(0))[:-1]
                args.loss_weight = occur.min()/occur

            self.model = add_model(args, self.data_record, self.device) ###

            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay) ###

    def run(self):
        if self.args.model in ['KnowDDI', 'EmerGNN', 'SumGNN']:
            if self.args.model == 'KnowDDI':
                run_knowddi(self.args, self.file_name, self.save_path)
            elif self.args.model == 'EmerGNN':
                if self.args.dataset == 'drugbank':
                    run_emergnn_drugbank(self.args, self.file_name, self.save_path)
                elif self.args.dataset == 'twosides':
                    run_emergnn_twosides(self.args, self.file_name, self.save_path)
            elif self.args.model == 'SumGNN':
                run_sumgnn(self.args, self.file_name, self.save_path)
        else:
            self.model.train()
            self.valid_split = [j for j in self.data_record.split_not_train if 'valid' in j]
            self.test_split = [j for j in self.data_record.split_not_train if 'test' in j]
            self.best_val_acc = {j:0. for j in self.valid_split}
            self.no_update_epoch = {j:0 for j in self.valid_split}
            # save_path = os.path.join('./checkpoints', self.args.name)
            for epoch in range(self.args.epoch):
                train_loss  = self.run_epoch(epoch)

                print(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + ' [Epoch {}]: Training Loss: {:.5}'.format(epoch, train_loss))
                with open(os.path.join('record', self.file_name), 'a+') as f:
                    f.write(time.strftime("\n%Y-%m-%d %H:%M:%S",time.localtime()) + ' [Epoch {}]: Training Loss: {:.5}\n'.format(epoch, train_loss))

                if epoch % self.args.eval_skip == 0:
                    val_results = self.evaluate('valid', epoch)

                    break_flag = self.update_result(val_results)
                    if break_flag:
                        print("Early Stopping!!")
                        break

            print('Loading best model, Evaluating on Test data')
            test_results = self.evaluate('test', epoch)
            return 

    def run_epoch(self, epoch):
        self.model.train()
        losses = []

        if self.args.model in ['CSMDDI']: 
            loss = train_wo_batch(self.model, self.optimizer, self.device, self.args, self.data_record)
            return loss
        train_iter = iter(self.data_record.data_iter['train'])
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            split = 'train'
            data, label = read_batch(batch, split, self.device, self.args, self.data_record) 

            pred	= self.model.forward(data)
            loss	= self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' [E:{}| {}]: Train Loss:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.args.name))

        loss = np.mean(losses)

        return loss

    def evaluate(self, split, epoch):
        results = {}
        result_record = []
        split_this = self.valid_split if split == 'valid' else self.test_split
        if split == 'valid' and self.args.model in ['CSMDDI']:
            self.model.pre_process()
        for j in split_this:
            if 'test' in j:
                self.load_model(self.save_path + j[-3:])
            valid_results, valid_record = self.predict(j, epoch)
            result_record.append(valid_record)
            results[j] = valid_results
        for j in result_record:
            print(j)
            with open(os.path.join('record', self.file_name), 'a+') as f:
                f.write(j)
        return results

    def predict(self, split, epoch):
        self.model.eval()
        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_record.data_iter[split])

            label_list = []
            pred_list = []

            for step, batch in enumerate(train_iter):
                data, label	= read_batch(batch, split, self.device, self.args, self.data_record) 
                pred = self.model.forward(data)
                if self.args.use_reverse_edge:
                    pred = pred[:,:len(label[1])]
                if self.args.eval_skip:
                    pred = pred[:,:num_rel[self.args.dataset]]
                if self.args.dataset == 'drugbank':
                    pred = pred.argmax(1).cpu().numpy()
                    label = label.argmax(1).cpu().numpy()
                    pred_list.append(pred)
                    label_list.append(label)
                elif self.args.dataset == 'twosides':
                    pred = torch.sigmoid(pred).cpu().numpy()
                    label = label.cpu().numpy()
                    pred_list.append(pred)
                    label_list.append(label)
            
            if self.args.dataset == 'drugbank':
                pred_final = np.concatenate(pred_list)
                label_final = np.concatenate(label_list)
                accuracy = np.sum(pred_final == label_final) / len(pred_final)
                f1 = f1_score(label_final, pred_final, average='macro')
                kappa = cohen_kappa_score(label_final, pred_final)

                results['accuracy'] = accuracy
                results['f1'] = f1
                results['kappa'] = kappa
                str_record = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' {} [Epoch {} {}]: F1-score : {:.5}, Accuracy : {:.5}, Kappa : {:.5}\n'.format(split ,epoch, split, results['f1'], results['accuracy'], results['kappa'])

            elif self.args.dataset == 'twosides':
                pred_final = np.concatenate(pred_list)
                label_final = np.concatenate(label_list)
                pred_cun = []
                label_cun = []
                for j in range(pred_final.shape[1]):
                    where_is = np.where(label_final[:,j]==1)[0]
                    pred_cun.append(pred_final[where_is,j])
                    label_cun.append(label_final[where_is,j]*label_final[where_is,-1])
                roc_auc = [ roc_auc_score(label_cun[l], pred_cun[l]) for l in range(pred_final.shape[1])]
                prc_auc = [ average_precision_score(label_cun[l], pred_cun[l]) for l in range(pred_final.shape[1])]
                ap =  [accuracy_score(label_cun[l], (pred_cun[l] > 0.5).astype('float')) for l in range(pred_final.shape[1])]

                results['PR-AUC'] = np.array(prc_auc).mean()
                results['AUC-ROC'] = np.array(roc_auc).mean()
                results['accuracy'] = np.array(ap).mean()
                str_record = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' {} [Epoch {} {}]: PR-AUC : {:.5},  AUC-ROC: {:.5}, Accuracy : {:.5}\n'.format(split ,epoch, split, results['PR-AUC'], results['AUC-ROC'], results['accuracy'])

        return results, str_record

    def update_result(self, results):
        for j in results:
            if results[j]['accuracy'] > self.best_val_acc[j]:
                self.best_val_acc[j] = results[j]['accuracy']
                self.no_update_epoch[j] = 0
                self.save_model(self.save_path + j[-3:])
            else:
                self.no_update_epoch[j] += 1
        for j in self.no_update_epoch:
            if self.no_update_epoch[j] <= 20:
                return 0
        return 1

    def save_model(self, save_path):
        state = {
			'state_dict'	: self.model.state_dict(),
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.args)
		}
        torch.save(state, save_path)

    def load_model(self, load_path):
        # print(torch.cuda.device_count())
        state			= torch.load(load_path, map_location = self.device)
        state_dict		= state['state_dict']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

        # with torch.no_grad():
        #     results = {}
        #     train_iter = iter(self.data_record.data_iter['{}_{}'.format(split, mode.split('_')[0])])

        #     for step, batch in enumerate(train_iter):
        #         sub, obj, rel, label	= self.read_batch(batch, split) # rel -> obj, obj -> rel
        #         pred			= self.model.forward(sub, obj)
        #         b_range			= torch.arange(pred.size()[0], device=self.device)
        #         target_pred		= pred[b_range, rel] ### obtain the corresponding place in the prediction (of)
        #         pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
        #         pred[b_range, rel] 	= target_pred
        #         ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, rel]
        #         ranks 			= ranks.float()
        #         results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
        #         results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
        #         results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
        #         for k in range(10):
        #             results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

        #         if step % 100 == 0:
        #             print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()) + ' [{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.args.name))
        # return results
