import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import  cohen_kappa_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def print_attn_weight(self):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                s = r_labels_pos.cpu().numpy().tolist()

                if 19 in s:
                    print(s, targets_pos)
                    score_pos = self.graph_classifier(data_pos)
                    s = score_pos.detach().cpu().numpy()
                    # with open('Drugbank/result.txt', 'a') as f:
                    #     f.write()

    def print_result(self):
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        pos_labels = []
        pos_argscores = []
        pos_scores = []
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)
                label_ids = r_labels_pos.to('cpu').numpy()
                pos_labels += label_ids.flatten().tolist()
                pos_argscores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist() 
                print( torch.max(score_pos, dim=1, out=None))
                pos_scores += torch.max(score_pos, dim=1)[0].cpu().flatten().tolist() 

        with open('Drugbank/results.txt', 'w') as f:
            for (x,y,z) in zip(pos_argscores, pos_labels, pos_scores):
                f.write('%d %d %d\n'%(x, y, z))


    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        y_pred = []
        label_matrix = []
        accuracy_class = {}
        
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                score_pos = self.graph_classifier(data_pos)
                label_ids = r_labels_pos.to('cpu').numpy()
                pos_labels += label_ids.flatten().tolist()
                pos_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist()

        pos_labels = np.array(pos_labels)
        pos_scores = np.array(pos_scores)

        acc = metrics.accuracy_score(pos_labels, pos_scores)
        auc = metrics.f1_score(pos_labels, pos_scores, average='macro')
        auc_pr = metrics.f1_score(pos_labels, pos_scores, average='micro')
        f1 = metrics.f1_score(pos_labels, pos_scores, average=None)
        kappa = metrics.cohen_kappa_score(pos_labels, pos_scores)

        for relation_type, relation_id in self.params.relation_class.items():
            binary_pos_labels = np.isin(pos_labels, relation_id).astype(int)
            indices = np.where(binary_pos_labels == 1)[0]
            pos_labels_new = pos_labels[indices]
            pos_scores_new = pos_scores[indices]
            accuracy_class[relation_type] = accuracy_score(pos_labels_new, pos_scores_new)

        class_report = classification_report(pos_labels, pos_scores, labels=range(86))
        cm = confusion_matrix(pos_labels, pos_scores, labels=range(86))
        class_accuracies = np.diagonal(cm) / cm.sum(axis=1)

        return {'auc': auc, 'microf1': auc_pr, 'acc': acc, 'k': kappa}, {'acc': accuracy_class}, class_report, class_accuracies

class Evaluator_ddi2():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []

        y_pred = []
        y_label = []
        outputs = []

        pred_class = {}

        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):

                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()
                score_pos = self.graph_classifier(data_pos)

                m = nn.Sigmoid()
                #loss_fct = nn.BCELoss()
                pred = m(score_pos)
                #loss = loss_fct(pred, label)
                labels = r_labels_pos.detach().to('cpu').numpy() # batch * 200
                preds = pred.detach().to('cpu').numpy() # batch * 200
                targets_pos = targets_pos.detach().to('cpu').numpy()
                for (label_ids, pred, label_t) in zip(labels, preds, targets_pos):
                # label_ids = [x for x in label.detach().to('cpu').numpy() if x==1] # batch * 200
                # preds = pred.detach().to('cpu').numpy()[label_ids] # batch * 200
                    #print(label_ids, pred, label_t)
                    for i, (l, p) in enumerate(zip(label_ids, pred)):
                        #print(i, l, p)
                        if l == 1:
                            if i in pred_class:
                                pred_class[i]['pred'] += [p]
                                pred_class[i]['l'] += [label_t] 
                                pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                            else:
                                pred_class[i] = {'pred':[p], 'l':[label_t], 'pred_label':[1 if p > 0.5 else 0]}

        roc_auc = [ roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
        prc_auc = [ average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
        ap =  [accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]

        return {'auc': np.mean(roc_auc), 'auc_pr': np.mean(prc_auc), 'f1': np.mean(ap)}, {"auc_all":roc_auc,"aupr_all":prc_auc, "f1_all":ap}

