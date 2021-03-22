from collections import defaultdict
from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score
import numpy as np
import json

class SummaryRecorder:
    def __init__(self , num_folds , num_epochs):
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.metrics = ["loss" , "accuracy" , "precision" , "f1" , "recall"]
        self.history = {
            "train" : [[ defaultdict(list) for _ in range(num_epochs) ] for _ in range(num_folds)],
            "eval" : [[ defaultdict(list) for _ in range(num_epochs) ] for _ in range(num_folds)]
        }
        self.folds_history = {
            "train": [defaultdict(list)  for _ in range(num_folds)],
            "eval": [defaultdict(list)  for _ in range(num_folds)]
        }
        self.final_history = {
            "train" : defaultdict(list),
            "eval" : defaultdict(list)
        }
        self.raw = True
        self.summary = None

    def add_evaluation(self , mode , fold_id , epoch_id , loss_val , y_true , y_pred):

        self.history[mode][fold_id][epoch_id]['loss'].append(loss_val)
        self.history[mode][fold_id][epoch_id]['accuracy'].append(accuracy_score(y_true, y_pred))
        self.history[mode][fold_id][epoch_id]['precision'].append(precision_score(y_true, y_pred))
        self.history[mode][fold_id][epoch_id]['f1'].append(f1_score(y_true, y_pred))
        self.history[mode][fold_id][epoch_id]['recall'].append(recall_score(y_true, y_pred))

    def aggregate_batches(self):
        for mode in self.history:
            for fold in range(self.num_folds):
                for epoch in range(self.num_epochs):
                    for metric in self.history[mode][fold][epoch]:
                        self.history[mode][fold][epoch][metric] = np.mean(self.history[mode][fold][epoch][metric])

    def aggregate_epochs(self , main_metric = 'f1'):
        for mode in self.history:
            for fold in range(self.num_folds):

                values = []
                for epoch in range(self.num_epochs):
                    values.append(self.history[mode][fold][epoch][main_metric])

                best_epoch = max(enumerate(values) , key = lambda  x : x[1])[0]

                for metric in self.metrics:
                    self.folds_history[mode][fold][metric] = self.history[mode][fold][best_epoch][metric]


    def aggregate_folds(self):
        for mode in self.history:
            for metric in self.metrics:
                for fold in range(self.num_folds):
                    self.final_history[mode][metric].append(self.folds_history[mode][fold][metric])
                self.final_history[mode][metric] = np.mean(self.final_history[mode][metric])

    def generate_summary(self):
        if self.raw:
            self.raw = False
            self.aggregate_batches()
            self.aggregate_epochs()
            self.aggregate_folds()
            self.summary = json.dumps(self.final_history , indent=1)
        return self.summary
