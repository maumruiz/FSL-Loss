import os.path as osp
import json
import torch
import pandas as pd
import shutil
import numpy as np

class ExpLogger():
    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.max_acc = 0.0
        self.max_acc_epoch = 0
        self.test_acc = []
        self.mean_acc = 0.0
        self.parameters = 0
        self.elapsed_time = ''
        self.query_predictions = []
        self.query_ids = []
        self.query_labels = []
        self.logits = []
        self.support_ids = []
        self.support_labels = []
    
    def _to_obj(self):
        log = {}
        
        arg_keys = ['dataset', 'model', 'modules', 'backbone', 'way', 'train_way', 
        'shot', 'query', 'max_epoch', 'train_epi', 'val_epi', 'test_epi', 'optimizer', 'lr', 'step_size',
        'gamma', 'temperature', 'init_weights', 'step_size', 'save_path', 'seed', 'tag']

        log['args'] = {k: self.args[k] for k in arg_keys}
        log['train_loss'] = self.train_loss
        log['train_acc'] = self.train_acc
        log['val_loss'] = self.val_loss
        log['val_acc'] = self.val_acc
        log['max_acc'] = self.max_acc
        log['max_acc_epoch'] = self.max_acc_epoch
        log['test_acc'] = self.test_acc
        log['mean_acc'] = self.mean_acc
        log['parameters'] = self.parameters
        log['elapsed_time'] = self.elapsed_time
        return log
    
    def save(self, path):
        log = self._to_obj()
        torch.save(log, osp.join(path, 'experimentObj'))

    def save_json(self, path):
        log = self._to_obj()

        with open(osp.join(path, 'experiment.json'), "w") as write_file:
            json.dump(log, write_file, indent=4)

    def save_trainval(self, path):
        trainval_df = pd.DataFrame()
        trainval_df['epoch'] = [e+1 for e in range(len(self.train_loss))]
        trainval_df['train_loss'] = self.train_loss
        trainval_df['train_acc'] = self.train_acc
        trainval_df['val_loss'] = self.val_loss
        trainval_df['val_acc'] = self.val_acc
        trainval_df.to_csv(osp.join(path, 'trainval.csv'), index=False)

    def save_test(self, path):
        test_df = pd.DataFrame()
        test_df['batch'] = [e+1 for e in range(len(self.test_acc))]
        test_df['acc'] = self.test_acc
        test_df.to_csv(osp.join(path, 'test.csv'), index=False)

    def save_results(self, path):       
        results_df = pd.DataFrame()
        results_df['tag'] = [self.args['tag']]
        results_df['max_val_acc'] = [self.max_acc]
        results_df['test_acc'] = [self.mean_acc]
        results_df['optimizer'] = [self.args['optimizer']]
        results_df['lr'] = [self.args['lr']]
        results_df['step_size'] = [self.args['step_size']]
        results_df['gamma'] = [self.args['gamma']]
        results_df['temperature'] = [self.args['temperature']]
        results_df['num_parameters'] = [self.parameters]
        results_df['elapsed_time'] = [self.elapsed_time]
        results_df['train_epochs'] = [self.args['max_epoch']]
        results_df['train_episodes'] = [self.args['train_epi']]
        results_df['val_episodes'] = [self.args['val_epi']]
        results_df['test_episodes'] = [self.args['test_epi']]
        results_df['init_weights'] = [self.args['init_weights']]
        results_df['save_path'] = [self.args['save_path']]
        results_df['seed'] = [self.args['seed']]
        results_df['way'] = [self.args['way']]
        results_df['shot'] = [self.args['shot']]
        results_df['queries'] = [self.args['query']]
        results_df['train_way'] = [self.args['train_way']]
        results_df['name'] = [f"{self.args['dataset']}-{self.args['model']}-{self.args['backbone']}"]
        
        results_df.to_csv(osp.join(path, 'results.csv'), index=False)

    def save_features(self, path):
        # ICN save different sizes of features
        max_dim = max([x.size(1) for x in self.args['features']])
        features = torch.cat(
            [torch.cat(
                (feature, torch.full((feature.size(0), max_dim - feature.size(1)), -1, dtype=torch.float)), axis=1) 
            for feature in self.args['features']]
        )

        fts_df = pd.DataFrame(features.numpy())
        fts_df['label'] = torch.cat(self.args['fts_labels'], dim=0).numpy()
        fts_df['img_id'] = [id for ids in self.args['fts_ids'] for id in ids]
        cols = list(fts_df)
        cols = [cols[-1], cols[-2]] + cols[:-2]
        fts_df = fts_df[cols]
        fts_df.to_csv(osp.join(path, 'features.csv'), index=False)

    def save_logits(self, path):
        logits_df = pd.DataFrame()
        logits_df['label'] = self.query_labels
        logits_df['prediction'] = self.query_predictions
        logits_df['query_id'] = self.query_ids

        logits_t = torch.Tensor(self.logits).T
        for i in range(self.args['way']):
            logits_df[f'logits_proto_{i+1}'] = logits_t[i]

        supp_labels = np.repeat(self.support_labels, self.args['way'] * self.args['query'], axis=0)
        supp_labels_t = supp_labels.T
        for i in range(self.args['way']):
            for j in range(self.args['shot']):
                logits_df[f'support_label_category{i+1}_img{j+1}'] = supp_labels_t[i * j + j]

        supp_ids = np.repeat(self.support_ids, self.args['way'] * self.args['query'], axis=0)
        supp_ids_t = supp_ids.T
        for i in range(self.args['way']):
            for j in range(self.args['shot']):
                logits_df[f'support_id_category{i+1}_img{j+1}'] = supp_ids_t[i * j + j]

        logits_df.to_csv(osp.join(path, 'logits.csv'), index=False)

    def save_model(self, path, name):
        if not osp.exists(f'pretrained/{name}.pth'):
            shutil.copy(f'{path}/max_acc.pth', f'pretrained/{name}.pth')
        else:
            ii = 2
            new_name = f'pretrained/{name}{ii}.pth'
            while osp.exists(new_name):
                ii += 1
                new_name = f'pretrained/{name}{ii}.pth'
            shutil.copy(f'{path}/max_acc.pth', new_name)
            


