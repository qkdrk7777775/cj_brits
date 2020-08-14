import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os 
os.chdir(r'C:/Users/cjcho/Desktop/git/cj_brits')
import numpy as np

import time
import utils
import sys
sys.path.append(r"./models")
import models
import argparse
import data_loader
import pandas as pd
import ujson as json
from sklearn import metrics
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 1000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--model', type = str,default='brits')
args = parser.parse_args()

def collate_fn(recs):
    forward  = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))
    
    def to_tensor_dict(recs):
        values     =torch.FloatTensor(list(map(lambda r:list(map(lambda x:x['values'],r)), recs)))
        masks      =torch.FloatTensor(list(map(lambda r:list(map(lambda x:x['masks'], r)), recs)))
        deltas     =torch.FloatTensor(list(map(lambda r:list(map(lambda x:x['deltas'],r)), recs)))
        forwards   =torch.FloatTensor(list(map(lambda r:list(map(lambda x:x['forwards'],r)), recs)))
        evals      =torch.FloatTensor(list(map(lambda r:list(map(lambda x:x['evals'],r)), recs)))
        eval_masks =torch.FloatTensor(list(map(lambda r:list(map(lambda x:x['eval_masks'],r)), recs)))
        return {'values':values,'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks} 
    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}
    ret_dict['labels']   = torch.FloatTensor(list(map(lambda x: x['label']   , recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))
    return ret_dict

def train(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    data_set=data_loader.MySet()
    data_iter=torch.utils.data.DataLoader(data_set,batch_size=args.batch_size,collate_fn=collate_fn,shuffle=True,pin_memory=True)
#    data_iter = data_loader.get_loader(batch_size = args.batch_size)
    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0
        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)
            run_loss += ret['loss'].data

            print ('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))
        if epoch % 1 == 0:
            evaluate(model, data_iter)

def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    print ('AUC {}'.format(metrics.roc_auc_score(labels, preds)))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print ('MAE', np.abs(evals - imputations).mean())
    print ('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

def run():
    model = getattr(models, args.model).Model()

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)

if __name__ == '__main__':
    run()
