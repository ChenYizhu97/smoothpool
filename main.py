#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from tqdm import tqdm
import argparse
import time
import numpy as np
from gnn import GNN

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCELoss()
reg_criterion = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        link_loss, ent_loss = None, None

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if model.hpool_archi == "diffpool":
                pred, link_loss, ent_loss  = model(batch)
            else:
                pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            
            if link_loss and ent_loss:
                loss = loss + link_loss + ent_loss
            # print(loss)
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                if model.hpool_archi == "diffpool":
                    pred, link_loss, ent_loss = model(batch)
                else:
                    pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--hpool', type=str, default="topk",
                        help='hpool architecture')
    parser.add_argument('--patience', type=float, default=0.2,
                        help='patience for early stopping')
    parser.add_argument('--cpu', type=bool, default=False,
                        help='use cpu for training')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.cpu:
        device = torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    patience = int(args.patience * args.epochs)
    
    # get C for pooling
    num_of_nodes = 0
    for step, data in enumerate(dataset):
        num_of_nodes += data.x.size()[0]
    step += 1
    average_num_of_nodes = num_of_nodes/step

    C = int(average_num_of_nodes*0.5)
    
    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    hpool = args.hpool
    print("number of clusters is {}, pooling method is {}".format(C,hpool))
    model = GNN(300,dataset.num_tasks,AtomEncoder(emb_dim=300), BondEncoder(emb_dim=300), hpool=hpool, C=C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    current_best_val_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if 'classification' in dataset.task_type:
            current_best_val_epoch = np.argmax(np.array(valid_curve))
        else:
            current_best_val_epoch = np.argmin(np.array(valid_curve))
        if epoch > current_best_val_epoch + patience:
            break
    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Total epoch run: {}'.format(epoch))
    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train, 'Epoch run': epoch}, args.filename)


if __name__ == "__main__":
    main()
