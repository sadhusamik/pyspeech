#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:46:38 2018

@author: lucas ondel, minor changes by samik sadhu 
"""

'Prepare data, train MLP and do cross validation'


import argparse
import numpy as np
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from os import listdir
import os 
from os.path import isfile, join
import sys

def tidyData(data_dir):
    
    print('%s: Checking for train and test data...' % sys.argv[0])
    allfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    print('%s: In total %d train and test data files found..' % (sys.argv[0],len(allfiles)))
    train_files=[]; test_files=[]
    
    for i in range(len(allfiles)):
        if 'test' in allfiles[i]:
            test_files.append(allfiles[i])
            
    for i in range(len(allfiles)):
        if 'train' in allfiles[i]:
            train_files.append(allfiles[i])
    
    # Check the data dimension
    data=np.load(os.path.join(data_dir, train_files[0]))
    data_dim=data.shape[1]-1
    
    # Load all train and test data into big files 
    train_data=np.empty((0,data_dim)); test_data=np.empty((0,data_dim)); train_labels=np.array([]); test_labels=np.array([])
    
    print('%s: Loading training files...' % sys.argv[0])
    for i in range(len(train_files)):
        data=np.load(os.path.join(data_dir, train_files[i]))
        train_data=np.append(train_data,data[:,0:-1],axis=0)
        train_labels=np.append(train_labels,data[:,-1])
    
    print('%s: Loading test files...' % sys.argv[0])
    for i in range(len(test_files)):
        data=np.load(os.path.join(data_dir, test_files[i]))
        test_data=np.append(test_data,data[:,0:-1],axis=0)
        test_labels=np.append(test_labels,data[:,-1])
    
    return train_data, train_labels, test_data, test_labels

def error_rate(model, features, labels, loss_fn):
    outputs = model(features)
    #np.save('./test_output', outputs)
    #np.save('./test_labels', labels)
            
    loss = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)
    
    #lab=labels.data.numpy()
    #pred=predicted.data.numpy()
    #class_counts=np.unique(lab,return_counts=True) 
    #class_counts=class_counts[1]
    #phn_count=np.zeros(38)
    #for i in range(len(pred)):
    #    if pred[i]!=lab[i]:
    #        phn_count[int(lab[i])]+=1
    #np.save('./class_based_errors.npy',phn_count)  
    #np.save('./class_counts.npy',class_counts)        
    hits = (labels == predicted).float().sum()
    return loss.data[0], (1 - hits / labels.size(0)).data[0]


def run(train_data, train_labels, test_data, test_labels, args):

    if args.activation=='sigmoid':
        activ=nn.Sigmoid()
    elif args.activation=='tanh':
        activ=nn.Tanh()
    elif args.activation=='relu':
        activ=nn.ReLU()
    else:
        raise ValueError('Activation function not found!')

    if args.mvnorm:
        mean = train_data.mean(axis=0)
        var = train_data.var(axis=0)
        train_data -= mean
        train_data /= np.sqrt(var)
        test_data -= mean
        test_data /= np.sqrt(var)

    #print(np.shape(train_data))
    # Input/output dimension of the MLP.
    feadim, targetdim = train_data.shape[1], args.ntargets

    # Build the MLP.
    
    if args.put_kink:
        
        structure = [nn.Linear(feadim, 64), activ]
        for i in range(args.nlayers - 1):
            if i==0:
                structure += [nn.Linear(64, args.nunits), activ]
            else:
                structure += [nn.Linear(args.nunits, args.nunits), activ]
        structure += [nn.Linear(args.nunits, targetdim)]
        model = nn.Sequential(*structure)
    
    else:
        
        structure = [nn.Linear(feadim, args.nunits), activ]
        for i in range(args.nlayers - 1):
            structure += [nn.Linear(args.nunits, args.nunits), activ]
        structure += [nn.Linear(args.nunits, targetdim)]
        model = nn.Sequential(*structure)
    
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            model.cuda(args.gpu)
                  
    
    # Loss function.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate,
                                 weight_decay=args.weight_decay)

    train_data, train_labels = torch.from_numpy(train_data).float(), \
        torch.from_numpy(train_labels).long()
    test_data, test_labels = torch.from_numpy(test_data).float(), \
        torch.from_numpy(test_labels).long()
        
    #v_train_data, v_train_labels = Variable(train_data), Variable(train_labels)
    
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            v_test_data, v_test_labels = Variable(test_data).cuda(), Variable(test_labels).cuda()
    else:
        v_test_data, v_test_labels = Variable(test_data), Variable(test_labels)

    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                              shuffle=True)

    test_err_save=np.empty(0)
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            for epoch in range(args.epochs):
                t_loss = 0.0
                t_er = 0.0
                for i, data in enumerate(trainloader):
                    
                    inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
        
                    # Compute the error rate on the training set.
                    _, predicted = torch.max(outputs, dim=1)
                    hits = (labels == predicted).float().sum()
                    t_er += (1 - hits / labels.size(0)).data[0]
                    t_loss += loss.data[0]
        
                    loss.backward()
                    optimizer.step()
        
                    if i % args.validation_rate == args.validation_rate - 1:
                        t_loss /= args.validation_rate
                        t_er /= args.validation_rate
                        cv_loss, cv_er = error_rate(model, v_test_data, v_test_labels, loss_fn)
                        logmsg = 'epoch: {epoch}  mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
                                 'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                                 'error rate (cv): {cv_er:.3%}'.format(
                                 epoch=epoch+1, mbatch=i+1, t_loss=t_loss, t_er=t_er,
                                 cv_loss=cv_loss, cv_er=cv_er)
                        test_err_save=np.append(test_err_save,cv_er)
                        t_er = 0.0
                        t_loss = 0.0
                        print(logmsg)
            model=model.cpu()
            #np.save('./test_error_plot.npy',test_err_save)
            with open(args.outmodel, 'wb') as fid:
                pickle.dump(model, fid)
    else:
        
        for epoch in range(args.epochs):
                t_loss = 0.0
                t_er = 0.0
                for i, data in enumerate(trainloader):
                    
                    inputs, labels = Variable(data[0]), Variable(data[1])
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
        
                    # Compute the error rate on the training set.
                    _, predicted = torch.max(outputs, dim=1)
                    hits = (labels == predicted).float().sum()
                    t_er += (1 - hits / labels.size(0)).data[0]
                    t_loss += loss.data[0]
        
                    loss.backward()
                    optimizer.step()
        
                    if i % args.validation_rate == args.validation_rate - 1:
                        t_loss /= args.validation_rate
                        t_er /= args.validation_rate
                        cv_loss, cv_er = error_rate(model, v_test_data, v_test_labels, loss_fn)
                        logmsg = 'epoch: {epoch}  mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
                                 'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                                 'error rate (cv): {cv_er:.3%}'.format(
                                 epoch=epoch+1, mbatch=i+1, t_loss=t_loss, t_er=t_er,
                                 cv_loss=cv_loss, cv_er=cv_er)
                        test_err_save=np.append(test_err_save,cv_er)
                        t_er = 0.0
                        t_loss = 0.0
                        print(logmsg)
        
        
        np.save('./test_error_plot.npy',test_err_save)
        with open(args.outmodel, 'wb') as fid:
                pickle.dump(model, fid)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_directory', help='place to get all training and test data in .npy format')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--ntargets', type=int, help='number of targets')
    parser.add_argument('--nlayers', type=int, help='number of hidden layers')
    parser.add_argument('--nunits', type=int, help='number of units per leayer')
    parser.add_argument('--gpu',type=int,help='gpu device id (Ignore if you do not want to run on gpu!)')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--mvnorm', action='store_true',
                        help='mean-variance normalization of the features')
    parser.add_argument('--validation-rate', type=int, default=10,
                        help='frequency of the validation')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization')
    parser.add_argument('--activation', default='tanh',
                        help='tanh OR sigmoid OR relu')
    parser.add_argument('--put_kink', help='Puts a 64 dimension layer at the beginning to plot filters',action='store_true')
    args = parser.parse_args()

    assert args.nlayers > 0

    print('%s: Running  MLP training...' % sys.argv[0])
    
    train_data=np.load(join(args.data_directory,'train_data.npy'))
    train_labels=np.load(join(args.data_directory,'train_labels.npy'))
    test_data=np.load(join(args.data_directory,'test_data.npy'))
    test_labels=np.load(join(args.data_directory,'test_labels.npy'))
    
    run(train_data, train_labels, test_data, test_labels,args)
