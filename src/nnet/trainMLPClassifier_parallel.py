#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:09:56 2018

@author: samiksadhu
"""

'Multi-stream DNN Training for phoneme classification'

from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
import torch
import torch.utils.data
from os import listdir
import os 
from os.path import isfile, join
import argparse

class Model(nn.Module):
    
    def __init__(self, nstreams, coeff_num,num_classes,nlayers_ms,nunits_ms,nlayers_comb,nunits_comb):
        super().__init__()
        self.nstreams = nstreams
        self.ncoeff=coeff_num
        
        # Define the multi-stream network structure 
        structure_ms = [nn.Linear(coeff_num, nunits_ms), nn.Tanh()]
        for i in range(nlayers_ms):
            structure_ms += [nn.Linear(nunits_ms, nunits_ms), nn.Tanh()]
        #structure_ms += [nn.Linear(nunits_ms, num_classes)]
        
        for i in range(nstreams):
            vname = 'hs_ms' + str(i)
            setattr(self, vname, nn.Sequential(*structure_ms))
            
        #Define the Combination Network Structure 
        
        structure_comb = [nn.Linear(nunits_ms*nstreams, nunits_comb), nn.Tanh()]
        for i in range(nlayers_comb - 1):
            structure_comb += [nn.Linear(nunits_comb, nunits_comb), nn.Tanh()]
        structure_comb += [nn.Linear(nunits_comb, num_classes)]
        
        self.hs_comb=nn.Sequential(*structure_comb)
        
    def forward(self,x):
        coeff_num=self.ncoeff
        
        start=coeff_num*0;
        inp=x[:,start:start+coeff_num]
        out=self.hs_ms0(inp)
        out1=F.tanh(out)
        
        start=coeff_num*1;
        inp=x[:,start:start+coeff_num]
        out=self.hs_ms1(inp)
        out2=F.tanh(out)
        
        join_hidden=torch.cat([out1,out2],1)
        
        for i in range(2,self.nstreams):
            vname='hs_ms' + str(i)
            hs = getattr(self, vname)
        
            start=coeff_num*i;
            inp=x[:,start:start+coeff_num]
            out=hs(inp)
            out=F.tanh(out)
            join_hidden=torch.cat([join_hidden,out],1)
        
        out=self.hs_comb(join_hidden)
        #out=F.softmax(out)
        return out

def tidyData(data_dir):
    allfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    
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
    
    for i in range(len(train_files)):
        data=np.load(os.path.join(data_dir, train_files[i]))
        train_data=np.append(train_data,data[:,0:-1],axis=0)
        train_labels=np.append(train_labels,data[:,-1])
    
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
    
    lab=labels.data.numpy()
    pred=predicted.data.numpy()
    class_counts=np.unique(lab,return_counts=True) 
    class_counts=class_counts[1]
    phn_count=np.zeros(38)
    for i in range(len(pred)):
        if pred[i]!=lab[i]:
            phn_count[int(lab[i])]+=1
    np.save('./class_based_errors.npy',phn_count)  
    np.save('./class_counts.npy',class_counts)        
    hits = (labels == predicted).float().sum()
    return loss.data[0], (1 - hits / labels.size(0)).data[0]

def run(train_data, train_labels, test_data, test_labels, args):
    
    #if args.mvnorm:
    mean = train_data.mean(axis=0)
    var = train_data.var(axis=0)
    train_data -= mean
    train_data /= np.sqrt(var)
    test_data -= mean
    test_data /= np.sqrt(var)
    
    
    
    model=Model(coeff_num=args.ncoeff,num_classes=args.ntargets,nstreams=args.nstreams,nlayers_ms=args.nlayers_ms,nunits_ms=args.nunits_ms,nlayers_comb=args.nlayers_comb,nunits_comb=args.nunits_comb)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate,
                                 weight_decay=args.weight_decay)

    train_data, train_labels = torch.from_numpy(train_data).float(), \
        torch.from_numpy(train_labels).long()
    test_data, test_labels = torch.from_numpy(test_data).float(), \
        torch.from_numpy(test_labels).long()
        
    #v_train_data, v_train_labels = Variable(train_data), Variable(train_labels)
    v_test_data, v_test_labels = Variable(test_data), Variable(test_labels)

    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                              shuffle=True)
    
    test_err_save=np.empty(0)
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
    with open(args.outmodel, 'wb') as fid:
        pickle.dump(model, fid)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_directory', help='place to get all training and test data in .npy format')
    parser.add_argument('ntargets', type=int, help='number of targets')
    parser.add_argument('nlayers_ms', type=int, help='number of hidden layers for streams')
    parser.add_argument('nunits_ms', type=int, help='number of units per layer in streams')
    parser.add_argument('nlayers_comb', type=int, help='number of hidden layers for combination')
    parser.add_argument('nunits_comb', type=int, help='number of units per layer in combination')
    parser.add_argument('ncoeff', type=int, help='number of FDLP Modulation Coefficients')
    parser.add_argument('nstreams', type=int, help='number of streams')
    parser.add_argument('outmodel', help='output file')
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
    args = parser.parse_args()

    
    #parser.add_argument('nlayers', type=int, help='number of hidden layers')
    #parser.add_argument('nunits', type=int, help='number of units per leayer')
    train_data, train_labels, test_data, test_labels=tidyData(args.data_directory)
    run(train_data, train_labels, test_data, test_labels,args)