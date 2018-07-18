#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 23:33:03 2018

@author: samiksadhu
"""

'Prepare data, train vanilla nnet and do cross validation using batch loading of data'

import sys
sys.path.append('/export/b15/ssadhu/pyspeech/src/featgen/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/utils/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/nnet/')

from gen_utils import get_dim
from nnet import get_device_id, print_log, model_err
import argparse
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from os.path import join, dirname


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('egs_dir', help='Example data directory')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--ntargets', type=int, default=48, help='number of targets(48)')
    parser.add_argument('--nlayers', type=int, default=4, help='number of hidden layers(4)')
    parser.add_argument('--nunits', type=int, default=256, help='number of units per leayer(256)')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--split_num', type=int, help='number of splits of the data(5)', default=5)
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='L2 regularization')
    parser.add_argument('--cv_stop', type=int,
                        help='Stop after this many increases of CV error')
    parser.add_argument('--activation', default='tanh',
                        help='tanh OR sigmoid OR relu')
    return parser.parse_args()

def error_rate(model, features, labels, loss_fn):
    outputs = model(features)         
    loss_test = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)      
    hits = (labels == predicted).float().sum()
    return loss_test.data[0], (1 - hits / labels.size(0)).data[0]

def define_model(dim,nunits,nlayers,activ,targetdim,device_id):

    structure = [nn.Linear(dim, nunits), activ]
    
    for i in range(nlayers - 1):
        structure += [nn.Linear(nunits, nunits), activ]
        
    structure += [nn.Linear(nunits, targetdim)]
    model = nn.Sequential(*structure)

    if device_id!=-1:
        with torch.cuda.device(device_id):
            model.cuda(device_id)
        
    return model 

def train(model,egs_dir,split_num,epochs,gpu_id,cv_stop,lrate,weight_decay,bsize,outmodel):
    
    # Loss function.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate,
                                 weight_decay=weight_decay)
    
        
    if gpu_id!=-1:
        with torch.cuda.device(gpu_id):            
            cv_er_old=0
            warn_time=0
            for epoch in range(epochs):
                t_loss = 0.0
                t_er = 0.0
                batch_count=0
                for batch in range(1,split_num+1):
                    
                    train_data=pickle.load(open(join(egs_dir,'train','data.'+str(batch)+'.egs'),'rb'))
                    train_labels=pickle.load(open(join(egs_dir,'train','labels.'+str(batch)+'.egs'),'rb'))
                    
                    train_data, train_labels = torch.from_numpy(train_data).float().cuda(), \
                    torch.from_numpy(train_labels.flatten()-1).long().cuda()
                    
                    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                    trainloader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                              shuffle=True)
                    
                    for i, data in enumerate(trainloader):
                    
                        inputs, labels = Variable(data[0]), Variable(data[1]).cuda()
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
            
                        # Compute the error rate on the training set.
                        
                        _, predicted = torch.max(outputs, dim=1)
                        hits = (labels == predicted).float().sum()
                        t_er += (1 - hits / labels.size(0)).data[0]
                        t_loss += loss.data[0]
                        batch_count+=1
                        
                        loss.backward()
                        optimizer.step()
                
                # print the loss after every epoch 
                
                t_loss /= batch_count
                t_er /= batch_count
                
                cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
                
                logmsg = '# epoch: {epoch} loss (train): {t_loss:.3f}  ' \
                         'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                         'error rate (cv): {cv_er:.3%}'.format(epoch=epoch+1, t_loss=t_loss, t_er=t_er, cv_loss=cv_loss, cv_er=cv_er)
                t_er = 0.0
                t_loss = 0.0
                print(logmsg)
                sys.stdout.flush()
                        
                if cv_er>cv_er_old:
                    warn_time+=1
                    cv_er_old=cv_er
                            
                if warn_time>=cv_stop:
                    print('%s: Cross Validation Error found to increase in 2 epochs.. exiting with present model!' % sys.argv[0])
                    cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
                    print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],cv_er*100))
                    break
            
            print('%s: Maximum number of epochs exceeded!' % sys.argv[0])
            cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
            print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],cv_er*100))
            model=model.cpu()
            
        
        res_file=join(dirname(outmodel),'result')
            
        with open(res_file,'w') as f:
            f.write('Test set Frame Error Rate: %.2f %%' % (cv_er*100))
            
        with open(outmodel, 'wb') as fid:
            pickle.dump(model, fid)
                    
    else:
        cv_er_old=0
        warn_time=0
        for epoch in range(epochs):
            t_loss = 0.0
            t_er = 0.0
            batch_count=0
            for batch in range(1,split_num+1):
                train_data=pickle.load(open(join(egs_dir,'train','data.'+str(batch)+'.egs'),'rb'))
                train_labels=pickle.load(open(join(egs_dir,'train','labels.'+str(batch)+'.egs'),'rb'))
                
                train_data, train_labels = torch.from_numpy(train_data).float(), \
                torch.from_numpy(train_labels.flatten()-1).long()
                
                dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                trainloader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                          shuffle=True)
                
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
                    batch_count+=1
                    
                    loss.backward()
                    optimizer.step()
            
            # print the loss after every epoch 
            
            t_loss /= batch_count
            t_er /= batch_count
            
            cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
            
            logmsg = '# epoch: {epoch} loss (train): {t_loss:.3f}  ' \
                     'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                     'error rate (cv): {cv_er:.3%}'.format(epoch=epoch+1, t_loss=t_loss, t_er=t_er, cv_loss=cv_loss, cv_er=cv_er)
            t_er = 0.0
            t_loss = 0.0
            print(logmsg)
            sys.stdout.flush()
                    
            if cv_er>cv_er_old:
                warn_time+=1
                cv_er_old=cv_er
                        
            if warn_time>=cv_stop:
                print('%s: Cross Validation Error found to increase in 2 epochs.. exiting with present model!' % sys.argv[0])
                cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
                print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],cv_er*100))
                break
        
        print('%s: Maximum number of epochs exceeded!' % sys.argv[0])
        cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
        print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],cv_er*100))
        
        res_file=join(dirname(outmodel),'result')
        
        with open(res_file,'w') as f:
            f.write('Test set Frame Error Rate: %.2f %%' % (cv_er*100))
            
        with open(outmodel, 'wb') as fid:
                pickle.dump(model, fid)
            
             
if __name__=='__main__':
    
    args=get_args()
    gpu_id=get_device_id()
    
    if gpu_id!=-1:
        print('%s: Using GPU device %d for nnet' % (sys.argv[0],gpu_id))
    else:
        print_log('Training nnet   on sinlge CPU, this will take some time!')
    # Activation Function
    
    if args.activation=='sigmoid':
        activ=nn.Sigmoid()
    elif args.activation=='tanh':
        activ=nn.Tanh()
    elif args.activation=='relu':
        activ=nn.ReLU()
    else:
        sys.exit('%s: The activation function %s is invalid, exiting script!' % (sys.argv[0],args.activation))
    
    print_log('Defining nnet model')
    
    with open(join(args.egs_dir,'dim'),'r') as fid:
        dim=int(fid.readline())
   
    model=define_model(dim,args.nunits,args.nlayers,activ,args.ntargets,gpu_id)
    
    print_log('Training nnet model')
    
    # Main Training function
    train(model,args.egs_dir,args.split_num,args.epochs,gpu_id,args.cv_stop,args.lrate,args.weight_decay,args.bsize,args.outmodel)