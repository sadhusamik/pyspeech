#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:01:39 2018

@author: samiksadhu
"""
'Train CNN nnet with pytorch'

import sys
sys.path.append('/export/b15/ssadhu/pyspeech/src/featgen/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/utils/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/nnet/')

from gen_utils import get_dim
from nnet import get_device_id, print_log, model_err
import argparse
import pickle
import numpy as np 

# Pytorch stuff
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable

from os.path import join, dirname

class change_shape(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

def cnn_model(nlayers,ndepth,ksize,ntargets,insize,device_id):
    
    structure=[nn.Conv2d(1,ndepth,kernel_size=ksize), nn.MaxPool2d(2), nn.ReLU()]
    ori_size=insize
    insize=insize-ksize+1
    insize=insize/2
    pad_size=int((ori_size-insize)/2)
    
    for k in range(nlayers-1):
        structure += [nn.Conv2d(ndepth,ndepth,kernel_size=ksize,padding=pad_size), nn.ReLU(),  nn.MaxPool2d(2) ]
        insize=insize-ksize+1+2*pad_size
        insize=int(np.floor(insize/2))
    
    structure +=[change_shape(), nn.Linear(insize*insize*ndepth,ntargets)]
    model = nn.Sequential(*structure)
    
    if device_id!=-1:
        with torch.cuda.device(device_id):
            model.cuda(device_id)
        
    return model

    
def get_args():
    parser = argparse.ArgumentParser('Train CNN nnet with pytorch backend')
    parser.add_argument('egs_dir', help='Example data directory')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--ntargets', type=int, default=48, help='number of targets(48)')
    parser.add_argument('--nlayers', type=int, default=4, help='number of hidden layers(4)')
    parser.add_argument('--ndepth', type=int, default=20, help='Depth of each CNN layer(20)')
    parser.add_argument('--ksize', type=int, default=5, help='Kernel size(5)')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--split_num', type=int, help='number of splits of the data(5)', default=5)
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 regularization')
    parser.add_argument('--cv_stop', type=int,
                        help='Stop after this many increases of CV error')
    return parser.parse_args()


        
def error_rate(model, features, labels, loss_fn):
    outputs = model(features)         
    loss_test = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)      
    hits = (labels == predicted).float().sum()
    return loss_test.data[0], (1 - hits / labels.size(0)).data[0]


def train(model,egs_dir,split_num,epochs,gpu_id,cv_stop,lrate,weight_decay,bsize,outmodel):
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate,
                                 weight_decay=weight_decay)
    if gpu_id!=-1:
        with torch.cuda.device(gpu_id):
            model.cuda(gpu_id)
            
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
                    
                        inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
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
                    print('%s: Cross Validation Error found to increase continuously.. exiting with present model!' % sys.argv[0])
                    re_loss, re_er = model_err(model, egs_dir, loss_fn, bsize, gpu_id)
                    print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
                    break
            
            print('%s: Maximum number of epochs exceeded!' % sys.argv[0])
            re_loss, re_er = model_err(model, egs_dir, loss_fn, bsize, gpu_id)
            print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
            
            
            # Save performance
            res_file=join(dirname(outmodel),'result')
            with open(res_file,'w') as f:
                f.write('Test set Frame Error Rate: %.2f %%' % (re_er*100))
            
            # Save model
            model=model.cpu()
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
                print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
                break
        
        print('%s: Maximum number of epochs exceeded!' % sys.argv[0])
        re_loss, re_er =cv_loss, cv_er=model_err(model, egs_dir, loss_fn, bsize, gpu_id)
        print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
        
        # Save result
        res_file=join(dirname(outmodel),'result')
        with open(res_file,'w') as f:
            f.write('Test set Frame Error Rate: %.2f %%' % (re_er*100))
        
        # Save model
        with open(outmodel, 'wb') as fid:
                pickle.dump(model, fid)
            
                    
                
             
if __name__=='__main__':
    
    print_log('# BEGIN CNN TRAINING')
              
    args=get_args()
    gpu_id=get_device_id()
    
    if gpu_id!=-1:
        print('%s: Using GPU device %d for nnet' % (sys.argv[0],gpu_id))
    else:
        print_log('Training nnet on single CPU, this will take some time!')
    
    print_log('Defining nnet model')
    
    with open(join(args.egs_dir,'dim'),'r') as fid:
        insize=int(np.sqrt(int(fid.readline())))
        
    model=cnn_model(args.nlayers,args.ndepth,args.ksize,args.ntargets,insize,gpu_id)
      
    print_log('Training nnet model')
    
    # Main Training function
    
    train(model,args.egs_dir,args.split_num,args.epochs,gpu_id,args.cv_stop,args.lrate,args.weight_decay,args.bsize,args.outmodel)
    
    print_log('# FINISHED CNN TRAINING')