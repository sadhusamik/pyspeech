#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:46:38 2018

@author: samik sadhu 
"""

'Prepare data, train MLP and do cross validation using batch loading'


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


def get_megbatch(train_files,data_dim,meg_batch_num,outdir):
    print('%s: Getting mega match number %d with %d files...' % (sys.argv[0],meg_batch_num,len(train_files)))
    train_data=np.empty((0,data_dim)); train_labels=np.array([]);
    for i, dat_set in enumerate(train_files):
            data_dict=pickle.load(open(dat_set,'rb'))
            data,labels=dict_2_data(data_dict,data_dim)
            train_data=np.append(train_data,data,axis=0)
            train_labels=np.append(train_labels,labels)
    print('%s: Megabatch %d compiled!' % (sys.argv[0],meg_batch_num))
    np.save(join(outdir,'data_mbatch_'+str(meg_batch_num)+'.npy'),train_data)
    np.save(join(outdir,'labels_mbatch_'+str(meg_batch_num)+'.npy'),train_labels)
    sys.stdout.flush()
    

def get_data_dim(data_dict):
    utt_list=list(data_dict.keys())
    data_samp=data_dict[utt_list[0]]
    data_dim=data_samp.shape[1]-1
    return data_dim

def dict_2_data(data_dict,data_dim):
    
    data=np.empty(data_dim)
    labels=np.array([])
    utt_list=list(data_dict.keys())
    for i, utt_id in enumerate(utt_list):
        data=np.vstack((data,data_dict[utt_id][:,0:-1]))   
        labels=np.append(labels,data_dict[utt_id][:,-1])
    data=data[1:,:]
    return data, labels    
    
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
    
    


def get_sample_meanvar(train_files):
    
    size_acc=0
    data_dict=pickle.load(open(train_files[0],'rb'))
    data_dim=get_data_dim(data_dict)    
    data,labels=dict_2_data(data_dict,data_dim)
    size_acc+=np.shape(data)[0] 
    mean_acc=data.mean(axis=0)
   
    print('%s: Getting mean of training samples...' % sys.argv[0])
    for ind, file in enumerate(train_files):
        if ind==0:
            continue;            
        data_dict=pickle.load(open(file,'rb'))
        data,labels=dict_2_data(data_dict,data_dim)
        size_now=np.shape(data)[0]        
        mean_now=data.mean(axis=0)
        mean_acc=(mean_now*size_now+mean_acc*size_acc)/(size_now+size_acc)
        size_acc+=size_now
    
    size_acc=0
    data_dict=pickle.load(open(train_files[0],'rb'))
    data,labels=dict_2_data(data_dict,data_dim)
    size_acc+=np.shape(data)[0]
    var_acc=np.sum(np.square(data-mean_acc),axis=0)
    
    print('%s: Getting variance of training samples...' % sys.argv[0])   
    for ind,file in enumerate(train_files):
        if ind==0:
            continue;
        data_dict=pickle.load(open(file,'rb'))
        data,labels=dict_2_data(data_dict,data_dim)      
        size_now=np.shape(data)[0]
        size_acc+=size_now
        var_acc+=np.sum(np.square(data-mean_acc),axis=0)
    var_acc=var_acc/size_acc;
    
    return mean_acc, var_acc

def error_rate(model, features, labels, loss_fn):
    outputs = model(features)         
    loss_test = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)      
    hits = (labels == predicted).float().sum()
    return loss_test.data[0], (1 - hits / labels.size(0)).data[0]

def error_rate_2(model, d_data, d_labels, r_data, r_labels, loss_fn):
    d_out = model(d_data); r_out = model(r_data)  
    d_loss = loss_fn(d_out, d_labels); r_loss = loss_fn(r_out, r_labels)
    _, d_pred = torch.max(d_out, dim=1); _, r_pred = torch.max(r_out, dim=1)      
    d_hits = (d_labels == d_pred).float().sum(); r_hits = (r_labels == r_pred).float().sum()
    return d_loss.data[0], (1 - d_hits / d_labels.size(0)).data[0], r_loss.data[0], (1 - r_hits / r_labels.size(0)).data[0]


def run(train_files,test_data,test_labels,result_data,result_labels,args):
    
    if args.mvnorm:
        mean=np.load(join(args.data_directory,'data_mean.npy'))
        var=np.load(join(args.data_directory,'data_var.npy'))
        
    if args.activation=='sigmoid':
        activ=nn.Sigmoid()
    elif args.activation=='tanh':
        activ=nn.Tanh()
    elif args.activation=='relu':
        activ=nn.ReLU()
    else:
        raise ValueError('Activation function not found!')
            
    
    # Check the data dimension
    data_dict=pickle.load(open(train_files[0],'rb'))
    data_dim=get_data_dim(data_dict)   
    
    if args.mvnorm:
        test_data -= mean
        test_data /= np.sqrt(var)
        
    if args.mvnorm:
        result_data -= mean
        result_data /= np.sqrt(var)
        
    # Build the MLP.
    targetdim=args.ntargets
    print('%s: Building the MLP...' % sys.argv[0])
    sys.stdout.flush()
    if args.kink_dim:
        
        structure = [nn.Linear(data_dim, args.kink_dim), activ]
        for i in range(args.nlayers - 1):
            if i==0:
                structure += [nn.Linear(args.kink_dim, args.nunits), activ]
            else:
                structure += [nn.Linear(args.nunits, args.nunits), activ]
        structure += [nn.Linear(args.nunits, targetdim)]
        model = nn.Sequential(*structure)
    
    else:
        
        structure = [nn.Linear(data_dim, args.nunits), activ]
        for i in range(args.nlayers - 1):
            structure += [nn.Linear(args.nunits, args.nunits), activ]
        structure += [nn.Linear(args.nunits, targetdim)]
        model = nn.Sequential(*structure)
    
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            model.cuda(args.gpu)
                  
    print('%s: Defining Loss Function...' % sys.argv[0])
    sys.stdout.flush()
    # Loss function.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate,
                                 weight_decay=args.weight_decay)

    
    
    test_data, test_labels = torch.from_numpy(test_data).float(), \
        torch.from_numpy(test_labels).long()
        
    result_data, result_labels = torch.from_numpy(result_data).float(), \
        torch.from_numpy(result_labels).long()
        
    #v_train_data, v_train_labels = Variable(train_data), Variable(train_labels)
    
    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            v_test_data, v_test_labels = Variable(test_data).cuda(), Variable(test_labels).cuda()
            v_result_data, v_result_labels = Variable(result_data).cuda(), Variable(result_labels).cuda()
    else:
        v_test_data, v_test_labels = Variable(test_data), Variable(test_labels)
        v_result_data, v_result_labels = Variable(result_data), Variable(result_labels)

     
    print('%s: Start Training Iterations...' % sys.argv[0])
    sys.stdout.flush()
    
    megbatch_dir=join(args.data_directory,'mega_batches')
    
    if args.gpu is not None:
        
        with torch.cuda.device(args.gpu):
            
            #Start Each Epoch 
            cv_err_old=1
            warn_time=0
            for epoch in range(args.epochs):
                
                t_loss = 0.0
                t_er = 0.0
                # Start Each Mega_batch
                batch_num=0
                for meg_batch in range(1,args.mega_batch_num+1):
                    
                    train_data=np.load(join(megbatch_dir,'data_mbatch_'+str(meg_batch)+'.npy'))
                    train_labels=np.load(join(megbatch_dir,'labels_mbatch_'+str(meg_batch)+'.npy')) 
                    if args.mvnorm:
                        train_data -= mean
                        train_data /= np.sqrt(var)
        
                    train_data, train_labels = torch.from_numpy(train_data).float(), \
                    torch.from_numpy(train_labels).long()
                    
                    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                              shuffle=True)
    
                    for i, data in enumerate(trainloader):
                        batch_num+=1
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
                        
                
                t_loss /= batch_num
                t_er /= batch_num
                cv_loss, cv_er = error_rate(model, v_test_data, v_test_labels, loss_fn)
                #re_loss, re_er = error_rate(model, v_result_data, v_result_labels, loss_fn) 
                #cv_loss, cv_er, re_loss, re_er = error_rate_2(model, v_test_data, v_test_labels, v_result_data, v_result_labels, loss_fn)
                logmsg = 'epoch: {epoch} loss (train): {t_loss:.3f}  ' \
                         'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                         'error rate (cv): {cv_er:.3%}'.format( 
                         epoch=epoch+1, t_loss=t_loss, t_er=t_er,
                         cv_loss=cv_loss, cv_er=cv_er)
                sys.stdout.flush()
                print(logmsg)
                if args.cv_stop:
                    if cv_er>cv_err_old:
                        warn_time+=1
                    cv_err_old=cv_er
                    
                    if warn_time>=args.cv_stop:
                        print('%s: Cross Validation Error found to increase in %d epochs.. exiting with present model!' % (sys.argv[0],args.cv_stop))
                        re_loss, re_er = error_rate(model, v_result_data, v_result_labels, loss_fn)
                        print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
                        break
                    
            
            print('%s: Maximum number of epochs exceeded!' % sys.argv[0])
            re_loss, re_er = error_rate(model, v_result_data, v_result_labels, loss_fn)
            print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
            model=model.cpu()
            
            with open(args.outmodel, 'wb') as fid:
                pickle.dump(model, fid)
    else:
        
        #Start Each Epoch 
        cv_err_old=1
        warn_time=0
        for epoch in range(args.epochs):
             t_loss = 0.0
             t_er = 0.0
             
             # Start Each Mega_batch
             batch_num=0
             for meg_batch in range(1,args.mega_batch_num+1):
                
                train_data=np.load(join(megbatch_dir,'data_mbatch_'+str(meg_batch)+'.npy'))
                train_labels=np.load(join(megbatch_dir,'labels_mbatch_'+str(meg_batch)+'.npy')) 
                        
                if args.mvnorm:
                    train_data -= mean
                    train_data /= np.sqrt(var)
                    
                train_data, train_labels = torch.from_numpy(train_data).float(), \
                torch.from_numpy(train_labels).long()
                
                dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsize,
                                          shuffle=True)
                
                for i, data in enumerate(trainloader):
                    batch_num+=1
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
                        logmsg = 'epoch: {epoch} mega-batch: {meg_batch} mini-batch: {mbatch}  loss (train): {t_loss:.3f}  ' \
                                 'error rate (train): {t_er:.3%} loss (cv): {cv_loss:.3f} ' \
                                 'error rate (cv): {cv_er:.3%}'.format( 
                                 epoch=epoch+1, meg_batch=meg_batch, mbatch=i+1, t_loss=t_loss, t_er=t_er,
                                 cv_loss=cv_loss, cv_er=cv_er)
                        
                        t_er = 0.0
                        t_loss = 0.0
                        print(logmsg)
                        sys.stdout.flush()
                    
                        if cv_er>cv_err_old:
                            warn_time+=1
                            cv_err_old=cv_er
                        
                        if warn_time>=2:
                            print('%s: Cross Validation Error found to increase in 2 epochs.. exiting with present model!' % sys.argv[0])
                            re_loss, re_er = error_rate(model, v_result_data, v_result_labels, loss_fn)
                            print('%s: The final test performance is: %.2f %%' %  (sys.argv[0],re_er*100))
                            break
        
        with open(args.outmodel, 'wb') as fid:
                pickle.dump(model, fid)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_directory', help='place to get all training and test data in .npy format')
    parser.add_argument('outmodel', help='output file')
    parser.add_argument('--ntargets', type=int, default=41, help='number of targets(41)')
    parser.add_argument('--nlayers', type=int, default=4, help='number of hidden layers(4)')
    parser.add_argument('--nunits', type=int, default=256, help='number of units per leayer(256)')
    parser.add_argument('--gpu',type=int,help='gpu device id (Ignore if you do not want to run on gpu!)')
    parser.add_argument('--bsize', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--mega_batch_num', type=int, default=5,
                        help='number of big data batches to be uploaded as a whole on to RAM/GPU')
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
    parser.add_argument('--cv_stop', type=int,
                        help='Stop after this many increases of CV error')
    parser.add_argument('--activation', default='tanh',
                        help='tanh OR sigmoid OR relu')
    parser.add_argument('--kink_dim', type=int , help='Puts a kink_dim dimensional layer at the beginning to plot filters')
    args = parser.parse_args()

    assert args.nlayers > 0

    print('%s: Running  MLP training...' % sys.argv[0])
    sys.stdout.flush()
    allfiles = [f for f in listdir(args.data_directory) if isfile(join(args.data_directory, f))]
    
    train_files=[]; test_files=[]
               
    for i in range(len(allfiles)):
        if 'train' in allfiles[i]:
            train_files.append(os.path.join(args.data_directory,allfiles[i]))
            
    print('%s: In total %d train data files found..passing them for MLP training' % (sys.argv[0],len(train_files)))
    sys.stdout.flush()       
    test_data_all=np.load(join(args.data_directory,'test_data.npy'))
    test_labels_all=np.load(join(args.data_directory,'test_labels.npy'))
    
    reduce_size=int(test_data_all.shape[0]/4)   
    
    test_data=test_data_all[reduce_size:2*reduce_size,:]
    test_labels=test_labels_all[reduce_size:2*reduce_size]
      
    result_data=test_data_all[0:reduce_size,:]
    result_labels=test_labels_all[0:reduce_size]

    
    run(train_files,test_data,test_labels,result_data,result_labels,args)
