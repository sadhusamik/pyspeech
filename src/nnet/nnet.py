#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:41:36 2018

@author: samiksadhu
"""

'Nnet utility scripts'

import sys
sys.path.append('/export/b15/ssadhu/pyspeech/src/featgen/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/utils/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/nnet/')

from gen_utils import get_dim
from features import ark2Dict
from os.path import join
import numpy as np 
import subprocess
import pickle
import torch
from torch.autograd import Variable

def model_err(model, egs_dir, loss_fn, bsize, gpu_id):
    egs_dir=join(egs_dir,'test')
    cmd='find '+egs_dir+' -iname "data.*.egs"'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')
    data_files=[]
    for line in x.splitlines():
        line=line.strip()
        data_files.append(line)
        
    cmd='find '+egs_dir+' -iname "labels.*.egs"'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')

    label_files=[]
    for line in x.splitlines():
        line=line.strip()
        label_files.append(line)        
    if len(label_files)!=len(data_files):
        sys.exit('%s: Number of test data (%d) and label (%d) files are different, exiting script!' % (sys.argv[0],len(data_files),len(label_files)))
    
    split_num=len(data_files)
    batch_count=0
    t_loss = 0.0
    t_er = 0.0
    for batch in range(1,split_num+1):
        test_data=pickle.load(open(join(egs_dir,'data.'+str(batch)+'.egs'),'rb'))
        test_labels=pickle.load(open(join(egs_dir,'labels.'+str(batch)+'.egs'),'rb'))
        
        test_data, test_labels = torch.from_numpy(test_data).float(), \
        torch.from_numpy(test_labels.flatten()-1).long()
        
        dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=bsize,
                                  shuffle=True)
        
        for i, data in enumerate(trainloader):
            if gpu_id!=-1:
                inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
            else:
                inputs, labels = Variable(data[0]), Variable(data[1])
                
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Compute the error rate on the training set.
            
            _, predicted = torch.max(outputs, dim=1)
            hits = (labels == predicted).float().sum()
            t_er += (1 - hits / labels.size(0)).data[0]
            t_loss += loss.data[0]
            batch_count+=1
            
    t_loss /= batch_count
    t_er /= batch_count
    return t_loss, t_er
    
def print_log(text):
    print('%s: %s' % (sys.argv[0],text)); 
    sys.stdout.flush()
    
def get_device_id():
    cmd='free-gpu'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    return int(proc.stdout.decode('utf-8').strip().split()[0])

def dict_2_data(data_dict,data_dim):
    
    data=np.empty(data_dim)
    utt_list=list(data_dict.keys())
    for i, utt_id in enumerate(utt_list):
        data=np.vstack((data,data_dict[utt_id]))   
    data=data[1:,:]
    return data 

def fetch_feats(split_dir):
    kaldi_cmd='apply-cmvn --utt2spk=ark:'+split_dir+'/utt2spk scp:'+split_dir+'/cmvn.scp scp:'+split_dir+'/feats.scp ark,t:-'
    dim=int(get_dim(join(split_dir,'feats.scp')))
    proc=subprocess.run(kaldi_cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')
    feat_count=0
    start=0
    feats=np.empty((0,dim))
    all_feats={}
    fcount=0;
    for line in x.splitlines(): 

        line=line.strip().split()
        if len(line)>=1:
            if line[-1]=='[':
                start=1
                feat_count+=1 #Starting of a feature       
                uttname=line[0]
                feats=np.empty((0,dim))
                fcount+=1;
            if start==1 and line[-1]!='[':
                if line[-1]==']':
                    line=line[0:-1]
                    x=np.array(line).astype(np.float)
                    x=np.reshape(x,(1,len(x)))

                    feats=np.concatenate((feats,x),axis=0)
                    all_feats[uttname]=feats
                    # Refresh everything 
                    start=0
                    feats=np.empty((0,dim))
                else:
                    x=np.array(line).astype(np.float)
                    x=np.reshape(x,(1,len(x)))
                    feats=np.concatenate((feats,x),axis=0)
    print('%s: Tranfered %d utterances from ark to dict' % (sys.argv[0],fcount))  
    return dict_2_data(all_feats,dim), list(all_feats.keys())

def fetch_labels(data_dir,keys):
    ark=join(data_dir,'labels','normalized.1.ark')
    label_dict=ark2Dict(ark,1,'copy-feats')
    
    sub_dict={}
    for k in keys:
        sub_dict[k]=label_dict[k]
    
    return dict_2_data(sub_dict,1)