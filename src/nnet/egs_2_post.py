#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:48:26 2018

@author: samiksadhu
"""

'Convert nnet training examples to posteriors given a pytorch model'

import sys
sys.path.append('/export/b15/ssadhu/pyspeech/src/featgen/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/utils/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/nnet/')
from nnet import print_log

import numpy as np
import argparse
import pickle
from os.path import join
import subprocess
import torch
from torch.autograd import Variable 


def get_args():
    parser = argparse.ArgumentParser('Get posteriors from egs pytorch nnet')
    parser.add_argument('model', help='nnet model')
    parser.add_argument('egs_dir', help='Directory containing data in proper format')
    parser.add_argument('post_dir', help='output posterior file')
    
    return parser.parse_args()

def get_data_files(egs_dir):
    
    cmd='find '+egs_dir+' -iname "data.*.egs"'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')
    data_files=[]
    for line in x.splitlines():
        line=line.strip()
        data_files.append(line)
        
    return data_files

def softmax(x):
    
    frame_num=x.shape[0]
    feat_dim=x.shape[1]
    
    soft_out=np.zeros((frame_num,feat_dim))
    for i in range(frame_num):
        soft_out[i,:]=np.exp(x[i,:])/(np.sum(np.exp(x[i,:])))
    return soft_out

def get_post(model,egs):
    
    nnet=pickle.load(open(model,'rb'))
    data=pickle.load(open(egs,'rb'))
    
    x=Variable(torch.from_numpy(data).float())
    x=nnet(x)
    post=softmax(x.data.numpy())
    
    return post

if __name__=='__main__':
    
    args=get_args()
    
    print_log('Obtaining the data files from {egs_dir}'.format(egs_dir=args.egs_dir))
    files=get_data_files(args.egs_dir)
    
    print_log('Obtaining posteriors')
    count=1
    for i in files:
        post=get_post(args.model,i)
        pickle.dump(post,open(join(args.post_dir,'data.'+str(count)+'.post'),'wb'))
        count+=1
   
    print_log('Finished obtaining posteriors from {egs_dir}'.format(egs_dir=args.egs_dir))