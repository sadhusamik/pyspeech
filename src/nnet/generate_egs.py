#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:37:02 2018

@author: samiksadhu
"""

'Generate examples for nnet training'

import sys
sys.path.append('/export/b15/ssadhu/pyspeech/src/nnet/')
sys.path.append('/export/b15/ssadhu/pyspeech/src/utils/')
import argparse
from nnet import fetch_feats, fetch_labels, print_log
from gen_utils import get_dim
from os.path import join
import numpy as np
import pickle
import os

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_train', help='Training data directory')
    parser.add_argument('data_test', help='Test data directory')
    parser.add_argument('egs_dir', help='Directory to dump all data in pickle format')
    parser.add_argument('--split_num', type=int, help='number of splits of the data(5)', default=5)
    parser.add_argument('--nnet_type', help='Type of nnet to use for training vanilla/cnn', default='vanilla')
    
    return parser.parse_args()

def get_egs(data_train,data_test,egs_dir,split_num,nnet_type):
    
    os.mkdir(join(egs_dir,'test')) 
    os.mkdir(join(egs_dir,'train'))
    
    print_log('Getting data dimension')
    dim=int(get_dim(join(args.data_test,'feats.scp')))
    dim_2=int(get_dim(join(args.data_train,'feats.scp')))
    
    if dim_2!=dim:
        sys.exit('%s: Data dimensions of training and test data do not match, something is wrong, exiting script!' % sys.argv[0])
    with open(join(egs_dir,'dim'),'w') as fid:
        fid.write('%d' % dim)
    
    # Get egs for test data
    split_dir=join(data_test,'split'+str(split_num))
    for batch in range(1,split_num+1):
        test_data, keys=fetch_feats(join(split_dir,str(batch)))
        test_labels=fetch_labels(data_test,keys)
        
        if nnet_type=='cnn':
            pic_dim=int(np.sqrt(dim))
            test_data=np.reshape(test_data,(-1,pic_dim,pic_dim))
            test_data=test_data[:,np.newaxis,:,:]
        
        with open(join(egs_dir,'test','data.'+str(batch)+'.egs'), 'wb') as fid:
            pickle.dump(test_data, fid)
        
        with open(join(egs_dir,'test','labels.'+str(batch)+'.egs'), 'wb') as fid:
            pickle.dump(test_labels, fid)
            
        
    print_log('Finished generating test examples')
    
    # Get egs for train data
    split_dir=join(data_train,'split'+str(split_num))
    for batch in range(1,split_num+1):
        train_data, keys=fetch_feats(join(split_dir,str(batch)))
        train_labels=fetch_labels(data_train,keys)
        
        if nnet_type=='cnn':
            pic_dim=int(np.sqrt(dim))
            train_data=np.reshape(train_data,(-1,pic_dim,pic_dim))
            train_data=train_data[:,np.newaxis,:,:]
        
        with open(join(egs_dir,'train','data.'+str(batch)+'.egs'), 'wb') as fid:
            pickle.dump(train_data, fid)
        
        with open(join(egs_dir,'train','labels.'+str(batch)+'.egs'), 'wb') as fid:
            pickle.dump(train_labels, fid)
            
    print_log('Finished generating nnet training examples')
      
if __name__=='__main__':
    
    args=get_args()
    
    if args.nnet_type!='vanilla' and args.nnet_type!='cnn': 
        sys.exit('%s: Nnet type %s is not supported by pyspeech, exiting script!' % (sys.argv[0],args.nnet_type))
    print_log('Generating examples for nnet training')
    
    get_egs(args.data_train,args.data_test,args.egs_dir,args.split_num,args.nnet_type)
    
    