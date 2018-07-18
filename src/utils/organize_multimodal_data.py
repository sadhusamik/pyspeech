#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:07:51 2018

@author: samiksadhu
"""

'Check the multimodality data to generate parallel data' 

import sys
sys.path.append('../../src/featgen') # appned the source files


import argparse  
import numpy as np 
from os.path import join, basename, dirname 
from scipy.interpolate import interp1d
from gen_utils import get_ark_list, load_all_arks, split_dict_and_save_ark, print_log

def get_args():
    parser = argparse.ArgumentParser('Check multi-modal data and generate parallel data (Apply same sampling rate and get same number of frames')
    parser.add_argument('data_1', help='Data folder from mode 1')
    parser.add_argument('data_2', help='Data folder from mode 2')
    parser.add_argument('--kaldi_cmd',help='kaldi command to convert ark format to text', default='copy-feats')
    parser.add_argument('--split_num', help='Number of ark files to generate for final features', default=10)
    args=parser.parse_args()
    
    return args

def equalize(feats_1,feats_2):
    
    'Equalizes feats_2 to have the same number of samples as feats_1'
    
    fr_num_1, dim_1=np.shape(feats_1); fr_num_2, dim_2=np.shape(feats_2)
    
    # feats_1 has more samples and we need to equalize 
            
    x_source=np.linspace(1,fr_num_2)
    x_target=np.linspace(1,fr_num_1)
    
    for i in range(0,dim_2):
        y=feats_2[:,i]
        f=interp1d(x_source,y,kind='quadratic')
        y_target=f(x_target)
        
    return feats_1, y_target

def equalize_samples(all_feats_1,all_feats_2):
    
    'Equalize the sampling rate of all features'
    
    utt_list_1=list(all_feats_1.keys())
    utt_list_2=list(all_feats_2.keys())
    
    all_feats_11={}
    all_feats_22={}
    
    for key in utt_list_1:
        try:
            feats_2=utt_list_2[key]
        except Exception:
            print('The dictionary keys from different feature sources have a mismatch!')
        feats_1=utt_list_1[key]
        fr_num_1=np.shape(feats_1)[0]; fr_num_2=np.shape(feats_2)[0]
        
        if fr_num_1 > fr_num_2:
            
            feats_11, feats_22=equalize(feats_1,feats_2)
            
        elif fr_num_1 < fr_num_2:
                
            feats_22, feats_11=equalize(feats_2,feats_1)
        else:
            feats_11, feats_22=feats_1, feats_2
            
        
        all_feats_11[key]=feats_11; all_feats_22[key]=feats_22
        
    return all_feats_11; all_feats_22
    
    
if __name__=='__main__':
       
    print_log('Obtaining arguments')
    
    args=get_args()
    
    print_log('Fetch list of ark files from feats.scp')
    
    ark_list_1=get_ark_list(join(args.data_1,'feats.scp'))
    ark_list_2=get_ark_list(join(args.data_2,'feats.scp'))
    
    print_log('Loading all ark files')
    
    feat_dict_1=load_all_arks(args.data_1,join(args.data_1,'feats.scp'),ark_list_1,args.kaldi_cmd)
    feat_dict_2=load_all_arks(args.data_2,join(args.data_1,'feats.scp'),ark_list_2,args.kaldi_cmd)
    
    print_log('Equalizing features')
    
    feat_dict_eq_1, feat_dict_eq_2=equalize(feat_dict_1,feat_dict_2)
    
    print_log('Saving the equalized features in fresh ark files')
    
    name='normalized' # Name of the folder to save the features
    data_folder=join(dirname(args.data_1),basename(args.data_1),'normalized')
    split_dict_and_save_ark(feat_dict_eq_1,args.split_num,data_folder,name,args.kaldi_cmd)    
    data_folder=join(dirname(args.data_2),basename(args.data_2),'normalized')
    split_dict_and_save_ark(feat_dict_eq_2,args.split_num,data_folder,name,args.kaldi_cmd)
    
    print_log('Finished organizing data')
    