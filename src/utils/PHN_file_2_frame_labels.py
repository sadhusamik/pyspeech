#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:07:17 2018

@author: samiksadhu
"""

' Get frame-wise phonetic labels'

import sys
sys.path.append('../../src/featgen') # appned the feature files directory 
sys.path.append('../../src/utils') # appned the feature files directory 

import argparse
from os.path import join, isfile, abspath
import numpy as np 
from gen_utils import split_dict_and_save_ark, print_log

def get_args():
    parser=argparse.ArgumentParser('Get frame-wise phonetic labels')
    parser.add_argument('data',help='Data directory')
    parser.add_argument('PHN_file_dir',help='PHN file directory')
    parser.add_argument('phones',help='phones in database - like kaldi phonex.txt')
    parser.add_argument('--frm_rate',help='Frame rate of sampling (100 Hz)', type=float, default=100)
    parser.add_argument('--split_num', type=int, default=1, help='Number of ark files to split the label files into (1)')
    
    
    return parser.parse_args()

def get_phone_mapping(phones):
    phone_dict={}
    count=1
    
    if isfile(phones):
        with open(phones,'r') as fid:
            for line in fid:
                token=line.strip().split()
                phone_dict[token[0]]=count
                count+=1
    else:
        sys.exit('%s: The phone file list %s does not exist, exiting script!' % (sys.argv[0],phones))
    return phone_dict
        
        
def get_labels(data_dir,PHN_file_dir,phone_dict,frm_rate):
    
    feats=join(data_dir,'feats.scp')
    
    with open(feats,'r') as fid:
        all_labels={}
        for line in fid:
            
            tokens=line.strip().split()
            uttid=tokens[0]
            
            PHN_file_name=uttid+'.PHN'
            
            if isfile(join(PHN_file_dir,PHN_file_name)):
                labels=np.empty(0)
                with open(join(PHN_file_dir,PHN_file_name)) as fid2:
                    for line2 in fid2:
                         token2=line2.strip().split()
                         phn_idx=phone_dict[token2[2]]
                         duration=int(np.round(float(token2[1])*frm_rate))
                         
                         labels=np.append(labels,np.repeat(phn_idx,duration))
                labels=labels[:,np.newaxis]
                all_labels[uttid]=labels
            else:
                sys.exit('%s: The PHN file for %s does not exist in %s, exiting script!' % (sys.argv[0],PHN_file_name,PHN_file_dir))
            
    return all_labels
            


if __name__=='__main__':
    args=get_args()
    
    print_log('Getting the list of phonemes')
    phone_dict=get_phone_mapping(args.phones)
    
    print_log('Obtaining the frame-wise phone labels from PHN files')
    all_labels=get_labels(args.data,args.PHN_file_dir,phone_dict,args.frm_rate)
    
    print_log('Saving the label files')
    split_dict_and_save_ark(all_labels,args.split_num,abspath(args.data),'labels','copy-feats') # Split into ark files and save the labels
    
    print_log('Fininshed computing labels from phone files')