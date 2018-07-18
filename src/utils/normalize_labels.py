#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:33:29 2018

@author: samiksadhu
"""

'Check if frame numbers are the same for features and labels'

import sys
sys.path.append('../../src/featgen/') 
sys.path.append('../../src/utils/') 

import argparse
from gen_utils  import get_ark_list, load_all_arks, split_dict_and_save_ark, print_log
import numpy as np 
from os.path import join 

def get_args():
    parser=argparse.ArgumentParser('Take data.scp and labels.scp and make sure that the labels are of same duration as data')
    parser.add_argument('data', help='Data directory')
    
    return parser.parse_args()

def equalize_feats_and_labels(feats_scp,labels_scp):
    
    feats_ark_list=get_ark_list(feats_scp); labels_ark_list=get_ark_list(labels_scp)
    feats_all=load_all_arks('dummy',feats_scp,feats_ark_list,'copy-feats')
    labels_all=load_all_arks('dummy',labels_scp,labels_ark_list,'copy-feats')
    
    if len(list(feats_all.keys()))!=len(list(labels_all.keys())):
        sys.exit('%s: Utterance number for features and labels are different, exiting script!' % (sys.argv[0]))
    
    feat_keys=list(feats_all.keys()); label_keys=list(labels_all.keys())
    
    for key in feat_keys:
        if key not in label_keys:
            sys.exit('%s: The key %s is present in features but not in labels, exiting script!' % (sys.argv[0],key))
        else:
            feats=feats_all[key]; labels=labels_all[key];
            if np.shape(feats)[0] > np.shape(labels)[0]:
                # more feature vectors than labels extend labels
                print('%s: Key %s has a frame number mismatch' % (sys.argv[0],key))
                last_label=labels[-1,0]
                dif=np.shape(feats)[0]-np.shape(labels)[0]; add_end=np.repeat(last_label,dif); add_end=add_end[:,np.newaxis]
                labels=np.vstack((labels,add_end))
                labels_all[key]=labels
            elif np.shape(feats)[0] < np.shape(labels)[0]:
                print('%s: Key %s has a frame number mismatch' % (sys.argv[0],key))
                labels=labels[0:np.shape(feats)[0]]
                labels_all[key]=labels
    return labels_all

if __name__=='__main__':
    args=get_args()
    
    print_log('Equalizing features and labels')  
    labels_all=equalize_feats_and_labels(join(args.data,'feats.scp'),join(args.data,'labels','normalized.1.scp'))
    
    print_log('Saving the adjusted label files')
    split_dict_and_save_ark(labels_all,1,args.data,'labels','copy-feats')
    
    print_log('Finished equalizing labels and features')
    