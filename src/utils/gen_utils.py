#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:59:48 2018

@author: samiksadhu
"""
'General ulitlity methods'

import sys 
sys.path.append('/export/b15/ssadhu/pyspeech/src/featgen/') # appned the source files

import subprocess
from features import ark2Dict, dict2Ark
from os.path import join, exists
import numpy as np 
import os 

def get_ark_list(scp):
    cmd='cat '+scp+' | cut -d" " -f2 | cut -d":" -f1 | uniq'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')
    ark_files=[]
    for line in x.splitlines():
        ark_files.append(line)
        
    return ark_files

def get_dim(scp):
    
    cmd='feat-to-dim scp:'+scp+' -'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]
    

def load_all_arks(data_dir,scp,ark_files,kaldi_cmd): #TODO: data_dir is useless here.. remove it 
    
    all_feats={}
    dim=get_dim(scp);
    print('Feature dimension %s' % get_dim(scp))
    for ark in ark_files:
        all_feats.update(ark2Dict(ark,int(dim),kaldi_cmd))
        
    return all_feats

def chunks(l, n):
    
    for i in range(0, len(l), n):
        yield l[i:i + n]

def sub_dict(full_dict, sub_keys, default=None):
    return dict([ (k, full_dict.get(k, default)) for k in sub_keys ])

def split_dict_and_save_ark(in_dict,split_num,data_folder,name,kaldi_cmd):
    
    all_keys=list(in_dict.keys())
    dict_size=int(np.ceil(len(all_keys)/split_num))
    split_keys=list(chunks(all_keys,dict_size))
    
    if not exists(join(data_folder,name)):
        os.makedirs(join(data_folder,name))
    
    for i in range(0,split_num):
        ark_file_name=join(data_folder,name,'normalized.'+str(i+1))
        dict2Ark(sub_dict(in_dict,split_keys[i]),ark_file_name,kaldi_cmd)
        
def print_log(text):
    print('%s: %s' % (sys.argv[0],text)); 
    sys.stdout.flush()
