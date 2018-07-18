#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:39:37 2018

@author: samiksadhu
"""

'Combine egs from different nnet training'

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

def get_args():
    parser=argparse.ArgumentParser('Combine egs files/ posteriors from two different nnet trainings - provided they represent exactly the same thing')
    
    parser.add_argument('egs_dir_1', help='First egs dir')
    parser.add_argument('egs_dir_2', help='Second egs dir')
    parser.add_argument('egs_dir_out', help='Output egs dir')
    parser.add_argument('--egs_type', help='egs or post (default=egs)', default='egs')
    
    return parser.parse_args()

def combine(egs_1,egs_2):
    
    data_1=pickle.load(open(egs_1,'rb'))
    data_2=pickle.load(open(egs_2,'rb'))
    
    return np.hstack((data_1,data_2))
    
def get_data_files(egs_dir,egs_type):
    if egs_type=='egs':
        cmd='find '+egs_dir+' -iname "data.*.egs"'
    elif egs_type=='post':
        cmd='find '+egs_dir+' -iname "data.*.post"'
    else:
        print_log('The egs type {egs_type} is not supported, use "egs" or "post", exiting script!'.format(egs_type=egs_type))
        sys.exit()
    
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')
    data_files=[]
    for line in x.splitlines():
        line=line.strip()
        data_files.append(line)
    
    return data_files

if __name__=='__main__':
    
    args=get_args()
    
    print_log('Combining data files of type {egs_type} from {dir1} and {dir2}'.format(egs_type=args.egs_type,dir1=args.egs_dir_1,dir2=args.egs_dir_2))
    
    files_1=get_data_files(args.egs_dir_1,args.egs_type); files_2=get_data_files(args.egs_dir_2,args.egs_type)
    
    if len(files_1)!=len(files_2):
        print_log('Number of data files of type {egs_type} in {dir1} and {dir2} are not equal, exiting script!'.format(egs_type=args.egs_type,dir1=args.egs_dir_1,dir2=args.egs_dir_2))
        sys.exit()
    
    for i in range(len(files_1)):
        comb_dat=combine(files_1[i],files_2[i])
        pickle.dump(comb_dat,open(join(args.egs_dir_out,'data.'+str(i+1)+'.egs'),'wb'))
        
    
    print_log('Finished combination of data')