#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:50:23 2018

@author: samiksadhu
"""

' Generate Mel Filter Energy features '

import sys 
import argparse
import io
sys.path.append('../../tools')

import numpy as np
from features import createFbank, getFrames, computeLpcFast,computeModSpecFromLpc, addReverb
from scipy.fftpack import dct
import time
from scipy.io.wavfile import read
import subprocess
import pickle
from os.path import isfile,join

def extractModSpecFeatures(args, srate=16000,
                            window=np.hamming):
    """ Extract the mel scale filter-bank energy features
    """
    
    wavs=args.scp
    outfile=args.outfile
    add_reverb=args.add_reverb
    set_unity_gain=args.set_unity_gain
    nmodulations=args.nmodulations
    order=args.order
    fduration=args.fduration
    frate=args.frate
    nfilters=args.nfilters
    phone_map=args.phn_file
    phn_file_dir=args.phn_file_dir
    get_phone_labels=args.get_phone_labels
    
    
    fbank = createFbank(nfilters, int(2*fduration*srate), srate)
    
    if add_reverb:
        if add_reverb=='small_room':
            sr_r, rir=read('./RIR/RIR_SmallRoom1_near_AnglA.wav')
            rir=rir[:,1]
            rir=rir/np.power(2,15)
        elif add_reverb=='large_room':
            sr_r, rir=read('./RIR/RIR_LargeRoom1_far_AnglA.wav')
            rir=rir[:,1]
            rir=rir/np.power(2,15)
        elif add_reverb=='clean':
            print('%s: No reverberation added!' % sys.argv[0])
        else:
            raise ValueError('Invalid type of reverberation!')
        sys.stdout.flush()
        
    # Get list of phonemes
    phn_list=[]
    
    with open(phone_map,'r') as fid2:
        for line2 in fid2:
            line2=line2.strip().split()
            if len(line2)==2:
                if 'sil' not in line2 and 'SIL' not in line2:
                    phn_list.append(line2[1])
                    
    phn_list=list(set(phn_list))
    phn_list.sort()
    np.save('phone_list',phn_list)
    
    with open(wavs, 'r') as fid:

        all_feats={}
        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])
                
            fname_phn=uttid+'.PHN'
            fname_phn_base=uttid[0:-2]+'.PHN'
            
            if isfile(join(phn_file_dir,fname_phn_base)):
                fname_phn=fname_phn_base
            
            # Get all the locations of phonemes
            phone_now=np.empty(0); phone_end=np.empty(0); phone_beg=np.empty(0)
            
            if isfile(join(phn_file_dir,fname_phn)):
                print('%s: Computing Features for file: %s' % (sys.argv[0],uttid))
                sys.stdout.flush()
                with open(join(phn_file_dir,fname_phn)) as phn_file:
                    
                    
                    for phn_line in phn_file: 
                          
                        phn_locs=phn_line.strip().split()
                    
                        # Get phoneme information 
                    
                        phone_now=np.append(phone_now,phn_locs[2]) 
                        phone_end=np.append(phone_end,phn_locs[1])
                        phone_beg=np.append(phone_beg,phn_locs[0])
                phn_file.close()
                if np.size(phone_end)==0:
                    print('%s: Corrupted Phone file.. hence skipped...' % sys.argv[0])
                    continue
                
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                else:
                    sr, signal = read(inwav)
                assert sr == srate, 'Input file has different sampling rate.'
                
                signal=signal/np.power(2,15)
                
                if add_reverb:
                    if not add_reverb=='clean':
                        signal=addReverb(signal,rir)
                    
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
    
    
                cos_trans=dct(time_frames)/np.sqrt(2*int(srate * fduration))
            
                [frame_num, ndct]=np.shape(cos_trans)
     
                # Main feature computation loop 
                
                feats=np.zeros((frame_num,nfilters*nmodulations))
                sys.stdout.flush()
                for i in range(frame_num):
                    each_feat=np.zeros([nfilters,nmodulations])
                    for j in range(nfilters):
                        filt=fbank[j,0:-1]
                        band_dct=filt*cos_trans[i,:]
                        xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                        if set_unity_gain:
                            gg=1
                        mod_spec=computeModSpecFromLpc(gg,xlpc,nmodulations)
                        each_feat[j,:]=mod_spec
                    each_feat=np.reshape(each_feat,(1,nfilters*nmodulations))
                    feats[i,:]=each_feat
                
                
                if not get_phone_labels:
                    now_feats=np.empty(nfilters*nmodulations)
                else:
                    now_feats=np.empty(nfilters*nmodulations+1)
                
                for num, phn in enumerate(phone_now):
                    
                    now_frames=feats[int(phone_beg[num]):int(phone_end[num]),:]
                    if get_phone_labels:
                        ind=phn_list.index(phn)
                        fr_num=now_frames.shape[0]
                        now_frames=np.concatenate((now_frames,np.tile(ind,(fr_num,1))),axis=1)
                        
                    now_feats=np.vstack([now_feats, now_frames])
                now_feats=now_feats[1:,:]
            all_feats[uttid]=now_feats
                
        pickle.dump(all_feats,open(outfile,'wb'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract FDLP Modulation Spectral Features.')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output directory')
    parser.add_argument('phn_file', help='Phone map file')
    parser.add_argument('phn_file_dir', help='Location of all phone files')
    parser.add_argument('--nfilters', type=int, default=15,help='number of filters (15)')
    parser.add_argument('--nmodulations', type=int, default=12,help='number of modulations of the modulation spectrum (12)')
    parser.add_argument('--order', type=int, default=50,help='LPC filter order (50)')
    parser.add_argument('--fduration', type=float, default=0.5,help='Window length (0.5 sec)')
    parser.add_argument('--frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--get_phone_labels', action='store_true', help='get phone labels for each feature attached as another column to feature matrix(True)')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--set_unity_gain', action='store_true',help='Set LPC gain to 1 (True)')
    args = parser.parse_args()
    
    start_time=time.time()
    print('%s: Extracting features....' % sys.argv[0])
    sys.stdout.flush()
    extractModSpecFeatures(args)
    time_note='Execution Time: {t:.3f} seconds'.format(t=time.time()-start_time)  
    print(time_note)
    sys.stdout.flush()
