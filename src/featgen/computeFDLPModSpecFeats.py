#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:50:23 2018

@author: samiksadhu
"""

'Computing FDLP Modulation Spectral Features' 

import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis 
import sys
import pickle
import time 

from features import getFrames,createFbank,computeLpcFast,computeModSpecFromLpc, addReverb

def extractModSpecFeatures(args, srate=16000, window=np.hanning):
    
    wavs=args.scp
    outfile=args.outfile
    phone_map=args.phn_file
    phn_file_dir=args.phn_file_dir
    get_phone_labels=args.get_phone_labels
    add_reverb=args.add_reverb
    set_unity_gain=args.set_unity_gain
    nmodulations=args.nmodulations
    order=args.order
    fduration=args.fduration
    frate=args.frate
    nfilters=args.nfilters
    
    '''Extract the Modulation Spectral Features.

    Args:
        wavs (list): List of (uttid, 'filename or pipe-command').
        outdir (string): Output of an existing directory.
        phone_map(string): Map of the phonemes from Kaldi
        get_phone_labels(bool): Set True if you want to get the phoneme labels  
        fduration (float): Frame duration in seconds.
        frate (int): Frame rate in Hertz.
        hz2scale (function): Hz -> 'scale' conversion.
        nfft (int): Number of points to compute the FFT.
        nfilters (int): Number of filters.
        postproc (function): User defined post-processing function.
        srate (int): Expected sampling rate of the audio.
        scale2hz (function): 'scale' -> Hz conversion.
        srate (int): Expected sampling rate.
        window (function): Windowing function.

    Note:
        It is possible to use a Kaldi like style to read the audio
        using a "pipe-command" e.g.: "sph2pipe -f wav /path/file.wav |"

    '''

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
                         
    with open(wavs, 'r') as fid:
        all_feats={}
                
        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])
                
            if inwav[-1] == '|':
                proc = subprocess.run(inwav[:-1], shell=True,
                                      stdout=subprocess.PIPE)
                sr, signal = read(io.BytesIO(proc.stdout))
            else:
                sr, signal = read(inwav)
            assert sr == srate, 'Input file has different sampling rate.'
            
            # I want to work with numbers from 0 to 1 so.... 
            signal=signal/np.power(2,15)
            
            if add_reverb:
                if not add_reverb=='clean':
                    signal=addReverb(signal,rir)
                
            fname_phn=uttid+'.PHN'
            
            # Get all phones and their center 
            
            if os.path.isfile(os.path.join(phn_file_dir,fname_phn)):
                phn_file=open(os.path.join(phn_file_dir,fname_phn))
                phone_mid=np.empty(0)
                phone_now=np.empty(0)
                for line2 in phn_file:

                    phn_locs=line2.strip().split() 
                    if phn_locs[2] in phn_list:
                        ind=phn_list.index(phn_locs[2])   
                        phone_now=np.append(phone_now, ind) 
                        phone_mid=np.append(phone_mid
                        ,int(int(phn_locs[0])+int(phn_locs[1]))/2)
                
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
    
                cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                
                [frame_num, ndct]=np.shape(cos_trans)
                                
                only_compute=len(phone_mid)
 
                if get_phone_labels:
                    feats=np.zeros([only_compute,nmodulations*nfilters+1]) 
                else:
                    feats=np.zeros([only_compute,nmodulations*nfilters])
                
                print('Computing Features for file: %s' % uttid)
                sys.stdout.flush()
                for kk in range(only_compute):
                    i=int(np.floor((phone_mid[kk])))
                    each_feat=np.zeros([nfilters,nmodulations])
                    for j in range(nfilters):
                        filt=fbank[j,0:-1]
                        band_dct=filt*cos_trans[i,:]
                        #band_dct=band_dct[band_dct>0]
                        xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                        if set_unity_gain:
                            gg=1
                        mod_spec=computeModSpecFromLpc(gg,xlpc,nmodulations)
                        each_feat[j,:]=mod_spec
                    each_feat=np.reshape(each_feat,(1,nfilters*nmodulations))
                    if get_phone_labels:
                        feats[kk,:]=np.append(each_feat,phone_now[kk])
                    else:
                        feats[kk,:]=each_feat
            
                all_feats[uttid]=feats
        
        # Save the final BIG feature file 
        pickle.dump(all_feats,open(outfile,'wb'))
        np.save(os.path.join(os.path.dirname(outfile), 'phone_list'), phn_list)
    
    
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
    parser.add_argument('--get_phone_labels', action='store_true',
                        help='get phone labels for each feature attached as another column to feature matrix(True)')
    parser.add_argument('--add_reverb', 
                        help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--set_unity_gain', action='store_true',
                        help='Set LPC gain to 1 (True)')
    args = parser.parse_args()
    
    start_time=time.time()
    print('%s: Extracting features....' % sys.argv[0])
    sys.stdout.flush()
    extractModSpecFeatures(args)
    time_note='Execution Time: {t:.3f} seconds'.format(t=time.time()-start_time)  
    print(time_note)
    sys.stdout.flush()
