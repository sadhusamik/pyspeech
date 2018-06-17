#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:49:06 2018

@author: samiksadhu
"""

' Generate Mel Filter Energy features at center of phoneme'

import sys 
import argparse
import io
sys.path.append('../../tools')

import numpy as np
from features import getFrames,createFbank, spliceFeats, addReverb
from scipy.fftpack import fft 
from scipy.io.wavfile import read
import subprocess
import pickle
from os.path import isfile,join
import time 

def extractMelEnergyFeats(wavs, outfile, phone_map, phn_file_dir, fduration, context, frate,
                            nfft, nfilters, get_phone_labels=False, add_reverb=True, srate=16000,
                            window=np.hamming):
    """ Extract the mel scale filter-bank energy features
    """
    
    fbank = createFbank(nfilters, nfft, srate)
    
    if add_reverb:
        sr_r, rir=read('./RIR/RIR_LargeRoom1_far_AnglA.wav')
        rir=rir[:,1]
        rir=rir/np.power(2,15)
            
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
                
            fname_phn=uttid+'.PHN'
            
            # Get all the locations of phonemes
            phone_now=np.empty(0,dtype=int); phone_mid=np.empty(0,dtype=int); 
            
            if isfile(join(phn_file_dir,fname_phn)):
                print('%s: Computing Features for file: %s' % (sys.argv[0],uttid))
                sys.stdout.flush()
                with open(join(phn_file_dir,fname_phn)) as phn_file:
                    
                    
                    for phn_line in phn_file: 
                          
                        phn_locs=phn_line.strip().split()
                    
                        # Get phoneme information 
                        if phn_locs[2] in phn_list:
                            ind=phn_list.index(phn_locs[2])   
                            phone_now=np.append(phone_now, ind) 
                            phone_mid=np.append(phone_mid
                            ,int((int(phn_locs[0])+int(phn_locs[1]))/2))
                phn_file.close()
                if np.size(phone_mid)==0:
                    print('%s: Corrupted Phone file.. hence skipped...' % sys.argv[0])
                    sys.stdout.flush()
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
                    signal=addReverb(signal,rir)
                    
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
                    
                melEnergy_frames=np.log10(np.matmul(np.abs(fft(time_frames,int(nfft/2+1),axis=1)),np.transpose(fbank)))
                
                if args.context:
                    melEnergy_frames=spliceFeats(melEnergy_frames,context)
                feats=melEnergy_frames[phone_mid,:]
                
                if get_phone_labels:
                    feats=np.append(feats,phone_now.reshape(len(phone_now),1),axis=1)

                all_feats[uttid]=feats
                
        pickle.dump(all_feats,open(outfile,'wb'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Mel Energy Features')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('phn_file', help='Phone map file')
    parser.add_argument('phn_file_dir', help='Location of all phone files')
    parser.add_argument('--nfilters', type=int, default=30,help='number of filters (30)')
    parser.add_argument('--fduration', type=float, default=0.02,help='Window length (0.02 sec)')
    parser.add_argument('--frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--context', type=int, help='Frame Context')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of points of computing FFT')
    parser.add_argument('--get_phone_labels', action='store_true',
                        help='get phone labels for each feature attached as another column to feature matrix(True)')
    parser.add_argument('--add_reverb', action='store_true',
                        help='Add reverberation to speech(True)')
    args = parser.parse_args()
    
    start_time=time.time()
    extractMelEnergyFeats(args.scp, args.outfile, args.phn_file, args.phn_file_dir, args.fduration, args.context,  args.frate,
                            args.nfft, args.nfilters, args.get_phone_labels, args.add_reverb)
    time_note='Execution Time: {t:.3f} seconds'.format(t=time.time()-start_time)
    print(time_note)
    sys.stdout.flush()
