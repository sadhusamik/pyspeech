#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 10:35:56 2018

@author: samik sadhu
"""

' Generate Mel Filter Energy features '

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
from os.path import isfile, join, abspath

def extractMelEnergyFeats(args, srate=16000,
                            window=np.hamming):
    """ Extract the mel scale filter-bank energy features
    """
    
    wavs=args.scp
    outfile=args.outfile
    phone_map=args.phn_file
    phn_file_dir=args.phn_file_dir
    get_phone_labels=args.get_phone_labels
    add_reverb=args.add_reverb
    nfft=args.nfft
    context=args.context
    fduration=args.fduration
    frate=args.frate
    nfilters=args.nfilters
    
    
    fbank = createFbank(nfilters, nfft, srate)
    
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
            uttid_base=uttid.split('_')[0]
            fname_phn_base=uttid_base+'.PHN'
            
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
                    
                melEnergy_frames=np.log10(np.matmul(np.abs(fft(time_frames,int(nfft/2+1),axis=1)),np.transpose(fbank)))
                
                if args.context:
                    melEnergy_frames=spliceFeats(melEnergy_frames,context)
                    
                if get_phone_labels:
                    if args.context:
                        feats=np.empty(nfilters*(2*context+1)+1)
                    else:
                        feats=np.empty(nfilters+1)
                else:
                    if args.context:
                        feats=np.empty(nfilters)
                    else:
                        feats=np.empty(nfilters*(2*context+1))
                
                for num, phn in enumerate(phone_now):
                    
                    now_frames=melEnergy_frames[int(phone_beg[num]):int(phone_end[num]),:]
                    if get_phone_labels:
                        ind=phn_list.index(phn)
                        fr_num=now_frames.shape[0]
                        now_frames=np.concatenate((now_frames,np.tile(ind,(fr_num,1))),axis=1)
                        
                    feats=np.vstack([feats, now_frames])
                feats=feats[1:,:]
                all_feats[uttid]=feats
                
        outfile=abspath(outfile); pkl_file=outfile+'.pkl'       
        pickle.dump(all_feats,open(pkl_file,'wb'))
        with open(outfile+'.scp', 'w+') as file:
            for item in list(all_feats.keys()):
                file.write("%s %s\n" % (item,pkl_file))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Mel Energy Features')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('phn_file_dir', help='Location of all phone files')
    parser.add_argument('--phn_file', help='Phone map file')
    parser.add_argument('--nfilters', type=int, default=30,help='number of filters (30)')
    parser.add_argument('--fduration', type=float, default=0.02,help='Window length (0.02 sec)')
    parser.add_argument('--frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--context', type=int, help='Frame Context')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of points of computing FFT')
    parser.add_argument('--get_phone_labels', action='store_true', help='get phone labels for each feature attached as another column to feature matrix(True)')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    args = parser.parse_args()
    
    extractMelEnergyFeats(args)