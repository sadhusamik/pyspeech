#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:31:21 2018

@author: samiksadhu
"""

'Compute MFCC + Modulation Spectral features'



import numpy as np 
import argparse
from features import getFrames,createFbank,computeLpcFast,computeModSpecFromLpc, addReverb, spliceFeats, dict2Ark
import sys
import time
import io 
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis
from scipy.fftpack import fft, dct

def get_modspec(args, srate=16000, window=np.hanning):
    
    wavs=args.scp
    add_reverb=args.add_reverb
    set_unity_gain=args.set_unity_gain
    nmodulations=args.nmodulations
    order=args.order
    fduration=args.fduration_modspec
    frate=args.frate
    nfilters=args.nfilters
    
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
                
            time_frames = np.array([frame for frame in
                getFrames(signal, srate, frate, fduration, window)])

            cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
            
            [frame_num, ndct]=np.shape(cos_trans)
            
            if set_unity_gain:
                feats=np.zeros((frame_num,nfilters*(nmodulations-1)))
            else:
                feats=np.zeros((frame_num,nfilters*nmodulations))
                
            print('%s: Computing Features for file: %s' % (sys.argv[0],uttid))
            sys.stdout.flush()
            for i in range(frame_num):
                if set_unity_gain:
                    each_feat=np.zeros([nfilters,nmodulations-1])
                else:
                    each_feat=np.zeros([nfilters,nmodulations])
                for j in range(nfilters):
                    filt=fbank[j,0:-1]
                    band_dct=filt*cos_trans[i,:]
                    xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                    if set_unity_gain:
                        gg=1
                    mod_spec=computeModSpecFromLpc(gg,xlpc,nmodulations)
                    if set_unity_gain:
                        mod_spec=mod_spec[1:]
                    each_feat[j,:]=mod_spec
                if set_unity_gain:
                    each_feat=np.reshape(each_feat,(1,nfilters*(nmodulations-1)))
                else:
                    each_feat=np.reshape(each_feat,(1,nfilters*nmodulations))
                feats[i,:]=each_feat
        
            all_feats[uttid]=feats

    return all_feats

def get_mfcc(args,srate=16000, window=np.hamming):

    wavs=args.scp
    add_reverb=args.add_reverb
    nfft=args.nfft
    context=args.context
    fduration=args.fduration_mfcc
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
    
    with open(wavs, 'r') as fid:

        all_feats={}
        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])
            
            print('%s: Computing Features for file: %s' % (sys.argv[0],uttid))
            sys.stdout.flush()
         
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
            mfcc_feats=dct(melEnergy_frames,axis=1)
            mfcc_feats=mfcc_feats[:,0:13]
            
            if args.context:
                mfcc_feats=spliceFeats(mfcc_feats,context)
                
            all_feats[uttid]=mfcc_feats
    return all_feats

if __name__=='__main__':
    parser=argparse.ArgumentParser('Computing MFCC + Modulation spectral features')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('--nfilters', type=int, default=15,help='number of filters (15)')
    parser.add_argument('--nmodulations', type=int, default=12,help='number of modulations of the modulation spectrum (12)')
    parser.add_argument('--order', type=int, default=50,help='LPC filter order (50)')
    parser.add_argument('--fduration_mfcc', type=float, default=0.5,help='Window length (0.5 sec)')
    parser.add_argument('--fduration_modspec', type=float, default=0.5,help='Window length (0.5 sec)')
    parser.add_argument('--frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--context', type=int, help='Frame Context for MFCC')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of points of computing FFT')
    parser.add_argument('--set_unity_gain', action='store_true', help='Set LPC gain to 1 (True)')
    parser.add_argument('--kaldi_cmd', help='Kaldi command to use to get ark files')
    
    args=parser.parse_args()
    
    start_time=time.time()
    print('%s: Computing MFCC features' % sys.argv[0])
    sys.stdout.flush()
    
    all_mfcc=get_mfcc(args)
    
    print('%s: Computing Modulation Spectral features' % sys.argv[0])
    sys.stdout.flush()
    
    all_modspec=get_modspec(args)
    
    print('%s: Combining Modulation Spectral features' % sys.argv[0])
    sys.stdout.flush()
    
    all_feats={}
    for uttid in list(all_mfcc.keys()):
        mfcc=all_mfcc[uttid]
        modspec=all_modspec[uttid]
        all_feats[uttid]=np.concatenate((modspec,mfcc), axis=1)
    
    dict2Ark(all_feats,args.outfile,args.kaldi_cmd)
    time_note='Execution Time: {t:.3f} seconds'.format(t=time.time()-start_time)  
    print(time_note)
    sys.stdout.flush()