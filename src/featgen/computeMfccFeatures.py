#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:37:30 2018

@author: samiksadhu
"""

' Compute MFCC features '

import sys 
import argparse
import io
sys.path.append('../../tools')

import numpy as np
from features import getFrames,createFbank, spliceFeats, addReverb
from scipy.fftpack import fft, dct
from scipy.io.wavfile import read
import subprocess

import os


def getKaldiArk(feat_dict,outfile,kaldi_cmd):
    with open(outfile+'.txt','w+') as file:
        for key, feat in feat_dict.items():
            np.savetxt(file,feat,fmt='%.3f',header=key+' [',footer=' ]',comments='')
    cmd=kaldi_cmd+' ark,t:'+outfile+'.txt'+' ark,scp:'+outfile+'.ark,'+outfile+'.scp'
    subprocess.run(cmd, shell=True)
    os.remove(outfile+'.txt')     
            
def extractMelEnergyFeats(args, srate=16000,
                            window=np.hamming):
    """ Extract the mel scale filter-bank energy features
    """
    
    wavs=args.scp
    outfile=args.outfile
    add_reverb=args.add_reverb
    nfft=args.nfft
    context=args.context
    fduration=args.fduration
    frate=args.frate
    nfilters=args.nfilters
    kaldi_cmd=args.kaldi_cmd
    
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
                proc = subprocess.run(inwav[:-1], shell=True,stdout=subprocess.PIPE)
                if inwav[0:6]=='ffmpeg':
                    riff_chunk_size=len(proc.stdout)-8
                    q=riff_chunk_size
                    b=[]
                    for i in range(4):
                        q, r = divmod(q,256)
                        b.append(r)

                    riff=proc.stdout[:4]+bytes(b)+proc.stdout[8:]
                    sr, signal = read(io.BytesIO(riff))
                else:
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
            
        getKaldiArk(all_feats,outfile,kaldi_cmd)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Mel Energy Features')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('--nfilters', type=int, default=30,help='number of filters (30)')
    parser.add_argument('--fduration', type=float, default=0.02,help='Window length (0.02 sec)')
    parser.add_argument('--frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--context', type=int, help='Frame Context')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of points of computing FFT')
    parser.add_argument('--add_reverb', help='input "clean" OR "small_room" OR "large_room"')
    parser.add_argument('--kaldi_cmd', help='Kaldi command to use to get ark files')
    args = parser.parse_args()
    
    extractMelEnergyFeats(args)
